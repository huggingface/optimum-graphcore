# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn

import poptorch
import numpy as np
from optimum.utils import logging
from transformers import (
    HubertForSequenceClassification,
    HubertPreTrainedModel,
)
from transformers.models.hubert.modeling_hubert import (
    HubertForSequenceClassification,
    HubertEncoder,
    BaseModelOutput,
    HubertSamePadLayer,
    ACT2FN,
)

from ...modeling_utils import PipelineMixin, register


logger = logging.get_logger(__name__)


def recomputation_checkpoint(module: torch.nn.Module) -> torch.utils.hooks.RemovableHandle:
    """Annotates the output of a module to be checkpointed instead of
    recomputed"""

    def recompute_outputs(module, inputs, outputs):
        if type(outputs) is tuple:
            return tuple(poptorch.recomputationCheckpoint(y) for y in outputs)
        else:
            return poptorch.recomputationCheckpoint(outputs)

    return module.register_forward_hook(recompute_outputs)


def _get_layer_ipu(layers_per_ipu):
    # List of the IPU Id for each encoder layer
    layer_ipu = []
    for ipu, n_layers in enumerate(layers_per_ipu):
        layer_ipu += [ipu] * n_layers
    return layer_ipu


class IPUHubertEncoder(HubertEncoder):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            # make sure padded tokens output 0
            # hidden_states[~attention_mask] = 0.0
            print(hidden_states.shape)
            print(attention_mask.shape)
            hidden_states.masked_fill(~attention_mask.unsqueeze(-1), 0.0)

            # extend attention_mask
            attention_mask = (1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = np.random.uniform(0, 1)

            skip_the_layer = True if self.training and (dropout_probability < self.config.layerdrop) else False
            if not skip_the_layer:
                layer_outputs = layer(
                    hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
                )
                hidden_states = layer_outputs[0]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


def _get_feature_vector_attention_mask(self, feature_vector_length: int, attention_mask: torch.LongTensor):
    output_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)
    batch_size = attention_mask.shape[0]
    print("HERE")

    attention_mask = torch.zeros(
        (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
    )
    # these two operations makes sure that all values before the output lengths idxs are attended to
    attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
    attention_mask = attention_mask.flip(1).cumsum(-1).flip(1).bool()
    return attention_mask


HubertPreTrainedModel._get_feature_vector_attention_mask = _get_feature_vector_attention_mask


class IPUHubertPositionalConvEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            padding=config.num_conv_pos_embeddings // 2,
            groups=config.num_conv_pos_embedding_groups,
        )

        # Weight norm not supported atm!
        # self.conv = nn.utils.weight_norm(self.conv, name="weight", dim=2)

        self.padding = HubertSamePadLayer(config.num_conv_pos_embeddings)
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = self.conv(hidden_states)
        hidden_states = self.padding(hidden_states)
        hidden_states = self.activation(hidden_states)

        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


@register(HubertForSequenceClassification)
class PipelinedHubertForSequenceClassification(HubertForSequenceClassification, PipelineMixin):
    def __init__(self, config):
        super().__init__(config)
        self.hubert.encoder = IPUHubertEncoder(config)
        # Weight norm not supported atm
        self.hubert.encoder.pos_conv_embed = IPUHubertPositionalConvEmbedding(config)

    def parallelize(self):
        self._hooks = []
        self.hubert.feature_extractor = poptorch.BeginBlock(self.hubert.feature_extractor, ipu_id=0)
        self.hubert.feature_projection = poptorch.BeginBlock(self.hubert.feature_projection, ipu_id=0)
        self.hubert.encoder.pos_conv_embed = poptorch.BeginBlock(self.hubert.encoder.pos_conv_embed, ipu_id=0)
        self.hubert.encoder.layer_norm = poptorch.BeginBlock(self.hubert.encoder.layer_norm, ipu_id=0)
        self.hubert.encoder.dropout = poptorch.BeginBlock(self.hubert.encoder.dropout, ipu_id=0)

        # for layer in self.hubert.feature_extractor.conv_layers[2:]:
        #     h = recomputation_checkpoint(layer)
        #     self._hooks.append(h)

        # h = recomputation_checkpoint(self.hubert.feature_projection)
        # self._hooks.append(h)

        # h = recomputation_checkpoint(self.hubert.encoder.pos_conv_embed)
        # self._hooks.append(h)

        layer_ipu = _get_layer_ipu(layers_per_ipu)
        for index, layer in enumerate(self.hubert.encoder.layers):
            # Put checkpoints on every encoder layer
            # h = recomputation_checkpoint(layer)
            # self._hooks.append(h)
            ipu = layer_ipu[index]
            self.hubert.encoder.layers[index] = poptorch.BeginBlock(layer, f"Encoder{index}", ipu_id=ipu)

        self.projector = poptorch.BeginBlock(self.projector, ipu_id=3)
        self.classifier = poptorch.BeginBlock(self.classifier, ipu_id=3)
        return self

    def forward(
        self,
        input_values,
        attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        outputs = self.hubert(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=False,
        )

        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[1]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = outputs[0]

        hidden_states = self.projector(hidden_states)
        if attention_mask is None:
            pooled_output = hidden_states.mean(dim=1)
        else:
            padding_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
            hidden_states.masked_fill_(~padding_mask.unsqueeze(-1), 0.0)
            pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        output = (logits,) + outputs[1:]
        return ((loss,) + output) if loss is not None else output
