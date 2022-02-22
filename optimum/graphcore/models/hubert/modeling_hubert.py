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
from optimum.utils import logging
from transformers import (
    HubertForSequenceClassification,
)
from transformers.models.hubert.modeling_hubert import (
    HubertForSequenceClassification,
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


@register(HubertForSequenceClassification)
class PipelinedHubertForSequenceClassification(HubertForSequenceClassification, PipelineMixin):
    def parallelize(self):
        super().parallelize()
        self.hubert.feature_extractor = poptorch.BeginBlock(self.hubert.feature_extractor, ipu_id=0)
        self.hubert.feature_projection = poptorch.BeginBlock(self.hubert.feature_projection, ipu_id=0)
        self.hubert.encoder = poptorch.BeginBlock(self.hubert.encoder, ipu_id=0)

        # for layer in self.hubert.feature_extractor.conv_layers[2:]:
        #     h = recomputation_checkpoint(layer)
        #     self._hooks.append(h)

        # h = recomputation_checkpoint(self.hubert.feature_projection)
        # self._hooks.append(h)

        # h = recomputation_checkpoint(self.hubert.encoder.pos_conv_embed)
        # self._hooks.append(h)

        layer_ipu = _get_layer_ipu(self.config.layers_per_ipu)
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
        labels=None,
        attention_mask=None,
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
            hidden_states[~padding_mask] = 0.0
            pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        output = (logits,)  # + outputs[1:]
        return ((loss,) + output) if loss is not None else output
