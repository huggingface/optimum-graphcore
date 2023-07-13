# Copyright 2021 The Fairseq Authors and the HuggingFace Inc. team. All rights reserved.
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

from typing import Optional, Tuple, Union

import poptorch
import torch
from transformers import HubertForCTC, HubertForSequenceClassification
from transformers.modeling_outputs import CausalLMOutput
from transformers.models.hubert.modeling_hubert import HubertEncoder, HubertEncoderStableLayerNorm

from optimum.utils import logging

from ...modeling_utils import PipelineMixin, get_layer_ipu, recomputation_checkpoint, register
from .ipu_layer_drop import IPUHubertEncoder, IPUHubertEncoderStableLayerNorm


logger = logging.get_logger(__name__)


@register(HubertForSequenceClassification)
class PipelinedHubertForSequenceClassification(HubertForSequenceClassification, PipelineMixin):
    def change_hubert_encoder_class(self, restore: bool):
        """Changes the encoder class to update its forward pass so that it uses our custom version.

        Args:
            restore: whether to restore the encoder to its original version or not.
        """
        if self.config.do_stable_layer_norm:
            new_cls = HubertEncoderStableLayerNorm if restore else IPUHubertEncoderStableLayerNorm
        else:
            new_cls = HubertEncoder if restore else IPUHubertEncoder
        self.hubert.encoder.__class__ = new_cls

    def parallelize(self):
        super().parallelize()

        self.change_hubert_encoder_class(False)

        self.hubert.feature_extractor = poptorch.BeginBlock(self.hubert.feature_extractor, ipu_id=0)
        self.hubert.feature_projection = poptorch.BeginBlock(self.hubert.feature_projection, ipu_id=0)
        self.hubert.encoder = poptorch.BeginBlock(self.hubert.encoder, ipu_id=0)

        layer_ipu = get_layer_ipu(self.ipu_config, self.hubert.encoder.layers)
        for index, layer in enumerate(self.hubert.encoder.layers):
            # Put checkpoints on every encoder layer
            h = recomputation_checkpoint(layer)
            self._hooks.append(h)
            ipu = layer_ipu[index]
            self.hubert.encoder.layers[index] = poptorch.BeginBlock(layer, f"Encoder{index}", ipu_id=ipu)

        last_ipu = self.ipu_config._ipus_per_replica - 1
        self.projector = poptorch.BeginBlock(self.projector, ipu_id=last_ipu)
        self.classifier = poptorch.BeginBlock(self.classifier, ipu_id=last_ipu)
        return self

    def deparallelize(self):
        """
        Undo the changes to the model done by `parallelize`.
        """
        super().deparallelize()
        self.change_hubert_encoder_class(True)
        return self


@register(HubertForCTC)
class PipelinedHubertCTC(HubertForCTC, PipelineMixin):
    def change_hubert_encoder_class(self, restore: bool):
        """Changes the encoder class to update its forward pass so that it uses our custom version.

        Args:
            restore: whether to restore the encoder to its original version or not.
        """
        if self.config.do_stable_layer_norm:
            new_cls = HubertEncoderStableLayerNorm if restore else IPUHubertEncoderStableLayerNorm
        else:
            new_cls = HubertEncoder if restore else IPUHubertEncoder
        self.hubert.encoder.__class__ = new_cls

    def _add_begin_block(self, module, name, ipu_id):
        poptorch.BeginBlock(module, name, ipu_id)

    def parallelize(self):
        super().parallelize()

        self.freeze_feature_encoder()
        self.change_hubert_encoder_class(False)

        if self.ipu_config._ipus_per_replica != 1:
            logger.info("---------- Device Allocation -----------")
            layers = []
            # Conv layers
            for index, layer in enumerate(self.hubert.feature_extractor.conv_layers):
                layers.append((f"Conv {index:<2}", layer))
            # Positional Embedding
            layers.append(("Positional Embedding", self.hubert.encoder.pos_conv_embed))
            # Encoder layers
            for index, layer in enumerate(self.hubert.encoder.layers):
                self._hooks.append(recomputation_checkpoint(layer))
                layers.append((f"Encoder {index:<2}", layer))
            # Project Hidden
            layers.append(("Project Hidden", self.lm_head))

            layer_ipu = get_layer_ipu(self.ipu_config, layers)

            for i, (name, layer) in enumerate(layers):
                logger.info(f"{name} --> IPU {layer_ipu[i]}")
                self._add_begin_block(layer, name, ipu_id=layer_ipu[i])

            logger.info("---------------------------------------")
        return self

    def deparallelize(self):
        """
        Undo the changes to the model done by `parallelize`.
        """
        super().deparallelize()
        self.change_hubert_encoder_class(True)
        return self

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.hubert(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            # flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1).transpose(0, 1)

            loss_fn = torch.nn.CTCLoss(
                blank=self.config.pad_token_id,
                reduction=self.config.ctc_loss_reduction,
                zero_infinity=self.config.ctc_zero_infinity,
            )
            loss = loss_fn(log_probs, labels, input_lengths, target_lengths)
            loss = poptorch.identity_loss(loss, "none")

        if not return_dict:
            if loss is not None:
                return loss, logits
            return (logits, hidden_states)
        return CausalLMOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states)
