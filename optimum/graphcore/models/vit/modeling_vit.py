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

import poptorch
import transformers
from optimum.utils import logging
from typing import Dict, List, Optional, Set, Tuple, Union
import torch
import torch.nn as nn
import math

from transformers import ViTConfig
from transformers.models.vit.modeling_vit import ViTSelfAttention
from ...modeling_utils import PipelineMixin, get_layer_ipu, recomputation_checkpoint, register


logger = logging.get_logger(__name__)


class IPUViTSelfAttention(ViTSelfAttention):
    def fused_qkv(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weights = (self.query.weight, self.key.weight, self.value.weight)
        combined_weight = torch.cat(weights, dim=0)
        combined_result = hidden_state @ torch.transpose(combined_weight, -2, -1)
        biases = (self.query.bias, self.key.bias, self.value.bias)
        if all(map(lambda b: b is not None, biases)):
            combined_bias = torch.cat(biases, dim=0)
            combined_result += combined_bias
        elif any(map(lambda b: b is not None, biases)):
            raise RuntimeError(
                "Some attention layers had biases but not all. This is not supported. "
                "Please enable biases on all Query, Key and Value or none. "
                f"query.bias = {biases[0] is not None}, "
                f"key.bias = {biases[1] is not None}, "
                f"value.bias = {biases[2] is not None}"
            )
        hidden_size = hidden_state.shape[-1]
        return torch.split(combined_result, hidden_size, dim=-1)

    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        mixed_query_layer, mixed_key_layer, mixed_value_layer = self.fused_qkv(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores * (1. / math.sqrt(self.attention_head_size))

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs

@register(transformers.ViTForImageClassification)
class PipelinedViTForImageClassification(transformers.ViTForImageClassification, PipelineMixin):
    def parallelize(self):
        super().parallelize()
        
        # Use faster fused-qkv self-attention
        for layer in self.vit.encoder.layer:
            layer.attention.attention.__class__ = IPUViTSelfAttention 

        logger.info("---------- Device Allocation -----------")
        logger.info("Embedding  --> IPU 0")
        self.vit.embeddings = poptorch.BeginBlock(self.vit.embeddings, "Embedding", ipu_id=0)

        layer_ipu = get_layer_ipu(self.ipu_config.layers_per_ipu)
        for index, layer in enumerate(self.vit.encoder.layer):
            if self.ipu_config.recompute_checkpoint_every_layer:
                # Put checkpoints on every encoder layer
                h = recomputation_checkpoint(layer)
                self._hooks.append(h)
            ipu = layer_ipu[index]
            logger.info(f"Encoder {index:<2} --> IPU {ipu}")
            self.vit.encoder.layer[index] = poptorch.BeginBlock(layer, f"Encoder{index}", ipu_id=ipu)

        logger.info("Head       --> IPU 3")
        logger.info("---------------------------------------")
        self.vit.layernorm = poptorch.BeginBlock(self.vit.layernorm, "LayerNorm", ipu_id=3)
        self.classifier = poptorch.BeginBlock(self.classifier, "Classifier", ipu_id=3)
        return self

    def forward(self, pixel_values, labels=None):
        return super().forward(pixel_values=pixel_values, labels=labels, return_dict=False)
