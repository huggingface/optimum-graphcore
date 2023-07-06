# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
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

import math
from typing import Tuple

import poptorch
import torch
import torch.nn as nn
from transformers.models.mpnet.modeling_mpnet import MPNetForMaskedLM, MPNetModel, MPNetSelfAttention

from optimum.utils import logging

from ...modeling_utils import (
    PipelineMixin,
    SerializedLinear,
    get_layer_ipu,
    outline_attribute,
    recomputation_checkpoint,
    register,
)


logger = logging.get_logger(__name__)


class MPNetFusedSelfAttention(MPNetSelfAttention):
    def fused_qkv(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weights = (self.q.weight, self.k.weight, self.v.weight)
        combined_weight = torch.cat(weights, dim=0)
        combined_result = hidden_state @ torch.transpose(combined_weight, -2, -1)
        biases = (self.q.bias, self.k.bias, self.v.bias)
        if all((b is not None for b in biases)):
            combined_bias = torch.cat(biases, dim=0)
            combined_result += combined_bias
        elif any((b is not None for b in biases)):
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
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        position_bias=None,
        output_attentions=False,
    ):
        # --- Change: Use fused matmul implementation ---
        q, k, v = self.fused_qkv(hidden_states)

        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(q, k.transpose(-1, -2))

        # --- Change: Use reciprocal multiply for speed ---
        attention_scores = attention_scores * (1.0 / math.sqrt(self.attention_head_size))

        # Apply relative position embedding (precomputed in MPNetEncoder) if provided.
        if position_bias is not None:
            attention_scores += position_bias

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        c = torch.matmul(attention_probs, v)

        c = c.permute(0, 2, 1, 3).contiguous()
        new_c_shape = c.size()[:-2] + (self.all_head_size,)
        c = c.view(*new_c_shape)

        o = self.o(c)

        outputs = (o, attention_probs) if output_attentions else (o,)

        return outputs


@register(MPNetModel)
class PipelinedMPNetModel(MPNetModel, PipelineMixin):
    def __init__(self, config):
        super().__init__(config)

    def parallelize(self):
        super().parallelize()

        for layer in self.encoder.layer:
            layer.attention.attn.__class__ = MPNetFusedSelfAttention

        logger.info("-------------------- Device Allocation --------------------")
        logger.info("Embedding --> IPU 0")

        self.embeddings = poptorch.BeginBlock(self.embeddings, "Embedding", ipu_id=0)
        hs = outline_attribute(self.embeddings.LayerNorm, "embedding")
        self._hooks.extend(hs)

        layer_ipu = get_layer_ipu(self.ipu_config, self.encoder.layer)
        for index, layer in enumerate(self.encoder.layer):
            ipu = layer_ipu[index]
            if self.ipu_config.recompute_checkpoint_every_layer and index != self.config.num_hidden_layers - 1:
                h = recomputation_checkpoint(layer)
                self._hooks.append(h)
            self.encoder.layer[index] = poptorch.BeginBlock(layer, f"Encoder{index}", ipu_id=ipu)
            logger.info(f"Encoder {index:<2} --> IPU {ipu}")

        logger.info("Pooler --> IPU 0")
        self.pooler = poptorch.BeginBlock(self.pooler, "Pooler", ipu_id=0)

        return self

    def deparallelize(self):
        """
        Undo the changes to the model done by `parallelize`.
        You should call this before doing `save_pretrained` so that the `model.state_dict` is
        compatible with the original model.
        """
        super().deparallelize()

        for layer in self.encoder.layer:
            layer.attention.attn.__class__ = MPNetSelfAttention

        return self


@register(MPNetForMaskedLM)
class PipelinedMPNetForMaskedLM(MPNetForMaskedLM, PipelineMixin):
    def __init__(self, config):
        super().__init__(config)
        self.mpnet = MPNetModel(config, add_pooling_layer=False)

    def parallelize(self):
        super().parallelize()

        for layer in self.mpnet.encoder.layer:
            layer.attention.attn.__class__ = MPNetFusedSelfAttention

        # Serialise the prediction head decoder
        if self.ipu_config.embedding_serialization_factor > 1:
            self.lm_head.decoder = SerializedLinear.from_model(
                self.lm_head.decoder, self.ipu_config.embedding_serialization_factor
            )
            self.tie_weights()

        logger.info("-------------------- Device Allocation --------------------")
        logger.info("Embedding --> IPU 0")

        self.mpnet.embeddings = poptorch.BeginBlock(self.mpnet.embeddings, "Embedding", ipu_id=0)

        # Preventing the embeddings.LayerNorm from being outlined with the encoder.layer.LayerNorm
        # improves the tile mapping of the pipeline stashes
        hs = outline_attribute(self.mpnet.embeddings.LayerNorm, "embedding")
        self._hooks.extend(hs)

        layer_ipu = get_layer_ipu(self.ipu_config, self.mpnet.encoder.layer)
        for index, layer in enumerate(self.mpnet.encoder.layer):
            ipu = layer_ipu[index]
            if self.ipu_config.recompute_checkpoint_every_layer and index != self.config.num_hidden_layers - 1:
                h = recomputation_checkpoint(layer)
                self._hooks.append(h)
            self.mpnet.encoder.layer[index] = poptorch.BeginBlock(layer, f"Encoder{index}", ipu_id=ipu)
            logger.info(f"Encoder {index:<2} --> IPU {ipu}")

        logger.info("Classifier --> IPU 0")
        self.lm_head = poptorch.BeginBlock(self.lm_head, "Classifier", ipu_id=0)
        logger.info("-----------------------------------------------------------")

        return self

    def deparallelize(self):
        """
        Undo the changes to the model done by `parallelize`.
        You should call this before doing `save_pretrained` so that the `model.state_dict` is
        compatible with the original model.
        """
        super().deparallelize()

        for layer in self.mpnet.encoder.layer:
            layer.attention.attn.__class__ = MPNetSelfAttention

        if isinstance(self.lm_head.decoder, SerializedLinear):
            self.lm_head.decoder = self.lm_head.decoder.to_model()
            self.tie_weights()

        return self
