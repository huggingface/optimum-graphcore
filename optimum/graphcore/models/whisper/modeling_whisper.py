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
import torch
import transformers
from optimum.utils import logging

from ...modeling_utils import PipelineMixin, get_layer_ipu, recomputation_checkpoint, register

from torch import nn

from transformers.models.whisper.modeling_whisper import WhisperEncoderLayer
from transformers import WhisperConfig


logger = logging.get_logger(__name__)

from transformers.activations import ACT2FN

class _WhisperEncoderLayerClamp(nn.Module):
    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = WhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # NOTE: Never clamp. This differs from the original implementation
        # With the code below, compilation fails with an unhelpful error
        # 023-02-08T10:24:33.204593Z popart:popart 3749453.3749453 E: np broadcasting failed on Op 197 (ai.onnx.Add:7), incompatible types FLOAT and FLOAT16 (shapes [1500 384] and [384]) 

        # clamp_value = torch.finfo(hidden_states.dtype).max - 1000
        # hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs  


@register(transformers.WhisperForConditionalGeneration)
class PipelinedWhisperForConditionalGeneration(transformers.WhisperForConditionalGeneration, PipelineMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def change_encoder_layer_class(self, restore: bool):
        """Changes the encoder layer class to avoid the dynamic 'if'

        Args:
            restore: whether to restore the encoder and decoder to their original version or not.
        """
        for layer in self.model.encoder.layers:
            layer.__class__ = WhisperEncoderLayer if restore else _WhisperEncoderLayerClamp  


    def parallelize(self):
        super().parallelize()

        self.change_encoder_layer_class(restore=False)

        logger.info("---------- Device Allocation -----------")
        logger.info("conv1, conv2, embed_positions  --> IPU 0")
        self.model.encoder.conv1 = poptorch.BeginBlock(self.model.encoder.conv1, "Conv1", ipu_id=0)
        self.model.encoder.conv2 = poptorch.BeginBlock(self.model.encoder.conv2, "Conv2", ipu_id=0)
        self.model.encoder.embed_positions = poptorch.BeginBlock(self.model.encoder.embed_positions, "Embed Positions", ipu_id=0)

        # From modeling_bart.py
        number_of_layers = len(self.model.encoder.layers) + len(self.model.decoder.layers)
        layer_ipu = get_layer_ipu(self.ipu_config.layers_per_ipu, number_of_layers)
        for index, layer in enumerate(self.model.encoder.layers):
            ipu = layer_ipu[index]
            if self.ipu_config.recompute_checkpoint_every_layer and index != self.config.num_hidden_layers - 1:
                recomputation_checkpoint(layer)
            self.model.encoder.layers[index] = poptorch.BeginBlock(layer, f"Encoder{index}", ipu_id=ipu)
            logger.info(f"Encoder {index:<2} --> IPU {ipu}")

        # we need to deal with the model.encoder.layer norm
        self.model.encoder.layer_norm = poptorch.BeginBlock(self.model.encoder.layer_norm, "Encoder Layer Norm", ipu_id=ipu+1)  ###


        shift = len(self.model.encoder.layers)
        for index, layer in enumerate(self.model.decoder.layers):
            ipu = layer_ipu[index + shift]
            if self.ipu_config.recompute_checkpoint_every_layer and index != self.config.num_hidden_layers - 1:
                recomputation_checkpoint(layer)
            self.model.decoder.layers[index] = poptorch.BeginBlock(layer, f"Decoder{index}", ipu_id=ipu)
            logger.info(f"Decoder {index:<2} --> IPU {ipu}")

        last_ipu = ipu

        logger.info(f"Head       --> IPU {last_ipu}")
        logger.info("---------------------------------------")
        self.model.decoder.layer_norm = poptorch.BeginBlock(self.model.decoder.layer_norm, "Decoder Layer Norm", ipu_id=last_ipu)
        self.proj_out = poptorch.BeginBlock(self.proj_out, "Output Projection", ipu_id=last_ipu)
        return self

    def forward(self, audio, decoder_input_ids):
        return super().forward(input_features=audio.squeeze(dim=0), decoder_input_ids=decoder_input_ids)

