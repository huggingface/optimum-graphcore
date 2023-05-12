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
import copy
from typing import Optional, Tuple

import torch
from torch import nn

import poptorch
from optimum.utils import logging
from transformers import WhisperConfig
from transformers.models.whisper.modeling_whisper import (
    WhisperAttention,
    WhisperDecoder,
    WhisperEncoderLayer,
    WhisperForConditionalGeneration,
    WhisperPositionalEmbedding,
)

from ...generation import IPUAttentionMixin, IPUGenerationMixin, supports_kv_cache
from ...modeling_utils import (
    PipelineMixin,
    get_layer_ipu,
    recomputation_checkpoint,
    register,
    split_encoder_decoder_ipu_config,
)


logger = logging.get_logger(__name__)

from transformers.activations import ACT2FN


FLOAT16_LIMIT = 1e4


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), -FLOAT16_LIMIT)
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), -FLOAT16_LIMIT)


class IPUWhisperAttention(WhisperAttention, IPUAttentionMixin):
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, tgt_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states) * self.scaling
        if key_value_states is not None:
            # cross attention
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif self.kv_cache_initialised:
            # self attention with kv cache
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

            if tgt_len != 1:
                raise ValueError(f"KV cache expects tgt_len = 1, received {tgt_len}.")

            key_states, value_states = self.add_to_kv_cache(key_states, value_states)
            attention_mask = self.update_attention_mask(attention_mask)
        else:
            # self attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # We handle the KV cache via buffers, not via the eager approach of passing the cache around.
            # This is retained so upstream DecoderLayer can stay as is and that tests pass.
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)

        # Change: optionally serialize attention, mainly intended for the encoder with large sequence length.
        if self.is_attention_serialized:
            if layer_head_mask is not None:
                raise ValueError("layer_head_mask is not supported yet with serialized attention.")

            if self.dropout or self.training:
                raise ValueError("dropout is not supported yet with serialized attention.")

            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                    )
                attention_mask = attention_mask.view(bsz, tgt_len, src_len).repeat(self.num_heads, 1, 1)

            attn_output = self.serialized_attention(query_states, key_states, value_states, 1.0, attention_mask)
        else:
            attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

            if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
                attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

            if layer_head_mask is not None:
                if layer_head_mask.size() != (self.num_heads,):
                    raise ValueError(
                        f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                        f" {layer_head_mask.size()}"
                    )
                attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(
                    bsz, self.num_heads, tgt_len, src_len
                )
                attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

            # Change: delete optional reshaping of attn_weights

            attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

            attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        # Change: modified check for output_attentions
        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


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

        # NOTE: This differs from the original implementation
        # There is a type mismatch bug with this call to clamp so we remove it here. It is anyway not needed on IPU because FP16 values are clamped to maximum value by default.
        # TODO: when bug is fixed in future SDK remove this entire class;

        # clamp_value = torch.finfo(hidden_states.dtype).max - 1000
        # hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class IPUWhisperPositionalEmbedding(WhisperPositionalEmbedding):
    @classmethod
    def from_model(cls, model: WhisperPositionalEmbedding):
        clone = copy.deepcopy(model)
        clone.__class__ = cls
        clone.register_buffer("_generation_step", torch.tensor([0], dtype=torch.int32), persistent=False)
        return clone

    def to_model(self) -> WhisperPositionalEmbedding:
        del self._generation_step

        original = copy.deepcopy(self)
        original.__class__ = WhisperPositionalEmbedding
        return original

    def forward(self, input_ids: torch.Tensor, past_key_values_length: int = 0):
        del past_key_values_length

        if input_ids.shape[-1] == 1:
            # KV cache enabled.
            return poptorch.dynamic_slice(self.weight, 0, self._generation_step, 1, 1)
        else:
            return self.weight[: input_ids.shape[-1]]


class _WhisperDecoderWithCustomMakeCausalAndExpandMask(WhisperDecoder):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`WhisperDecoderLayer`]

    Args:
        config: WhisperConfig
    """

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None

        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(inputs_embeds.device)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask


@supports_kv_cache
@register(WhisperForConditionalGeneration)
class PipelinedWhisperForConditionalGeneration(WhisperForConditionalGeneration, PipelineMixin, IPUGenerationMixin):
    def change_encoder_layer_class(self, restore: bool):
        """Changes the encoder layer class to avoid the dynamic 'if'

        Args:
            restore: whether to restore the encoder layers to their original version or not.
        """
        for layer in self.model.encoder.layers:
            layer.__class__ = WhisperEncoderLayer if restore else _WhisperEncoderLayerClamp

    def change_decoder_class(self, restore: bool):
        """Changes the decoder class to update the _prepare_decoder_attention_mask method.

        Args:
            restore: whether to restore the decoder to its original version or not.
        """
        self.model.decoder.__class__ = WhisperDecoder if restore else _WhisperDecoderWithCustomMakeCausalAndExpandMask

    def change_decoder_positional_embedding(self, restore: bool):
        """Changes the decoder positional embedding to support an optional static KV cache.

        Args:
            restore: whether to restore the decoder positional embedding to their original version or not.
        """
        position_embedding = self.model.decoder.embed_positions
        self.model.decoder.embed_positions = (
            position_embedding.to_model() if restore else IPUWhisperPositionalEmbedding.from_model(position_embedding)
        )

    def change_attention_class(self, restore=False, **kwargs):
        """Change the attention layers to support a KV cache.

        Args:
            restore: whether to restore the attention layers to their original version or not.
        """
        use_cache = kwargs.get("use_cache", False)
        batch_size = kwargs.get("batch_size", 1)
        max_length = kwargs.get("max_length", 448)
        num_beams = kwargs.get("num_beams", 1)
        batch_serialization_factor = kwargs.get("batch_serialization_factor", 1)
        sequence_serialization_factor = kwargs.get("sequence_serialization_factor", 1)

        for encoder_layer in self.model.encoder.layers:
            if restore:
                encoder_layer.self_attn = encoder_layer.self_attn.to_model(WhisperAttention)
                continue

            encoder_layer.self_attn = IPUWhisperAttention.from_model(
                encoder_layer.self_attn,
                use_cache=False,
                batch_serialization_factor=batch_serialization_factor,
                sequence_serialization_factor=sequence_serialization_factor,
            )

        for decoder_layer in self.model.decoder.layers:
            if restore:
                decoder_layer.self_attn = decoder_layer.self_attn.to_model(WhisperAttention)
                decoder_layer.encoder_attn = decoder_layer.encoder_attn.to_model(WhisperAttention)
                continue

            decoder_layer.self_attn = IPUWhisperAttention.from_model(
                decoder_layer.self_attn,
                use_cache=use_cache,
                batch_size=batch_size,
                max_length=max_length,
                num_beams=num_beams,
                dtype=decoder_layer.self_attn.k_proj.weight.dtype,
            )
            decoder_layer.encoder_attn = IPUWhisperAttention.from_model(
                decoder_layer.encoder_attn,
                use_cache=False,
            )

    def parallelize(self, for_generation=False, use_cache=False, **kwargs):
        super().parallelize()

        self.change_encoder_layer_class(restore=False)
        self.change_decoder_class(restore=False)
        self.change_decoder_positional_embedding(restore=False)
        self.change_attention_class(restore=False, use_cache=use_cache and for_generation, **kwargs)
        self.change_lm_head_to_indexed_input_linear(restore=use_cache or not for_generation)
        self.use_encoder_output_buffer = kwargs.get("use_encoder_output_buffer", False)
        self.set_on_device_generation_steps(kwargs.get("on_device_generation_steps", 0))

        logger.info("---------- Device Allocation -----------")
        logger.info("conv1, conv2, embed_positions  --> IPU 0")
        self.model.encoder.conv1 = poptorch.BeginBlock(self.model.encoder.conv1, "Conv1", ipu_id=0)
        self.model.encoder.conv2 = poptorch.BeginBlock(self.model.encoder.conv2, "Conv2", ipu_id=0)
        self.model.encoder.embed_positions = poptorch.BeginBlock(
            self.model.encoder.embed_positions, "Embed Positions", ipu_id=0
        )

        num_encoder_layers = len(self.model.encoder.layers)
        num_decoder_layers = len(self.model.decoder.layers)

        if for_generation:
            # If running for text generation we split the IPU config into two configs
            # because we run the encoder and decoder as separate Poplar executors.
            ipu_configs = split_encoder_decoder_ipu_config(self.ipu_config, num_encoder_layers, num_decoder_layers)
            self.encoder_ipu_config, self.decoder_ipu_config = ipu_configs
            encoder_layer_ipu = get_layer_ipu(self.encoder_ipu_config, num_encoder_layers)
            decoder_layer_ipu = get_layer_ipu(self.decoder_ipu_config, num_decoder_layers)
        else:
            number_of_layers = num_encoder_layers + num_decoder_layers
            layer_ipu = get_layer_ipu(self.ipu_config, number_of_layers)
            encoder_layer_ipu = layer_ipu[:num_encoder_layers]
            decoder_layer_ipu = layer_ipu[num_encoder_layers:]

        for index, (layer, ipu) in enumerate(zip(self.model.encoder.layers, encoder_layer_ipu)):
            if self.ipu_config.recompute_checkpoint_every_layer and index != self.config.num_hidden_layers - 1:
                self._hooks.append(recomputation_checkpoint(layer))
            self.model.encoder.layers[index] = poptorch.BeginBlock(layer, f"Encoder{index}", ipu_id=ipu)
            logger.info(f"Encoder {index:<2} --> IPU {ipu}")

        # we need to deal with the model.encoder.layer norm
        self.model.encoder.layer_norm = poptorch.BeginBlock(
            self.model.encoder.layer_norm, "Encoder Layer Norm", ipu_id=ipu
        )
        logger.info(f"Encoder LN {index:<2} --> IPU {ipu}")

        for index, (layer, ipu) in enumerate(zip(self.model.decoder.layers, decoder_layer_ipu)):
            if self.ipu_config.recompute_checkpoint_every_layer and index != self.config.num_hidden_layers - 1:
                self._hooks.append(recomputation_checkpoint(layer))
            self.model.decoder.layers[index] = poptorch.BeginBlock(layer, f"Decoder{index}", ipu_id=ipu)
            logger.info(f"Decoder {index:<2} --> IPU {ipu}")

        self.model.decoder.layer_norm = poptorch.BeginBlock(
            self.model.decoder.layer_norm, "Decoder Layer Norm", ipu_id=ipu
        )

        logger.info(f"Head       --> IPU 0")
        logger.info("---------------------------------------")
        self.proj_out = poptorch.BeginBlock(self.proj_out, "Output Projection", ipu_id=0)
        return self

    def deparallelize(self):
        super().deparallelize()

        self.change_encoder_layer_class(restore=True)
        self.change_decoder_class(restore=True)
        self.change_decoder_positional_embedding(restore=True)
        self.change_attention_class(restore=True)
        self.change_lm_head_to_indexed_input_linear(restore=True)
        self.set_on_device_generation_steps(0)

    def prepare_inputs_for_generation(
        self, decoder_input_ids, past=None, use_cache=None, encoder_outputs=None, attention_mask=None, **kwargs
    ):
        # We don't use `past` for KV caching, and rely on `use_cache` instead.
        beam_idx = None
        if use_cache:
            decoder_input_ids = decoder_input_ids[:, -1:]
            beam_idx = kwargs.get("beam_idx", torch.arange(decoder_input_ids.shape[0], dtype=torch.long))

        return {
            "encoder_outputs": encoder_outputs,
            "past_key_values": None,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": None,
            "beam_idx": beam_idx,
        }
