# Copyright 2022 The OpenAI Authors and The HuggingFace Inc. team. All rights reserved.
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
from typing import Optional, Tuple, Union

import poptorch
import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.models.whisper.modeling_whisper import (
    WhisperAttention,
    WhisperDecoder,
    WhisperEncoder,
    WhisperEncoderLayer,
    WhisperForConditionalGeneration,
    WhisperPositionalEmbedding,
)

from optimum.utils import logging

from ...generation import IPUAttentionMixin, IPUGenerationMixin, assert_poptorch_supports_cond, supports_kv_cache
from ...modeling_utils import (
    PipelineMixin,
    SerializedLinear,
    get_layer_ipu,
    recomputation_checkpoint,
    register,
    shift_tokens_right,
    split_encoder_decoder_ipu_config,
)


logger = logging.get_logger(__name__)


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
            if self.cross_kv_cache_initialized:
                # cross attention with cross kv cache
                key_states, value_states = self.add_to_cross_kv_cache(
                    key_value_states,
                    lambda x: self._shape(self.k_proj(x), -1, bsz),
                    lambda x: self._shape(self.v_proj(x), -1, bsz),
                )
            else:
                # cross attention
                key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
                value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif self.kv_cache_initialized:
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
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

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
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
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
        if input_ids.shape[-1] == 1:
            # KV cache enabled.
            del past_key_values_length
            return torch.index_select(self.weight, 0, self._generation_step)
        else:
            return super().forward(input_ids, past_key_values_length=past_key_values_length)


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


class IPUWhisperConditionalEncoder(WhisperEncoder):
    @classmethod
    def from_model(cls, model: WhisperEncoder, batch_size: int, num_beams: int):
        clone = model
        clone.__class__ = cls
        clone.register_buffer(
            "_encoder_last_hidden_state",
            torch.zeros((batch_size, model.config.max_source_positions, model.config.d_model), dtype=model.dtype),
            persistent=False,
        )
        clone.register_buffer("_generation_step", torch.tensor([0], dtype=torch.int32), persistent=False)
        clone._batch_size = batch_size
        clone._num_beams = num_beams
        return clone

    def to_model(self) -> WhisperEncoder:
        del self._encoder_last_hidden_state
        del self._generation_step
        del self._batch_size
        del self._num_beams

        original = self
        original.__class__ = WhisperEncoder
        return original

    def forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if attention_mask is not None or head_mask is not None or output_attentions or output_hidden_states:
            raise ValueError(f"{self.__class__.__name__} only accepts `input_features`.")

        def run_encoder(input_features):
            encoder_output = WhisperEncoder.forward(self, input_features, return_dict=True)
            return encoder_output.last_hidden_state

        def skip_encoder(input_features):
            return self._encoder_last_hidden_state

        self._encoder_last_hidden_state.copy_(
            poptorch.cond(self._generation_step == 0, run_encoder, [input_features], skip_encoder, [input_features])[0]
        )
        last_hidden_state = self._encoder_last_hidden_state
        if self._num_beams > 1:
            # Before being passed to the decoder, we must expand the encoder outputs when beam search is enabled
            # as this would be done on host.
            last_hidden_state = last_hidden_state.repeat_interleave(
                self._num_beams, dim=0, output_size=self._batch_size * self._num_beams
            )
        return BaseModelOutput(last_hidden_state=last_hidden_state)


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

    def change_encoder_class(self, restore: bool, **kwargs):
        """Changes the encoder class to run the encoder under a `poptorch.cond` op.

        Args:
            restore: whether to restore the encoder to its original version or not.
        """
        batch_size = kwargs.get("batch_size", 1)
        num_beams = kwargs.get("num_beams", 1)
        encoder = self.model.get_encoder()
        if restore:
            if isinstance(encoder, IPUWhisperConditionalEncoder):
                self.model.encoder = encoder.to_model()
        else:
            if self.ipu_config.inference_ipus_per_replica > 1:
                raise ValueError(
                    f"`{self.ipu_config.inference_ipus_per_replica=}` should be 1 when placing encoder and decoder on the same IPU."
                )
            assert_poptorch_supports_cond(
                context="Whisper encoder is being conditionally run on the same IPU as the decoder since `use_cond_encoder=True`."
            )
            self.model.encoder = IPUWhisperConditionalEncoder.from_model(encoder, batch_size, num_beams)

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
        batch_size = kwargs.get("batch_size", 1)
        num_beams = kwargs.get("num_beams", 1)
        use_cache = kwargs.get("use_cache", False)
        max_length = kwargs.get("max_length", 448)
        use_cross_cache = kwargs.get("use_cross_cache", False)
        encoder_max_length = kwargs.get("encoder_max_length", 1500)
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
                use_cross_cache=False,
                batch_size=batch_size,
                max_length=max_length,
                num_beams=num_beams,
                dtype=decoder_layer.self_attn.k_proj.weight.dtype,
            )
            decoder_layer.encoder_attn = IPUWhisperAttention.from_model(
                decoder_layer.encoder_attn,
                use_cache=False,
                use_cross_cache=use_cross_cache,
                batch_size=batch_size,
                encoder_max_length=encoder_max_length,
                num_beams=num_beams,
                dtype=decoder_layer.encoder_attn.k_proj.weight.dtype,
            )

    def change_lm_head(self, restore: bool, use_cache: bool = None):
        # Maybe use _IndexedInputLinear
        self.change_lm_head_to_indexed_input_linear(restore or use_cache)
        # Maybe use SerializedLinear
        if restore:
            lm_head = self.get_output_embeddings()
            if isinstance(lm_head, SerializedLinear):
                self.set_output_embeddings(lm_head.to_model())
                self.tie_weights()
        else:
            projection_serialization_factor = max(
                self.ipu_config._projection_serialization_factor or 1,
                sum(self.ipu_config._serialized_projection_splits_per_ipu or [1]),
            )
            if projection_serialization_factor > 1:
                self.set_output_embeddings(
                    SerializedLinear.from_model(self.get_output_embeddings(), projection_serialization_factor)
                )
                self.tie_weights()

    def quantize_linear_layers(self, restore: bool, num_groups: int = 16):
        if not restore:
            from ...quantization.group_quantize import GroupQuantLinear

            logger.info("Group quantizing linear layers")
            for module in self.model.encoder.layers:
                module.self_attn.q_proj = GroupQuantLinear.from_model(module.self_attn.q_proj, num_groups)
                module.self_attn.k_proj = GroupQuantLinear.from_model(module.self_attn.k_proj, num_groups)
                module.self_attn.v_proj = GroupQuantLinear.from_model(module.self_attn.v_proj, num_groups)
                module.self_attn.out_proj = GroupQuantLinear.from_model(module.self_attn.out_proj, num_groups)
                module.fc1 = GroupQuantLinear.from_model(module.fc1, num_groups)
                module.fc2 = GroupQuantLinear.from_model(module.fc2, num_groups)

            for module in self.model.decoder.layers:
                module.self_attn.q_proj = GroupQuantLinear.from_model(module.self_attn.q_proj, num_groups)
                module.self_attn.k_proj = GroupQuantLinear.from_model(module.self_attn.k_proj, num_groups)
                module.self_attn.v_proj = GroupQuantLinear.from_model(module.self_attn.v_proj, num_groups)
                module.self_attn.out_proj = GroupQuantLinear.from_model(module.self_attn.out_proj, num_groups)
                module.encoder_attn.q_proj = GroupQuantLinear.from_model(module.encoder_attn.q_proj, num_groups)
                module.encoder_attn.k_proj = GroupQuantLinear.from_model(module.encoder_attn.k_proj, num_groups)
                module.encoder_attn.v_proj = GroupQuantLinear.from_model(module.encoder_attn.v_proj, num_groups)
                module.encoder_attn.out_proj = GroupQuantLinear.from_model(module.encoder_attn.out_proj, num_groups)
                module.fc1 = GroupQuantLinear.from_model(module.fc1, num_groups)
                module.fc2 = GroupQuantLinear.from_model(module.fc2, num_groups)

    def parallelize(self, for_generation=False, use_cache=False, use_cross_cache=False, **kwargs):
        super().parallelize()

        if use_cache:
            kwargs = self._populate_parallelize_kwargs_with_generation_config(**kwargs)

        self._use_cond_encoder = kwargs.get("use_cond_encoder", False)
        self._use_encoder_output_buffer = kwargs.get("use_encoder_output_buffer", False)
        if self._use_cond_encoder and self._use_encoder_output_buffer:
            raise ValueError(
                "`use_cond_encoder=True` is incompatible with `use_encoder_output_buffer=True`, only set one to True."
            )
        self._use_group_quantized_linears = kwargs.get("use_group_quantized_linears", False)

        self.change_encoder_layer_class(restore=False)
        self.change_decoder_class(restore=False)
        self.change_decoder_positional_embedding(restore=False)
        self.change_attention_class(
            restore=False,
            use_cache=use_cache and for_generation,
            use_cross_cache=use_cross_cache and for_generation,
            **kwargs,
        )
        self.change_lm_head(restore=False, use_cache=use_cache or not for_generation)
        self.change_encoder_class(restore=not self._use_cond_encoder, **kwargs)
        self.quantize_linear_layers(restore=not self._use_group_quantized_linears, num_groups=16)
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

        if for_generation and not self._use_cond_encoder:
            # If running for text generation (and the encoder and decoder are run as separate Poplar executors)
            # we split the IPU config into two configs.
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
        logger.info(f"Encoder LN --> IPU {ipu}")

        decoder_embedding_ipu = decoder_layer_ipu[0]
        if (serialized_projection_splits_per_ipu := self.ipu_config._serialized_projection_splits_per_ipu) is not None:
            serialized_projection_ipus = [i for i, x in enumerate(serialized_projection_splits_per_ipu) if x]
            if len(serialized_projection_ipus) > 1:
                # This is because we are using SerializedLinear. All splits of a SerializedLinear layer must be on the
                # same IPU. We are using SerializedLinear instead of SplitLinear because we must tie the weights, which
                # cannot be done when using SplitLinear.
                raise ValueError(
                    "`serialized_projection_splits_per_ipu` must only have 1 non-zero element for Whisper."
                )
            decoder_embedding_ipu = serialized_projection_ipus[0]
        self.model.decoder.embed_tokens = poptorch.BeginBlock(
            self.model.decoder.embed_tokens, "Decoder Embedding", ipu_id=decoder_embedding_ipu
        )
        logger.info(f"Decoder Embedding  --> IPU {decoder_embedding_ipu}")

        for index, (layer, ipu) in enumerate(zip(self.model.decoder.layers, decoder_layer_ipu)):
            if self.ipu_config.recompute_checkpoint_every_layer and index != self.config.num_hidden_layers - 1:
                self._hooks.append(recomputation_checkpoint(layer))
            self.model.decoder.layers[index] = poptorch.BeginBlock(layer, f"Decoder{index}", ipu_id=ipu)
            logger.info(f"Decoder {index:<2} --> IPU {ipu}")

        self.model.decoder.layer_norm = poptorch.BeginBlock(
            self.model.decoder.layer_norm, "Decoder Layer Norm", ipu_id=ipu
        )

        logger.info(f"Head       --> IPU {decoder_embedding_ipu}")
        logger.info("---------------------------------------")
        self.proj_out = poptorch.BeginBlock(self.proj_out, "Output Projection", ipu_id=decoder_embedding_ipu)
        return self

    def deparallelize(self):
        super().deparallelize()

        self.change_encoder_layer_class(restore=True)
        self.change_decoder_class(restore=True)
        self.change_decoder_positional_embedding(restore=True)
        self.change_attention_class(restore=True)
        self.change_lm_head(restore=True)
        self.change_encoder_class(restore=True)
        self.set_on_device_generation_steps(0)

        return self

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        use_cache=None,
        encoder_outputs=None,
        attention_mask=None,
        **kwargs,
    ):
        # We don't use `past_key_values` for KV caching, and rely on `use_cache` instead.
        beam_idx = None
        if use_cache:
            decoder_input_ids = decoder_input_ids[:, -1:]
            beam_idx = kwargs.get("beam_idx", torch.arange(decoder_input_ids.shape[0], dtype=torch.long))

        ret = {
            "past_key_values": None,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": None,
            "beam_idx": beam_idx,
        }
        if self.cond_encoder_enabled:
            input_features = kwargs.get("input_features", None)
            if input_features is None:
                raise ValueError("Missing `input_features` with `use_cond_encoder=True`.")
            ret["input_features"] = input_features
        else:
            ret["encoder_outputs"] = encoder_outputs
        return ret

    # TODO: consider making such output subsetting a decorator
    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
        # Duplicate this portion of upstream logic so we can intercept the call to `shift_tokens_right`.
        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        output = super().forward(
            input_features=input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        # Minimize IO and only return loss when training.
        return Seq2SeqLMOutput(
            loss=output.loss,
            logits=None if self.training else output.logits,
            encoder_last_hidden_state=None if self.training else output.encoder_last_hidden_state,  # for tests to pass
            past_key_values=None if self.training else output.past_key_values,  # for tests to pass
        )
