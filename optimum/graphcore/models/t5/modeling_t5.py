#  Copyright 2022 The HuggingFace Team. All rights reserved.
#  Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import warnings
from typing import Optional, Tuple, Union

import poptorch
import torch
import torch.nn as nn
from torch import Tensor
from transformers import T5ForConditionalGeneration
from transformers.activations import NewGELUActivation
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.models.t5.modeling_t5 import __HEAD_MASK_WARNING_MSG, T5Attention, T5Block, T5EncoderModel, T5Stack

from optimum.utils import logging

from ...generation import IPUAttentionMixin, IPUGenerationMixin, supports_kv_cache
from ...modeling_utils import (
    PipelineMixin,
    SerializedLinear,
    SharedEmbedding,
    get_layer_ipu,
    recomputation_checkpoint,
    register,
    split_encoder_decoder_ipu_config,
)


logger = logging.get_logger(__name__)


class UpCastWrapper(nn.Module):
    def __init__(self, module: nn.Module, scale: float = 1.0):
        super().__init__()
        self.module = module
        self.scale = scale

    def forward(self, input):
        return self.module(input).to(torch.float32) * self.scale


class CustomGELU(NewGELUActivation):
    # Work-around bug with torch.nn.GELU(approximate="tanh")
    # TODO: Remove this when bug is fixed
    def forward(self, input: Tensor) -> Tensor:
        safe = torch.logical_and(-39 < input, input < 39)
        safe_input = torch.where(safe, input, 0.0)
        gelu = super().forward(safe_input)
        relu = nn.functional.relu(input)
        return torch.where(safe, gelu, relu)


class IPUT5Attention(T5Attention, IPUAttentionMixin):
    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]
        real_seq_length = seq_length

        # On the IPU the real sequence length is the padded sequence. If self attention
        # kv caching is enabled, this length can be obtained from the kv cache.
        # for cross kv caching computing the relative attention bias is disabled
        # so we do not need to be aware of the decoder max length
        if self.kv_cache_initialized:
            real_seq_length = self._k_cache.shape[-2]

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        if key_value_states is None:
            # self-attn
            # (batch_size, n_heads, seq_length, dim_per_head)
            key_states = shape(self.k(hidden_states))
            value_states = shape(self.v(hidden_states))
        elif not self.cross_kv_cache_initialized:
            # cross-attn
            # (batch_size, n_heads, seq_length, dim_per_head)
            key_states = shape(self.k(key_value_states))
            value_states = shape(self.v(key_value_states))

        if self.kv_cache_initialized or self.cross_kv_cache_initialized:
            # Change: remove branch to support prefix tuning
            # This requires the IPU on device to cache to be aware of
            # the prefix tokens
            if key_value_states is None:
                # caching key states for self attention
                # self-attn
                # (batch_size, n_heads, key_length, dim_per_head)
                tgt_len = key_states.shape[-2]
                if tgt_len != 1:
                    raise ValueError(f"KV cache expects tgt_len = 1, received {tgt_len}.")
                key_states, value_states = self.add_to_kv_cache(key_states, value_states)
            else:
                # cached cross-attn
                key_states, value_states = self.add_to_cross_kv_cache(
                    key_value_states,
                    lambda x: shape(self.k(x)),
                    lambda x: shape(self.v(x)),
                )

        # compute scores
        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                if self.cross_kv_cache_initialized:
                    raise NotImplementedError(
                        f"Cross KV caching with {self.has_relative_attention_bias=} is not yet supported on the IPU."
                    )
                position_bias = self.compute_bias(real_seq_length, key_length, device=scores.device)

            # if key and values are already calculated
            # we want only the last query position bias
            if self.kv_cache_initialized:
                position_bias = poptorch.dynamic_slice(position_bias, 2, self._generation_step, 1, 1)

            if mask is not None:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

        if self.pruned_heads:
            mask = torch.ones(position_bias.shape[1])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias

        scores += position_bias_masked

        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length

        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)

        return outputs


class CustomT5Block(T5Block):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
    ):
        if past_key_value is not None:
            if not self.is_decoder:
                logger.warning("`past_key_values` is passed to the encoder. Please make sure this is intended.")
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        # Custom: Remove check for inf
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.tensor(torch.finfo(hidden_states.dtype).max - 1000, dtype=hidden_states.dtype)
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            # Custom: Remove check for inf
            if hidden_states.dtype == torch.float16:
                clamp_value = torch.tensor(torch.finfo(hidden_states.dtype).max - 1000, dtype=hidden_states.dtype)
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        # Custom: Remove check for inf
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.tensor(torch.finfo(hidden_states.dtype).max - 1000, dtype=hidden_states.dtype)
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)


class CustomT5Stack(T5Stack):
    def invert_attention_mask(self, *args, **kwargs) -> Tensor:
        return super().invert_attention_mask(*args, **kwargs) * 0.75

    def get_extended_attention_mask(self, *args, **kwargs) -> Tensor:
        return super().get_extended_attention_mask(*args, **kwargs) * 0.75

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        Intercept the forward call in order to provide the correct attention mask if self-attention kv caching is enabled.
        The alternative is to replicate the parent forward call here so that we can prevent the (default) construction of an attention mask
        when kv caching is enabled. This would allow the attention layers to make the call `IPUT5Attention.update_attention_mask` to create
        an attention mask with the knowledge of the decoder max length. To avoid replicating all but a few lines of code the former option
        is kept.
        """
        if self.is_decoder and self.block[0].layer[0].SelfAttention.kv_cache_initialized:
            if attention_mask is None:
                attention_layer = self.block[0].layer[0].SelfAttention
                bsz, _, src_len, _ = attention_layer._k_cache.shape
                attention_mask = torch.ones((1, src_len))
                mask_cond = torch.arange(src_len).view(1, src_len)
                attention_mask.masked_fill_(mask_cond >= attention_layer._generation_step + 1, 0)
                attention_mask = attention_mask.to(attention_layer._k_cache.dtype)
                attention_mask = attention_mask.expand(bsz, 1, src_len)
            else:
                raise ValueError(
                    f"Providing an {attention_mask=} to the decoder when kv-caching is enabled is currently not supported."
                )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


@supports_kv_cache
@register(T5ForConditionalGeneration)
class PipelinedT5ForConditionalGeneration(T5ForConditionalGeneration, PipelineMixin, IPUGenerationMixin):
    @property
    def is_encoder_and_decoder_embeddings_computation_shared(self):
        return isinstance(self.shared, SharedEmbedding)

    def encoder_and_decoder_embeddings_computation(self, use_shared_embedding: bool):
        """Sets the T5ForConditionalGeneration shared embedding layer to SharedEmbedding that combines the computation under one layer.

        Args:
            use_shared_embedding: whether to use SharedEmbedding or not.
        """

        if use_shared_embedding:
            if isinstance(self.shared, SharedEmbedding):
                logger.warning("encoder and decoder embeddings computation is already shared")
            else:
                self.shared = SharedEmbedding(self.shared)
        else:
            if isinstance(self.shared, nn.Embedding):
                logger.warning("encoder and decoder embeddings computation is not shared")
            else:
                self.shared = self.shared.shared

    def parallelize(self, for_generation=False, use_cache=False, use_cross_cache=False, **kwargs):
        """
        Transform the model to run in an IPU pipeline.
        - Adds pipeline stages to the model
        - (If enabled) Replaces the shared embedding with a SerializedEmbedding
        - Adds recomputation checkpoints

        Recommended usage:
        ```
        model = PipelinedT5ForConditionalGeneration(config).parallelize().half()
        ```
        """
        PipelineMixin.parallelize(self)

        if use_cache:
            kwargs = self._populate_parallelize_kwargs_with_generation_config(**kwargs)

        logger.info("-------------------- Device Allocation --------------------")
        logger.info("Embedding  --> IPU 0")

        if self.ipu_config.embedding_serialization_factor > 1:
            self.lm_head = SerializedLinear.from_model(self.lm_head, self.ipu_config.embedding_serialization_factor)
            # TODO: is it needed to check?
            if self.config.tie_word_embeddings:
                self.tie_weights()

        self.change_lm_head_to_indexed_input_linear(restore=not (for_generation and not use_cache))

        self.encoder_and_decoder_embeddings_computation(True)
        self.shared = poptorch.BeginBlock(self.shared, "Embedding", ipu_id=0)

        # Use a custom T5Stack implementation because sharing the position bias causes OOM error
        self.encoder.__class__ = CustomT5Stack
        self.decoder.__class__ = CustomT5Stack

        # Optimisations for generation
        self.change_attention_class(
            restore=False,
            use_cache=use_cache and for_generation,
            use_cross_cache=use_cross_cache and for_generation,
            **kwargs,
        )
        self._use_encoder_output_buffer = kwargs.get("use_encoder_output_buffer", False)
        self.set_on_device_generation_steps(kwargs.get("on_device_generation_steps", 0))

        # Upcast input embeddings so that the residuals remain in FP32. This
        # cast is reversed where necessary by the T5LayerNorm layers in:
        # - first layer of T5LayerSelfAttention
        # - first layer of T5LayerFF
        # - final_layer_norm
        # Which, conveniently, are all the places that this needs to happen.
        # Therefore, so we just need to upcast immediately before the residual
        # adds in T5LayerSelfAttention and T5LayerFF. This is handled in the
        # for loop below.
        self.encoder.embed_tokens = UpCastWrapper(self.encoder.embed_tokens)

        # Use a custom T5Block implementation that removes a dynamic if blocks that can't be statically traced
        for block in self.encoder.block:
            block.__class__ = CustomT5Block
            # Dropout happens immediately before the residual add. Inserting a
            # cast in T5LayerSelfAttention and T5LayerFF keeps the residual
            # structure in FP32
            block.layer[0].dropout = UpCastWrapper(block.layer[0].dropout)
            # Scale down the weights for the T5LayerFF down-projection and
            # then scale its output back up again after it is cast to FP32
            scale = 8.0
            with torch.no_grad():
                block.layer[1].DenseReluDense.wo.weight /= scale
            block.layer[1].dropout = UpCastWrapper(block.layer[1].dropout, scale)
            # Prevent overflow in NewGELUActivation
            if self.config.dense_act_fn == "gelu_new":
                # TODO: Work-around bug with torch.nn.GELU(approximate="tanh"). Replace
                # this with block.layer[1].DenseReluDense.act = torch.nn.GELU(approximate="tanh")
                # when bug is fixed
                block.layer[1].DenseReluDense.act = CustomGELU()
        for block in self.decoder.block:
            block.__class__ = CustomT5Block
            # Work-around bug with torch.nn.GELU(approximate="tanh")
            # TODO: Remove this when bug is fixed
            if self.config.dense_act_fn == "gelu_new":
                block.layer[2].DenseReluDense.act = CustomGELU()

        num_encoder_layers = len(self.encoder.block)
        num_decoder_layers = len(self.decoder.block)

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

        for index, (layer, ipu) in enumerate(zip(self.encoder.block, encoder_layer_ipu)):
            if self.ipu_config.recompute_checkpoint_every_layer and index != self.config.num_layers - 1:
                self._hooks.append(recomputation_checkpoint(layer))
            self.encoder.block[index] = poptorch.BeginBlock(layer, f"Encoder{index}", ipu_id=ipu)
            logger.info(f"Encoder {index:<2} --> IPU {ipu}")

        self.encoder.final_layer_norm = poptorch.BeginBlock(
            self.encoder.final_layer_norm, "Encoder Stack Final LayerNorm", ipu_id=ipu
        )

        for index, (layer, ipu) in enumerate(zip(self.decoder.block, decoder_layer_ipu)):
            if self.ipu_config.recompute_checkpoint_every_layer and index != self.config.num_layers - 1:
                self._hooks.append(recomputation_checkpoint(layer))
            self.decoder.block[index] = poptorch.BeginBlock(layer, f"Decoder{index}", ipu_id=ipu)
            logger.info(f"Decoder {index:<2} --> IPU {ipu}")

        self.decoder.final_layer_norm = poptorch.BeginBlock(
            self.decoder.final_layer_norm, "Decoder Stack Final LayerNorm", ipu_id=ipu
        )

        logger.info("LM Head Output --> IPU 0")
        self.lm_head = poptorch.BeginBlock(self.lm_head, "LM Head Output", ipu_id=0)
        logger.info("-----------------------------------------------------------")
        return self

    def deparallelize(self):
        """
        Undo the changes to the model done by `parallelize`.
        You should call this before doing `save_pretrained` so that the `model.state_dict` is
        fully compatible with `transformers.T5ForConditionalGeneration`.
        """
        # T5ForConditionalGeneration has a deparallelize method, so make sure that the PipelineMixin one is used here.
        PipelineMixin.deparallelize(self)

        self.encoder_and_decoder_embeddings_computation(False)

        self.set_on_device_generation_steps(0)
        self.change_attention_class(restore=True)
        self.encoder.__class__ = T5Stack
        self.decoder.__class__ = T5Stack

        self.encoder.embed_tokens = self.encoder.embed_tokens.module

        for block in self.encoder.block:
            block.__class__ = T5Block
            block.layer[0].dropout = block.layer[0].dropout.module
            with torch.no_grad():
                block.layer[1].DenseReluDense.wo.weight *= block.layer[1].dropout.scale
            block.layer[1].dropout = block.layer[1].dropout.module
            if self.config.dense_act_fn == "gelu_new":
                block.layer[1].DenseReluDense.act = NewGELUActivation()
        for block in self.decoder.block:
            block.__class__ = T5Block
            if self.config.dense_act_fn == "gelu_new":
                block.layer[2].DenseReluDense.act = NewGELUActivation()

        self.change_lm_head_to_indexed_input_linear(restore=True)

        if isinstance(self.lm_head, SerializedLinear):
            self.lm_head = self.lm_head.to_model()
            # TODO: is it needed to check?
            if self.config.tie_word_embeddings:
                self.tie_weights()

        return self

    def change_attention_class(self, restore=False, **kwargs):
        """Changes the attention layers to either use the original T5Attention forward
        or IPUT5Attention forward.

        Args:
            restore (bool, optional): whether to restore the attention layers to their original version or not. Defaults to False.
        """
        use_cache = kwargs.get("use_cache", False)
        use_cross_cache = kwargs.get("use_cross_cache", False)
        batch_size = kwargs.get("batch_size", 1)
        max_length = kwargs.get("max_length", 128)
        encoder_max_length = kwargs.get("encoder_max_length", 1500)
        num_beams = kwargs.get("num_beams", 1)

        for layer in self.encoder.block:
            if restore:
                layer.layer[0].SelfAttention = layer.layer[0].SelfAttention.to_model(T5Attention)
                continue

            layer.layer[0].SelfAttention = IPUT5Attention.from_model(
                layer.layer[0].SelfAttention,
                use_cache=False,
            )

        for layer in self.decoder.block:
            if restore:
                layer.layer[0].SelfAttention = layer.layer[0].SelfAttention.to_model(T5Attention)
                layer.layer[1].EncDecAttention = layer.layer[1].EncDecAttention.to_model(T5Attention)
                continue

            layer.layer[0].SelfAttention = IPUT5Attention.from_model(
                layer.layer[0].SelfAttention,
                use_cache=use_cache,
                use_cross_cache=False,
                batch_size=batch_size,
                max_length=max_length,
                num_beams=num_beams,
                num_heads=layer.layer[0].SelfAttention.n_heads,
                head_dim=layer.layer[0].SelfAttention.key_value_proj_dim,
                dtype=layer.layer[0].SelfAttention.k.weight.dtype,
            )

            layer.layer[1].EncDecAttention = IPUT5Attention.from_model(
                layer.layer[1].EncDecAttention,
                use_cache=False,
                use_cross_cache=use_cross_cache,
                batch_size=batch_size,
                encoder_max_length=encoder_max_length,
                num_beams=num_beams,
                num_heads=layer.layer[1].EncDecAttention.n_heads,
                head_dim=layer.layer[1].EncDecAttention.key_value_proj_dim,
                dtype=layer.layer[1].EncDecAttention.k.weight.dtype,
            )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        decoder_attention_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # We don't use `past_key_values` for KV caching, and rely on `use_cache` instead.
        beam_idx = None
        if use_cache:
            # cut decoder_input_ids if past is used
            input_ids = input_ids[:, -1:]
            beam_idx = kwargs.get("beam_idx", torch.arange(input_ids.shape[0], dtype=torch.long))

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": None,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
            "beam_idx": beam_idx,
        }

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.is_encoder_and_decoder_embeddings_computation_shared:
            inputs_embeds, decoder_inputs_embeds = self.shared(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
            )
            if inputs_embeds is not None:
                input_ids = None
            if decoder_inputs_embeds is not None:
                decoder_input_ids = None

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_scale_modifier = getattr(self, "lm_scale_modifier", None)
        if lm_scale_modifier is not None:
            sequence_output = sequence_output * lm_scale_modifier
        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        # Only returning the loss to make the communication between the host and the device faster.
        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return (loss,) if labels is not None else output

        if loss is not None:
            return Seq2SeqLMOutput(
                loss=loss,
            )

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


@register(T5EncoderModel)
class PipelinedT5EncoderModel(T5EncoderModel, PipelineMixin):
    def parallelize(self):
        """
        Transform the model to run in an IPU pipeline.
        - Adds pipeline stages to the model
        - Adds recomputation checkpoints

        Recommended usage:
        ```
        model = PipelinedT5EncoderModel(config).parallelize().half()
        ```
        """
        PipelineMixin.parallelize(self)

        logger.info("-------------------- Device Allocation --------------------")
        logger.info("Embedding  --> IPU 0")

        self.shared = poptorch.BeginBlock(self.shared, "Embedding", ipu_id=0)

        # Use a custom T5Stack implementation because sharing the position bias causes OOM error
        self.encoder.__class__ = CustomT5Stack

        # Upcast input embeddings so that the residuals remain in FP32. This
        # cast is reversed where necessary by the T5LayerNorm layers in:
        # - first layer of T5LayerSelfAttention
        # - first layer of T5LayerFF
        # - final_layer_norm
        # Which, conveniently, are all the places that this needs to happen.
        # Therefore, so we just need to upcast immediately before the residual
        # adds in T5LayerSelfAttention and T5LayerFF. This is handled in the
        # for loop below.
        self.encoder.embed_tokens = UpCastWrapper(self.encoder.embed_tokens)

        # Use a custom T5Block implementation that removes a dynamic if blocks that can't be statically traced
        for block in self.encoder.block:
            block.__class__ = CustomT5Block
            # Dropout happens immediately before the residual add. Inserting a
            # cast in T5LayerSelfAttention and T5LayerFF keeps the residual
            # structure in FP32
            block.layer[0].dropout = UpCastWrapper(block.layer[0].dropout)
            # Scale down the weights for the T5LayerFF down-projection and
            # then scale its output back up again after it is cast to FP32
            scale = 8.0
            with torch.no_grad():
                block.layer[1].DenseReluDense.wo.weight /= scale
            block.layer[1].dropout = UpCastWrapper(block.layer[1].dropout, scale)
            # Prevent overflow in NewGELUActivation
            if self.config.dense_act_fn == "gelu_new":
                # TODO: Work-around bug with torch.nn.GELU(approximate="tanh"). Replace
                # this with block.layer[1].DenseReluDense.act = torch.nn.GELU(approximate="tanh")
                # when bug is fixed
                block.layer[1].DenseReluDense.act = CustomGELU()

        num_encoder_layers = len(self.encoder.block)
        number_of_layers = num_encoder_layers
        encoder_layer_ipu = get_layer_ipu(self.ipu_config, number_of_layers)

        for index, (layer, ipu) in enumerate(zip(self.encoder.block, encoder_layer_ipu)):
            if self.ipu_config.recompute_checkpoint_every_layer and index != self.config.num_layers - 1:
                self._hooks.append(recomputation_checkpoint(layer))
            self.encoder.block[index] = poptorch.BeginBlock(layer, f"Encoder{index}", ipu_id=ipu)
            logger.info(f"Encoder {index:<2} --> IPU {ipu}")

        self.encoder.final_layer_norm = poptorch.BeginBlock(
            self.encoder.final_layer_norm, "Encoder Stack Final LayerNorm", ipu_id=ipu
        )

        return self

    def deparallelize(self):
        """
        Undo the changes to the model done by `parallelize`.
        You should call this before doing `save_pretrained` so that the `model.state_dict` is
        fully compatible with `transformers.T5ForConditionalGeneration`.
        """
        # T5ForConditionalGeneration has a deparallelize method, so make sure that the PipelineMixin one is used here.
        PipelineMixin.deparallelize(self)

        self.encoder.__class__ = T5Stack
        self.encoder.embed_tokens = self.encoder.embed_tokens.module

        for block in self.encoder.block:
            block.__class__ = T5Block
            block.layer[0].dropout = block.layer[0].dropout.module
            with torch.no_grad():
                block.layer[1].DenseReluDense.wo.weight *= block.layer[1].dropout.scale
            block.layer[1].dropout = block.layer[1].dropout.module
            if self.config.dense_act_fn == "gelu_new":
                block.layer[1].DenseReluDense.act = NewGELUActivation()

        return self
