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
from transformers.models.t5.modeling_t5 import __HEAD_MASK_WARNING_MSG, T5Block, T5EncoderModel, T5Stack

from optimum.utils import logging

from ...generation import IPUGenerationMixin
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

    def parallelize(self, for_generation=False):
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

        logger.info("-------------------- Device Allocation --------------------")
        logger.info("Embedding  --> IPU 0")

        if self.ipu_config.embedding_serialization_factor > 1:
            self.lm_head = SerializedLinear.from_model(self.lm_head, self.ipu_config.embedding_serialization_factor)
            # TODO: is it needed to check?
            if self.config.tie_word_embeddings:
                self.tie_weights()

        self.change_lm_head_to_indexed_input_linear(restore=not for_generation)

        self.encoder_and_decoder_embeddings_computation(True)
        self.shared = poptorch.BeginBlock(self.shared, "Embedding", ipu_id=0)

        # Use a custom T5Stack implementation because sharing the position bias causes OOM error
        self.encoder.__class__ = CustomT5Stack
        self.decoder.__class__ = CustomT5Stack

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
