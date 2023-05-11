#  Copyright 2022 The HuggingFace Team. All rights reserved.
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

import torch
import torch.nn as nn
from torch import Tensor

import poptorch
from optimum.utils import logging
from transformers import MT5ForConditionalGeneration
from transformers.activations import NewGELUActivation
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.models.mt5.modeling_mt5 import __HEAD_MASK_WARNING_MSG, MT5Block, MT5Stack

from ...generation import IPUGenerationMixin
from ...modeling_utils import (
    PipelineMixin,
    SerializedLinear,
    SharedEmbedding,
    SerializedEmbedding,
    SplitProjection,
    get_layer_ipu,
    recomputation_checkpoint,
    register,
    split_encoder_decoder_ipu_config,
)


logger = logging.get_logger(__name__)


# Copied from optimum.graphcore.models.t5.modeling_t5.UpCastWrapper
class UpCastWrapper(nn.Module):
    def __init__(self, module: nn.Module, scale: float = 1.0):
        super().__init__()
        self.module = module
        self.scale = scale

    def forward(self, input):
        return self.module(input).to(torch.float32) * self.scale


# Copied from optimum.graphcore.models.t5.modeling_t5.CustomGELU
class CustomGELU(NewGELUActivation):
    # Work-around bug with torch.nn.GELU(approximate="tanh")
    # TODO: Remove this when bug is fixed
    def forward(self, input: Tensor) -> Tensor:
        safe = torch.logical_and(-39 < input, input < 39)
        safe_input = torch.where(safe, input, 0.0)
        gelu = super().forward(safe_input)
        relu = nn.functional.relu(input)
        return torch.where(safe, gelu, relu)


# Copied from optimum.graphcore.models.t5.modeling_t5.CustomT5Block with t5->mt5 and T5->MT5
class CustomMT5Block(MT5Block):
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


# Copied from optimum.graphcore.models.t5.modeling_t5.CustomT5Stack with t5->mt5 and T5->MT5
class CustomMT5Stack(MT5Stack):
    def invert_attention_mask(self, *args, **kwargs) -> Tensor:
        return super().invert_attention_mask(*args, **kwargs) * 0.75

    def get_extended_attention_mask(self, *args, **kwargs) -> Tensor:
        return super().get_extended_attention_mask(*args, **kwargs) * 0.75


@register(MT5ForConditionalGeneration)
class PipelinedMT5ForConditionalGeneration(MT5ForConditionalGeneration, PipelineMixin, IPUGenerationMixin):
    # Copied from optimum.graphcore.models.t5.modeling_t5.PipelinedT5ForConditionalGenerationCustomT5Stack.is_encoder_and_decoder_embeddings_computation_shared
    @property
    def is_encoder_and_decoder_embeddings_computation_shared(self):
        return isinstance(self.shared, SharedEmbedding)

    # Copied from optimum.graphcore.models.t5.modeling_t5.PipelinedT5ForConditionalGenerationCustomT5Stack.encoder_and_decoder_embeddings_computation with t5->mt5 and T5->MT5
    def encoder_and_decoder_embeddings_computation(self, use_shared_embedding: bool):
        """Sets the MT5ForConditionalGeneration shared embedding layer to SharedEmbedding that combines the computation under one layer.

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

    # Copied from optimum.graphcore.models.t5.modeling_t5.PipelinedT5ForConditionalGenerationCustomT5Stack.parallelize
    # with parallelization changes for MT5
    def parallelize(self, for_generation=False):
        """
        Transform the model to run in an IPU pipeline.
        - Adds pipeline stages to the model
        - (If enabled) Replaces the shared embedding with a SerializedEmbedding
        - Adds recomputation checkpoints

        Recommended usage:
        ```
        model = PipelinedMT5ForConditionalGeneration(config).parallelize().half()
        ```
        """
        PipelineMixin.parallelize(self)

        serialized_projection_splits_per_ipu = self.ipu_config._serialized_projection_splits_per_ipu
        projection_serialization_factor = (
            self.ipu_config._projection_serialization_factor
            if self._ipu_config._projection_serialization_factor
            else sum(serialized_projection_splits_per_ipu)
        )
        serialized_embedding_splits_per_ipu = self.ipu_config._serialized_embedding_splits_per_ipu
        embedding_serialization_factor = (
            self.ipu_config._embedding_serialization_factor
            if self.ipu_config._embedding_serialization_factor
            else sum(self.ipu_config._serialized_embedding_splits_per_ipu)
        )

        # Cannot shard input and output embeddings when using
        # tied weights. Using `SerializedLinear` is exempt since
        # the weights are not sharded
        if self.config.tie_word_embeddings and (
            embedding_serialization_factor > 1
            or serialized_projection_splits_per_ipu is not None
        ):
            serialized_projection_splits_per_ipu_mode_str = self.ipu_config._get_managed_attr_mode_name("serialized_projection_splits_per_ipu")
            serialized_embedding_splits_per_ipu_mode_str = self.ipu_config._get_managed_attr_mode_name("serialized_embedding_splits_per_ipu")
            embedding_serialization_factor_mode_str = self.ipu_config._get_managed_attr_mode_name("embedding_serialization_factor")
            raise ValueError(
                "Cannot shard input and output embedding layers when using tied weights."
                f" {serialized_projection_splits_per_ipu_mode_str}={serialized_projection_splits_per_ipu}"
                f" {serialized_embedding_splits_per_ipu_mode_str}={serialized_embedding_splits_per_ipu}"
                " should not be provided when using tied input and output embeddings as it is"
                " redundant to split layers that can only reside on 1 IPU."
                f" {embedding_serialization_factor_mode_str}={embedding_serialization_factor}" 
                " should also be set to 1 as creating a `SerializedEmbedding` will split the"
                " embedding table into sub embedding tables."
            )

        logger.info("-------------------- Device Allocation --------------------")
        logger.info("Embedding  --> IPU 0")

        if embedding_serialization_factor > 1:
            self.shared = SerializedEmbedding(self. shared, embedding_serialization_factor)
        
        if projection_serialization_factor > 1:
            if serialized_projection_splits_per_ipu is None:
                serialized_lm_head = SerializedLinear(
                    self.config.d_model,
                    self.shared.num_embeddings,
                    projection_serialization_factor,
                    bias=False,
                    mode=poptorch.MatMulSerializationMode.OutputChannels
                )
                serialized_lm_head.load_state_dict(self.lm_head.state_dict())
                self.lm_head = serialized_lm_head
                # TODO: is it needed to check?
                if self.config.tie_word_embeddings:
                    self.tie_weights()
            else:
                self.lm_head = SplitProjection(
                    self.lm_head,
                    serialization_factor=projection_serialization_factor
                )
        
        self.encoder_and_decoder_embeddings_computation(True)
        
        # Parallelize the embedding layer
        if embedding_serialization_factor > 1 and serialized_embedding_splits_per_ipu is not None:
            # Sharing encoder and decoder computation wraps the
            # SerializedEmbedding using SharedEmbedding
            logger.info("Embedding Placement: ")
            self.shared.shared = self.shared.shared.parallelize(serialized_embedding_splits_per_ipu)
        else:
            logger.info("Embedding  --> IPU 0")
            self.shared = poptorch.BeginBlock(self.shared, "Embedding", ipu_id=0)

        # Use a custom MT5Stack implementation because sharing the position bias causes OOM error
        self.encoder.__class__ = CustomMT5Stack
        self.decoder.__class__ = CustomMT5Stack
        
        # Upcast input embeddings so that the residuals remain in FP32. This
        # cast is reversed where necessary by the MT5LayerNorm layers in:
        # - first layer of MT5LayerSelfAttention
        # - first layer of MT5LayerFF
        # - final_layer_norm
        # Which, conveniently, are all the places that this needs to happen.
        # Therefore, so we just need to upcast immediately before the residual
        # adds in MT5LayerSelfAttention and MT5LayerFF. This is handled in the
        # for loop below.
        self.encoder.embed_tokens = UpCastWrapper(self.encoder.embed_tokens)

        # Use a custom MT5Block implementation that removes a dynamic if blocks that can't be statically traced
        for block in self.encoder.block:
            block.__class__ = CustomMT5Block
            # Dropout happens immediately before the residual add. Inserting a
            # cast in MT5LayerSelfAttention and MT5LayerFF keeps the residual
            # structure in FP32
            block.layer[0].dropout = UpCastWrapper(block.layer[0].dropout)
            # Scale down the weights for the MT5LayerFF down-projection and
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
            block.__class__ = CustomMT5Block
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

        # Parallelize the lm head
        if self.config.tie_word_embeddings:
            # Place LM head on IPU 0
            ipu_id = 0
            logger.info(f"LM Head Output --> IPU {ipu_id}")
            self.lm_head = poptorch.BeginBlock(self.lm_head, "LM Head Output", ipu_id=ipu_id)
        else:
            # Place LM head on the last IPU if serialized_projection_splits_per_ipu is not provided
            # For generation: override serialized_projection_splits_per_ipu for generation
            if for_generation:
                serialized_projection_splits_per_ipu = self.decoder_ipu_config.serialized_projection_splits_per_ipu
            # Parallelize `SplitLinear` layer if configuration is provided
            if serialized_projection_splits_per_ipu is not None:
                logger.info("LM Head Placement: ")
                self.lm_head = self.lm_head.parallelize(serialized_projection_splits_per_ipu)
            else:
                # Place SerializedLinear and nn.Linear forms of the lm head on the last IPU
                ipu_id = self.ipu_config._ipus_per_replica - 1
                logger.info(f"LM Head Output --> IPU {ipu_id}")
                self.lm_head = poptorch.BeginBlock(self.lm_head, "LM Head Output", ipu_id=ipu_id)

        self.change_lm_head_to_indexed_input_linear(restore=not for_generation)
        
        logger.info("-----------------------------------------------------------")
        return self

    # Copied from optimum.graphcore.models.t5.modeling_t5.PipelinedT5ForConditionalGenerationCustomT5Stack.parallelize
    # with deparallelization changes for MT5
    def deparallelize(self):
        """
        Undo the changes to the model done by `parallelize`.
        You should call this before doing `save_pretrained` so that the `model.state_dict` is
        fully compatible with `transformers.MT5ForConditionalGeneration`.
        """
        # MT5ForConditionalGeneration has a deparallelize method, so make sure that the PipelineMixin one is used here.
        PipelineMixin.deparallelize(self)

        self.encoder_and_decoder_embeddings_computation(False)

        self.encoder.__class__ = MT5Stack
        self.decoder.__class__ = MT5Stack

        self.encoder.embed_tokens = self.encoder.embed_tokens.module

        for block in self.encoder.block:
            block.__class__ = MT5Block
            block.layer[0].dropout = block.layer[0].dropout.module
            with torch.no_grad():
                block.layer[1].DenseReluDense.wo.weight *= block.layer[1].dropout.scale
            block.layer[1].dropout = block.layer[1].dropout.module
            if self.config.dense_act_fn == "gelu_new":
                block.layer[1].DenseReluDense.act = NewGELUActivation()
        for block in self.decoder.block:
            block.__class__ = MT5Block
            if self.config.dense_act_fn == "gelu_new":
                block.layer[2].DenseReluDense.act = NewGELUActivation()

        self.change_lm_head_to_indexed_input_linear(restore=True)

        if self.lm_head.__class__ == SerializedLinear:
            old_lm_head = nn.Linear(
                self.config.d_model,
                self.shared.num_embeddings,
                bias=False,
            )
            old_lm_head.load_state_dict(self.lm_head.state_dict())
            self.lm_head = old_lm_head
            # TODO: is it needed to check?
            if self.config.tie_word_embeddings:
                self.tie_weights()
        elif self.lm_head.__class__ == SplitProjection:
            self.lm_head = self.lm_head.deserialize()
        
        if self.shared.__class__ == SerializedEmbedding:
            self.shared = self.shared.deserialize()
            
        return self

    # Copied from optimum.graphcore.models.t5.modeling_t5.PipelinedT5ForConditionalGenerationCustomT5Stack.forward
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
