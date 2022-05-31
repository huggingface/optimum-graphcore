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

import torch
import torch.nn as nn
from torch import Tensor

import poptorch
from optimum.utils import logging
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.models.t5.modeling_t5 import __HEAD_MASK_WARNING_MSG, T5LayerNorm, T5Stack

from ...generation_utils import IPUGenerationMixin
from ...modeling_utils import (
    GenerationMethodsMixin,
    PipelineMixin,
    SerializedLinear,
    SharedEmbedding,
    get_layer_ipu,
    recomputation_checkpoint,
    register,
)


logger = logging.get_logger(__name__)


class CustomT5Stack(T5Stack):
    def invert_attention_mask(self, encoder_attention_mask: Tensor) -> Tensor:
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
        # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
        # /transformer/transformer_layers.py#L270
        # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
        # encoder_extended_attention_mask.transpose(-1, -2))
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility

        # Always use -1e4 to avoid NaN issues.
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e4
        return encoder_extended_attention_mask


@register(T5ForConditionalGeneration)
class PipelinedT5ForConditionalGeneration(
    GenerationMethodsMixin, T5ForConditionalGeneration, PipelineMixin, IPUGenerationMixin
):
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

    def scale_down_weights(self, factor: float = 1, restore: bool = False):
        self.lm_scale_modifier = 1 if not restore else None
        # self.lm_scale_modifier = nn.Parameter(torch.ones(self.config.d_model, dtype=torch.float16)) if not restore else None

        emb_scaling = 1 / 32.0 * factor
        att_v_scaling = 1 / 4.0 * factor
        att_o_scaling = 1 / 8.0 * factor
        ff_wi_scaling = 1 / 4.0 * factor
        ff_wo_scaling = 1 / 4.0 * factor
        ff_ln_scaling = 1 / 2.0 * factor

        if restore:
            emb_scaling = 1 / emb_scaling
            att_v_scaling = 1 / att_v_scaling
            att_o_scaling = 1 / att_o_scaling
            ff_wi_scaling = 1 / ff_wi_scaling
            ff_wo_scaling = 1 / ff_wo_scaling
            ff_ln_scaling = 1 / ff_ln_scaling

        with torch.no_grad():
            self.shared.weight *= emb_scaling
            for unit in self.encoder.block:
                unit.layer[0].SelfAttention.v.weight *= att_v_scaling
                unit.layer[0].SelfAttention.o.weight *= att_o_scaling
                unit.layer[1].DenseReluDense.wi.weight *= ff_wi_scaling
                unit.layer[1].DenseReluDense.wo.weight *= ff_wo_scaling
                unit.layer[1].layer_norm.weight *= ff_ln_scaling
            for unit in self.decoder.block:
                unit.layer[0].SelfAttention.v.weight *= att_v_scaling
                unit.layer[0].SelfAttention.o.weight *= att_o_scaling
                unit.layer[1].EncDecAttention.v.weight *= att_v_scaling
                unit.layer[1].EncDecAttention.o.weight *= att_o_scaling
                unit.layer[2].DenseReluDense.wi.weight *= ff_wi_scaling
                unit.layer[2].DenseReluDense.wo.weight *= ff_wo_scaling
                unit.layer[2].layer_norm.weight *= ff_ln_scaling

            if not restore:
                self.lm_scale_modifier /= emb_scaling

    def parallelize(self):
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
        layer_ipu = get_layer_ipu(self.ipu_config.layers_per_ipu)

        logger.info("-------------------- Device Allocation --------------------")
        logger.info("Embedding  --> IPU 0")

        if self.ipu_config.embedding_serialization_factor > 1:
            serialized_lm_head = SerializedLinear(
                self.config.d_model,
                self.shared.num_embeddings,
                self.ipu_config.embedding_serialization_factor,
                bias=False,
                mode=poptorch.MatMulSerializationMode.OutputChannels,
            )
            serialized_lm_head.load_state_dict(self.lm_head.state_dict())
            self.lm_head = serialized_lm_head
            # TODO: is it needed to check?
            if self.config.tie_word_embeddings:
                self.tie_weights()

        # self.scale_down_weights(factor=1)
        self.encoder_and_decoder_embeddings_computation(True)
        self.shared = poptorch.BeginBlock(self.shared, "Embedding", ipu_id=0)

        # Use a custom T5Stack implementation because sharing the position bias causes OOM error
        self.encoder.__class__ = CustomT5Stack
        self.decoder.__class__ = CustomT5Stack

        # Enable autocast for T5LayerNorm because computation cannot happen in fp16
        for mod in self.modules():
            if isinstance(mod, T5LayerNorm):
                mod.forward = poptorch.autocast(enabled=True)(mod.forward)

        for index, layer in enumerate(self.encoder.block):
            ipu = layer_ipu[index]
            if self.ipu_config.recompute_checkpoint_every_layer and index != self.config.num_layers - 1:
                recomputation_checkpoint(layer)
            self.encoder.block[index] = poptorch.BeginBlock(layer, f"Encoder{index}", ipu_id=ipu)
            logger.info(f"Encoder {index:<2} --> IPU {ipu}")

        self.encoder.final_layer_norm = poptorch.BeginBlock(
            self.encoder.final_layer_norm, "Encoder Stack Final LayerNorm", ipu_id=ipu
        )

        shift = len(self.encoder.block)
        for index, layer in enumerate(self.decoder.block):
            ipu = layer_ipu[index + shift]
            if self.ipu_config.recompute_checkpoint_every_layer and index != self.config.num_layers - 1:
                recomputation_checkpoint(layer)
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
        # self.scale_down_weights(factor=1, restore=True)

        self.encoder.__class__ = T5Stack
        self.decoder.__class__ = T5Stack

        for mod in self.modules():
            if isinstance(mod, T5LayerNorm):
                mod.forward = T5LayerNorm.forward.__get__(mod, T5LayerNorm)

        if self.ipu_config.embedding_serialization_factor > 1:
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

        return self

    def _forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=False,
    ):
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

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

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
            torch.cuda.set_device(self.encoder.first_device)
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

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

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

    def train(self, mode: bool = True) -> "PipelinedT5ForConditionalGeneration":
        mod = super(T5ForConditionalGeneration, self).train(mode=mode)
        # TODO: enable that once generation is supported.
        # mod.forward = mod._forward_for_train if mode else mod._forward_for_generate
        mod.forward = mod._forward_for_train
        return mod

    def _forward_for_train(self, input_ids, attention_mask, decoder_input_ids, labels=None):
        outputs = self._forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            use_cache=False,
            return_dict=False,
        )
        # Only returning the loss to make the communication between the host and the device faster.
        return outputs[0:1]

    def _forward_for_generate(self, encoder_outputs, decoder_input_ids, attention_mask, labels=None):
        outputs = super().forward(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            return_dict=False,
            use_cache=False,
            labels=labels,
        )
        # Only returning the loss (if labels is provided) and the logits.
        if labels is None:
            return outputs[:1]
        return outputs[:2]

    forward = _forward_for_train
