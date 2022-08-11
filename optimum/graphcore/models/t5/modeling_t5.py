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

import poptorch
from optimum.utils import logging
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.models.t5.modeling_t5 import __HEAD_MASK_WARNING_MSG, T5LayerNorm

from ....fx.optimization import ChangeTrueDivToMulByInverse, MergeLinears, ReversibleTransformation, compose
from ...fx.transformations import (
    AddPoptorchBlock,
    AddPoptorchBlocksInSeries,
    ClipValues,
    ClipValuesSymmetric,
    LinearToSerializedLinear,
    RecomputationCheckpoint,
    ShareEmbeddingComputation,
    TieWeights,
    TupleOutput,
)
from ...fx.utils import symbolic_trace_pipelined_model
from ...generation_utils import IPUGenerationMixin
from ...modeling_utils import GenerationMethodsMixin, PipelineMixin, SharedEmbedding, get_layer_ipu, register


logger = logging.get_logger(__name__)

_OPTIMIZATION_TRANSFORMATIONS = [
    ChangeTrueDivToMulByInverse(),
    MergeLinears(),
    #    FuseBiasInLinear(),
]

_NON_REVERSIBLE_TRANSFORMATIONS = [
    ClipValuesSymmetric(1e4, exclude_targets=["view"]),
    ClipValues(1e-4, float("inf"), include_targets=[torch.nn.LayerNorm]),
    TupleOutput(),
]


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

    def get_transformations(self):
        log_insertions = self.ipu_config.log_insertions
        layer_ipu = get_layer_ipu(self.ipu_config.layers_per_ipu)
        transformations = [
            AddPoptorchBlock("Embedding", 0, "shared", log_insertions=log_insertions),
            AddPoptorchBlocksInSeries(
                "Encoder", layer_ipu[: self.config.num_layers], r"encoder.block.[0-9]+", log_insertions=log_insertions
            ),
            AddPoptorchBlocksInSeries(
                "Decoder",
                layer_ipu[self.config.num_layers - 1 :],
                r"decoder.block.[0-9]+",
                log_insertions=log_insertions,
            ),
            AddPoptorchBlock("LM Head Output", 0, "lm_head", log_insertions=log_insertions),
        ]
        if self.ipu_config.recompute_checkpoint_every_layer:
            transformations += [
                RecomputationCheckpoint(
                    "encoder.block.[0-9]+", to_exclude=f"encoder.block.{self.config.num_layers - 1}"
                ),
                RecomputationCheckpoint(
                    "decoder.block.[0-9]+", to_exclude=f"decoder.block.{self.config.num_layers - 1}"
                ),
            ]

        if not isinstance(self, torch.fx.GraphModule):
            if self.ipu_config.embedding_serialization_factor > 1:
                transformations += [
                    LinearToSerializedLinear("lm_head"),
                    TieWeights("shared", "lm_head"),
                ]
            transformations += [ShareEmbeddingComputation()]
        return transformations

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
        PipelineMixin.parallelize(self)
        for mod in self.modules():
            if isinstance(mod, T5LayerNorm):
                mod.forward = poptorch.autocast(enabled=True)(mod.forward)
        traced = symbolic_trace_pipelined_model(self)
        transformations = self.get_transformations()
        transformations += _OPTIMIZATION_TRANSFORMATIONS
        composition = compose(*transformations)
        non_reversible_composition = compose(*_NON_REVERSIBLE_TRANSFORMATIONS)
        traced = composition(traced)
        traced = non_reversible_composition(traced)
        return traced

    def deparallelize(self):
        """
        Undo the changes to the model done by `parallelize`.
        You should call this before doing `save_pretrained` so that the `model.state_dict` is
        fully compatible with `transformers.T5ForConditionalGeneration`.
        """
        # T5ForConditionalGeneration has a deparallelize method, so make sure that the PipelineMixin one is used here.
        PipelineMixin.deparallelize(self)
        transformations = self.get_transformations()
        transformations += _OPTIMIZATION_TRANSFORMATIONS
        transformations = [t for t in transformations if isinstance(t, ReversibleTransformation)]
        composition = compose(*transformations)
        self = composition(self, reverse=True)
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
