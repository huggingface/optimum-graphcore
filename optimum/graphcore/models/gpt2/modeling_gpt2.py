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

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import GPT2ForSequenceClassification, GPT2ForTokenClassification, GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions, SequenceClassifierOutputWithPast

from ....fx.optimization import ReversibleTransformation, compose
from ....utils import logging
from ...fx import (
    DEFAULT_TRANSFORMATION_MANAGER,
    AddPoptorchBlock,
    AddPoptorchBlocksInSeries,
    LinearToSerializedLinear,
    OutlineAttribute,
    RecomputationCheckpoint,
    TieWeights,
    VocabEmbeddingToSerializedEmbedding,
    symbolic_trace_pipelined_model,
)
from ...modeling_utils import PipelineMixin, get_layer_ipu, register


logger = logging.get_logger(__name__)


class GPT2PipelineMixin(PipelineMixin):
    @property
    def actual_vocab_size(self):
        return self.config.vocab_size

    @property
    def new_vocab_size(self):
        new_vocab_size = (
            math.ceil(self.config.vocab_size / self.ipu_config.embedding_serialization_factor)
            * self.ipu_config.embedding_serialization_factor
        )
        return new_vocab_size

    def resize_vocab(self, restore: bool):
        if restore:
            # Resize token embeddings back to origianl vocab_size
            if self.config.vocab_size > self.actual_vocab_size:
                self.resize_token_embeddings(self.actual_vocab_size)
        else:
            if self.new_vocab_size > self.actual_vocab_size:
                self.resize_token_embeddings(self.new_vocab_size)

    def get_transformations(self):
        log_insertions = self.ipu_config.log_insertions
        layer_ipu = get_layer_ipu(self.ipu_config.layers_per_ipu)
        transformations = [
            AddPoptorchBlock("Token Embedding", 0, "transformer.wte", log_insertions=log_insertions),
            AddPoptorchBlock("Position Embedding", 1, "transformer.wtp", log_insertions=log_insertions),
            OutlineAttribute("transformer.ln_f", "LayerNorm"),
            AddPoptorchBlocksInSeries("Layer", layer_ipu, r"transformer.h.[0-9]+", log_insertions=log_insertions),
            # Only one of the following AddPoptorchBlock, will actually add a block.
            AddPoptorchBlock("Score", layer_ipu[-1], "score", log_insertions=log_insertions),
            AddPoptorchBlock("Classifier", layer_ipu[-1], "classifier", log_insertions=log_insertions),
        ]
        if self.ipu_config.recompute_checkpoint_every_layer:
            transformations.append(
                RecomputationCheckpoint(
                    "transformer.h.[0-9]+",
                    to_exclude=f"transformer.h.{self.config.num_hidden_layers - 1}",
                )
            )
        if self.ipu_config.embedding_serialization_factor > 1:
            transformations.append(VocabEmbeddingToSerializedEmbedding())

        return transformations

    def parallelize(self):
        """
        Transform the GPT-2 model body to run in an IPU pipeline.
        - Adds pipeline stages to the model
        - (If enabled) Replaces the word embedding with a SerializedEmbedding
        - Adds recomputation checkpoints
        """
        PipelineMixin.parallelize(self)
        if self.ipu_config.embedding_serialization_factor > 1:
            self.resize_vocab(False)
        traced = symbolic_trace_pipelined_model(self)
        transformations = self.get_transformations()
        transformations += DEFAULT_TRANSFORMATION_MANAGER.get_reversible_transformations(
            self.ipu_config.optimization_level
        )
        composition = compose(*transformations)
        non_reversible_composition = DEFAULT_TRANSFORMATION_MANAGER.compose_non_reversible_transformations(
            self.ipu_config.optimization_level
        )
        traced = composition(traced)
        traced = non_reversible_composition(traced)
        return traced

    def deparallelize(self):
        """
        Undo the changes to the model done by `parallelize`.
        You should call this before doing `save_pretrained` so that the `model.state_dict` is
        fully compatible with the original model.
        """
        super().deparallelize()
        transformations = self.get_transformations()
        transformations = [t for t in transformations if isinstance(t, ReversibleTransformation)]
        transformations += DEFAULT_TRANSFORMATION_MANAGER.get_reversible_transformations(
            self.ipu_config.optimization_level
        )
        composition = compose(*transformations)
        self = composition(self, reverse=True)
        if self.ipu_config.embedding_serialization_factor > 1:
            self.resize_vocab(True)
        return self


@register(GPT2LMHeadModel)
class PipelinedGPT2LMHeadModel(GPT2LMHeadModel, GPT2PipelineMixin):
    def get_transformations(self):
        log_insertions = self.ipu_config.log_insertions
        layer_ipu = get_layer_ipu(self.ipu_config.layers_per_ipu)
        transformations = [
            AddPoptorchBlock("Token Embedding", 0, "transformer.wte", log_insertions=log_insertions),
            AddPoptorchBlock("Position Embedding", 1, "transformer.wtp", log_insertions=log_insertions),
            OutlineAttribute("transformer.ln_f", "LayerNorm"),
            AddPoptorchBlocksInSeries("Layer", layer_ipu, r"transformer.h.[0-9]+", log_insertions=log_insertions),
            AddPoptorchBlock("LM Head", 0, "lm_head", log_insertions=log_insertions),
        ]
        if self.ipu_config.recompute_checkpoint_every_layer:
            transformations.append(
                RecomputationCheckpoint(
                    "transformer.h.[0-9]+",
                    to_exclude=f"transformer.h.{self.config.num_hidden_layers - 1}",
                )
            )
        if self.ipu_config.embedding_serialization_factor > 1:
            transformations += [
                LinearToSerializedLinear("lm_head"),
                TieWeights("transformer.wte", "lm_head"),
            ]

        return transformations

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)
        if self.ipu_config.embedding_serialization_factor > 1 and self.config.vocab_size > self.actual_vocab_size:
            # Ignore the padding logits. Use masking because in-place modification on a slice is not supported yet.
            padding_mask = torch.cat(
                (
                    torch.ones(self.actual_vocab_size),
                    torch.zeros(self.config.vocab_size - self.actual_vocab_size),
                )
            ).to(dtype=lm_logits.dtype, device=lm_logits.device)
            lm_logits = lm_logits * padding_mask + (1 - padding_mask) * -10000.0

            # TODO: Use the following line instead to ignore the padding logits
            # lm_logits[:, :, self.actual_vocab_size:] = -10000

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n. Use roll() + ignore_index instead of slicing for better efficiency on IPUs.
            labels = torch.roll(labels, -1, 1)
            # By default the ignore_index of CrossEntropyLoss is -100
            labels[:, -1] = -100
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        if self.ipu_config.embedding_serialization_factor > 1 and self.config.vocab_size > self.actual_vocab_size:
            lm_logits = lm_logits[:, :, : self.actual_vocab_size]

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            if loss is not None:
                return (loss,) if self.training else (loss,) + output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits if loss is None else None,
            past_key_values=transformer_outputs.past_key_values if loss is None else None,
            hidden_states=transformer_outputs.hidden_states if loss is None else None,
            attentions=transformer_outputs.attentions if loss is None else None,
            cross_attentions=transformer_outputs.cross_attentions if loss is None else None,
        )


@register(GPT2ForSequenceClassification)
class PipelinedGPT2ForSequenceClassification(GPT2ForSequenceClassification, GPT2PipelineMixin):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = super().forward(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # By default use_cache=True and the model would return past_key_values, which could be very large and cause OOM.
        # To prevent this we only return loss and logits during training and evaluation (i.e. when there are labels).
        if not return_dict:
            loss, logits = outputs[0], outputs[1]
            return (loss, logits) if labels is not None else outputs

        return SequenceClassifierOutputWithPast(
            loss=outputs.loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values if labels is None else None,
            hidden_states=outputs.hidden_states if labels is None else None,
            attentions=outputs.attentions if labels is None else None,
        )


@register(GPT2ForTokenClassification)
class PipelinedGPT2ForTokenClassification(GPT2ForTokenClassification, GPT2PipelineMixin):
    pass
