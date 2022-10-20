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

from typing import Optional, Tuple, Union

import torch
from torch.nn import CrossEntropyLoss

import poptorch
from transformers import (
    RobertaForMaskedLM,
    RobertaForMultipleChoice,
    RobertaForQuestionAnswering,
    RobertaForSequenceClassification,
    RobertaForTokenClassification,
)
from transformers.modeling_outputs import MaskedLMOutput, QuestionAnsweringModelOutput

from ....fx.optimization import compose
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
from ...modeling_utils import OnehotGather, PipelineMixin, get_layer_ipu, register


logger = logging.get_logger(__name__)


class RobertaPipelineMixin(PipelineMixin):
    def get_transformations(self):
        log_insertions = self.ipu_config.log_insertions
        layer_ipu = get_layer_ipu(self.ipu_config.layers_per_ipu)
        last_ipu = len(self.ipu_config.layers_per_ipu) - 1
        transformations = [
            AddPoptorchBlock("Embedding", 0, "roberta.embeddings", log_insertions=log_insertions),
            OutlineAttribute("roberta.embeddings.LayerNorm", "Embedding"),
            AddPoptorchBlocksInSeries(
                "Encoder", layer_ipu, r"roberta.encoder.layer.[0-9]+", log_insertions=log_insertions
            ),
            # Only one of the following AddPoptorchBlock, will actually add a block.
            AddPoptorchBlock("Classifier Output", last_ipu, "classifier", log_insertions=log_insertions),
            AddPoptorchBlock("QA Outputs", last_ipu, "qa_outputs", log_insertions=log_insertions),
        ]
        if self.ipu_config.recompute_checkpoint_every_layer:
            transformations.append(
                RecomputationCheckpoint(
                    "roberta.encoder.layer.[0-9]+",
                    to_exclude=f"roberta.encoder.layer.{self.config.num_hidden_layers - 1}",
                )
            )
        if self.ipu_config.embedding_serialization_factor > 1:
            transformations.append(VocabEmbeddingToSerializedEmbedding())
        return transformations

    def parallelize(self):
        """
        Transform the Roberta model body to run in an IPU pipeline.
        - Adds pipeline stages to the model
        - (If enabled) Replaces the word embedding with a SerializedEmbedding
        - Adds recomputation checkpoints
        """
        super().parallelize()
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
        transformations += DEFAULT_TRANSFORMATION_MANAGER.get_reversible_transformations(
            self.ipu_config.optimization_level
        )
        composition = compose(*transformations)
        self = composition(self, reverse=True)
        return self


@register(RobertaForMaskedLM)
class PipelinedRobertaForMaskedLM(RobertaForMaskedLM, RobertaPipelineMixin):
    """
    RobertaForMaskedLM transformed to run in an IPU pipeline.

    Recommended usage:
    ```
    model = PipelinedRobertaForMaskedLM(config).parallelize().half()
    ```
    """

    def __init__(self, config):
        super().__init__(config)
        self.gather_indices = OnehotGather()

    def get_transformations(self):
        log_insertions = self.ipu_config.log_insertions
        layer_ipu = get_layer_ipu(self.ipu_config.layers_per_ipu)
        transformations = [
            AddPoptorchBlock("Embedding", 0, "roberta.embeddings", log_insertions=log_insertions),
            OutlineAttribute("roberta.embeddings.LayerNorm", "Embedding"),
            AddPoptorchBlocksInSeries(
                "Encoder", layer_ipu, r"roberta.encoder.layer.[0-9]+", log_insertions=log_insertions
            ),
            AddPoptorchBlock("LM Head", 0, "lm_head", log_insertions=log_insertions),
        ]
        if self.ipu_config.recompute_checkpoint_every_layer:
            transformations.append(
                RecomputationCheckpoint(
                    "roberta.encoder.layer.[0-9]+",
                    to_exclude=f"roberta.encoder.layer.{self.config.num_hidden_layers - 1}",
                )
            )
        if self.ipu_config.embedding_serialization_factor > 1:
            transformations += [
                LinearToSerializedLinear("lm_head.decoder"),
                TieWeights("roberta.embeddings.word_embeddings", "lm_head.decoder"),
            ]
        return transformations

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output = outputs[0]

            if hasattr(self.config, "max_num_masked_tokens"):
                # Select only the masked tokens for the classifier
                labels, positions = torch.topk(labels, k=self.config.max_num_masked_tokens, dim=1)
                sequence_output = self.gather_indices(sequence_output, positions)

            prediction_scores = self.lm_head(sequence_output)

            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            # When training only return the loss
            if return_dict:
                return MaskedLMOutput(loss=masked_lm_loss)
            else:
                return (masked_lm_loss,)

        else:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                labels=labels,
                return_dict=return_dict,
            )


@register(RobertaForSequenceClassification)
class PipelinedRobertaForSequenceClassification(RobertaForSequenceClassification, RobertaPipelineMixin):
    """
    RobertaForSequenceClassificiation transformed to run in an IPU pipeline.

    Recommended usage:
    ```
    model = PipelinedRobertaForSequenceClassification(config).parallelize().half()
    ```
    """


@register(RobertaForMultipleChoice)
class PipelinedRobertaForMultipleChoice(RobertaForMultipleChoice, RobertaPipelineMixin):
    """
    RobertaForMultipleChoice transformed to run in an IPU pipeline.

    Recommended usage:
    ```
    model = PipelinedRobertaForMultipleChoice(config).parallelize().half()
    ```
    """


@register(RobertaForTokenClassification)
class PipelinedRobertaForTokenClassification(RobertaForTokenClassification, RobertaPipelineMixin):
    """
    RobertaForTokenClassification transformed to run in an IPU pipeline.

    Recommended usage:
    ```
    model = PipelinedRobertaForTokenClassification(config).parallelize().half()
    ```
    """


@register(RobertaForQuestionAnswering)
class PipelinedRobertaForQuestionAnswering(RobertaForQuestionAnswering, RobertaPipelineMixin):
    """
    RobertaForQuestionAnswering transformed to run in an IPU pipeline.

    Recommended usage:
    ```
    model = PipelinedRobertaForQuestionAnswering(config).parallelize().half()
    ```
    """

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        output = super().forward(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            start_positions=start_positions,
            end_positions=end_positions,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if start_positions is not None and end_positions is not None:
            output = (poptorch.identity_loss(output[0], reduction="none"),) + output[1:]
        return output
