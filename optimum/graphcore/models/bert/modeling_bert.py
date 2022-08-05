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
import torch.nn.functional as F

import poptorch
from optimum.utils import logging
from scipy.stats import truncnorm
from transformers import (
    BertForMaskedLM,
    BertForMultipleChoice,
    BertForPreTraining,
    BertForQuestionAnswering,
    BertForSequenceClassification,
    BertForTokenClassification,
)
from transformers.utils.fx import _gen_constructor_wrapper

from ....fx.optimization import ChangeTrueDivToMulByInverse, MergeLinears, compose
from ...fx.transformations import (
    AddPoptorchBlock,
    AddPoptorchBlocksInSeries,
    ClipValues,
    ClipValuesSymmetric,
    LinearToSerializedLinear,
    OutlineAttribute,
    RecomputationCheckpoint,
    TieWeights,
    TupleOutput,
    VocabEmbeddingToSerializedEmbedding,
)
from ...fx.utils import symbolic_trace_pipelined_model
from ...modeling_utils import OnehotGather, PipelineMixin, get_layer_ipu, register


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


@register(BertForPreTraining)
class PipelinedBertForPreTraining(BertForPreTraining, PipelineMixin):
    """
    BertForPretraining transformed to run in an IPU pipeline.

    Recommended usage:
    ```
    model = PipelinedBertForPretraining(config).parallelize().half().train()
    ```
    """

    def __init__(self, config):
        super().__init__(config)
        self.gather_indices = OnehotGather()

    def get_ops_to_wrap_for_tracing(self):
        return [
            ("torch.topk", *_gen_constructor_wrapper(torch.topk)),
            ("torch.nn.functional.one_hot", *_gen_constructor_wrapper(torch.nn.functional.one_hot)),
        ]

    def get_transformations(self):
        log_insertions = self.ipu_config.log_insertions
        layer_ipu = get_layer_ipu(self.ipu_config.layers_per_ipu)
        transformations = [
            AddPoptorchBlock("Embedding", 0, "bert.embeddings", log_insertions=log_insertions),
            OutlineAttribute("bert.embeddings.LayerNorm", "Embedding"),
            AddPoptorchBlocksInSeries(
                "Encoder", layer_ipu, r"bert.encoder.layer.[0-9]+", log_insertions=log_insertions
            ),
            AddPoptorchBlock("Pooler Output", 0, "bert.pooler", log_insertions=log_insertions),
            AddPoptorchBlock("Classifier Output", 0, "cls", log_insertions=log_insertions),
        ]
        if self.ipu_config.recompute_checkpoint_every_layer:
            transformations.append(
                RecomputationCheckpoint(
                    "bert.encoder.layer.[0-9]+", to_exclude=f"bert.encoder.layer.{self.config.num_hidden_layers - 1}"
                )
            )
        if self.ipu_config.embedding_serialization_factor > 1:
            transformations += [
                LinearToSerializedLinear("cls.predictions.decoder"),
                TieWeights("bert.embeddings.word_embeddings", "cls.predictions.decoder"),
            ]
        return transformations

    def parallelize(self):
        """
        Transform the model to run in an IPU pipeline.
        - Adds pipeline stages to the model
        - Replaces self-attention layers with fused-qkv self-attention layers
        - (If enabled) Replaces the word embedding projection with a SerializedLinear layer
        - Adds recomputation checkpoints
        """
        super().parallelize()
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
        compatible with the original model.
        """
        super().deparallelize()
        transformations = self.get_transformations()
        transformations += _OPTIMIZATION_TRANSFORMATIONS
        composition = compose(*transformations)
        self = composition(self, reverse=True)
        return self

    def _init_weights(self, module):
        """Initialize the weights"""

        def truncated_normal_(tensor, mean=0, std=1):
            """
            Truncated Normal distribution, truncated at 2 sigma
            """
            r = torch.tensor(truncnorm.rvs(-2, 2, loc=mean, scale=std, size=tensor.shape))
            tensor.data.copy_(r)

        if isinstance(module, nn.Linear):
            truncated_normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            truncated_normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        next_sentence_label: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BertForPreTrainingOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output, pooled_output = outputs[:2]

        if labels is not None:
            if hasattr(self.config, "max_num_masked_tokens"):
                # Select only the masked tokens for the classifier
                labels, positions = torch.topk(labels, k=self.config.max_num_masked_tokens, dim=1)
                sequence_output = self.gather_indices(sequence_output, positions)

        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        total_loss = None
        if labels is not None and next_sentence_label is not None:
            masked_lm_loss = F.cross_entropy(
                prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1),
            ).float()
            next_sentence_loss = F.cross_entropy(
                seq_relationship_score.view(-1, 2), next_sentence_label.view(-1)
            ).float()
            total_loss = poptorch.identity_loss(masked_lm_loss + next_sentence_loss, reduction="none")

        # If labels are provided (training mode) only output the loss
        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return (total_loss,) if total_loss is not None else output

        return BertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores if total_loss is None else None,
            seq_relationship_logits=seq_relationship_score if total_loss is None else None,
            hidden_states=outputs.hidden_states if total_loss is None else None,
            attentions=outputs.attentions if total_loss is None else None,
        )


@register(BertForMaskedLM)
class PipelinedBertForMaskedLM(BertForMaskedLM, PipelineMixin):
    """
    BertForMaskedLM transformed to run in an IPU pipeline.

    Recommended usage:
    ```
    model = PipelinedBertForMaskedLM(config).parallelize().half().train()
    ```
    """

    def __init__(self, config):
        super().__init__(config)
        self.gather_indices = OnehotGather()

    def get_ops_to_wrap_for_tracing(self):
        return [
            ("torch.topk", *_gen_constructor_wrapper(torch.topk)),
            ("torch.nn.functional.one_hot", *_gen_constructor_wrapper(torch.nn.functional.one_hot)),
        ]

    def get_transformations(self):
        log_insertions = self.ipu_config.log_insertions
        layer_ipu = get_layer_ipu(self.ipu_config.layers_per_ipu)
        transformations = [
            AddPoptorchBlock("Embedding", 0, "bert.embeddings", log_insertions=log_insertions),
            OutlineAttribute("bert.embeddings.LayerNorm", "Embedding"),
            AddPoptorchBlocksInSeries(
                "Encoder", layer_ipu, r"bert.encoder.layer.[0-9]+", log_insertions=log_insertions
            ),
            AddPoptorchBlock("Classifier Output", 0, "cls", log_insertions=log_insertions),
        ]
        if self.ipu_config.recompute_checkpoint_every_layer:
            transformations.append(
                RecomputationCheckpoint(
                    "bert.encoder.layer.[0-9]+", to_exclude=f"bert.encoder.layer.{self.config.num_hidden_layers - 1}"
                )
            )
        if self.ipu_config.embedding_serialization_factor > 1:
            transformations += [
                LinearToSerializedLinear("cls.predictions.decoder"),
                TieWeights("bert.embeddings.word_embeddings", "cls.predictions.decoder"),
            ]
        return transformations

    def parallelize(self):
        """
        Transform the model to run in an IPU pipeline.
        - Adds pipeline stages to the model
        - Replaces self-attention layers with fused-qkv self-attention layers
        - (If enabled) Replaces the word embedding projection with a SerializedLinear layer
        - Adds recomputation checkpoints
        """
        super().parallelize()
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
        compatible with the original model.
        """
        super().deparallelize()
        transformations = self.get_transformations()
        transformations += _OPTIMIZATION_TRANSFORMATIONS
        composition = compose(*transformations)
        self = composition(self, reverse=True)
        return self

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.training:
            outputs = self.bert(
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

            prediction_scores = self.cls(sequence_output)
            outputs = (prediction_scores,) + outputs[2:]

            masked_lm_loss = F.cross_entropy(
                prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
            ).float()
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


class BertPipelineMixin(PipelineMixin):
    def get_transformations(self):
        log_insertions = self.ipu_config.log_insertions
        layer_ipu = get_layer_ipu(self.ipu_config.layers_per_ipu)
        last_ipu = len(self.ipu_config.layers_per_ipu) - 1
        transformations = [
            AddPoptorchBlock("Embedding", 0, "bert.embeddings", log_insertions=log_insertions),
            OutlineAttribute("bert.embeddings.LayerNorm", "Embedding"),
            AddPoptorchBlocksInSeries(
                "Encoder", layer_ipu, r"bert.encoder.layer.[0-9]+", log_insertions=log_insertions
            ),
            # Only one of the following AddPoptorchBlock, will actually add a block.
            AddPoptorchBlock("Classifier Output", last_ipu, "classifier", log_insertions=log_insertions),
            AddPoptorchBlock("QA Outputs", last_ipu, "qa_outputs", log_insertions=log_insertions),
        ]
        if self.ipu_config.recompute_checkpoint_every_layer:
            transformations.append(
                RecomputationCheckpoint(
                    "bert.encoder.layer.[0-9]+", to_exclude=f"bert.encoder.layer.{self.config.num_hidden_layers - 1}"
                )
            )
        if self.ipu_config.embedding_serialization_factor > 1:
            transformations.append(VocabEmbeddingToSerializedEmbedding())
        return transformations

    def parallelize(self):
        """
        Transform the model to run in an IPU pipeline.
        - Adds pipeline stages to the model
        - Replaces self-attention layers with fused-qkv self-attention layers
        - (If enabled) Replaces the word embedding with a SerializedEmbedding
        - Adds recomputation checkpoints
        """
        super().parallelize()
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
        compatible with the original model.
        """
        super().deparallelize()
        transformations = self.get_transformations()
        transformations += _OPTIMIZATION_TRANSFORMATIONS
        composition = compose(*transformations)
        self = composition(self, reverse=True)
        return self


@register(BertForSequenceClassification)
class PipelinedBertForSequenceClassification(BertForSequenceClassification, BertPipelineMixin):
    """
    BertForSequenceClassification transformed to run in an IPU pipeline.

    Recommended usage:
    ```
    model = PipelinedBertForSequenceClassification(config).parallelize().half()
    ```
    """

    pass


@register(BertForMultipleChoice)
class PipelinedBertForMultipleChoice(BertForMultipleChoice, BertPipelineMixin):
    """
    BertForMultipleChoice transformed to run in an IPU pipeline.

    Recommended usage:
    ```
    model = PipelinedBertForMultipleChoice(config).parallelize().half()
    ```
    """

    pass


@register(BertForTokenClassification)
class PipelinedBertForTokenClassification(BertForTokenClassification, BertPipelineMixin):
    """
    BertForTokenClassification transformed to run in an IPU pipeline.

    Recommended usage:
    ```
    model = PipelinedBertForTokenClassification(config).parallelize().half()
    ```
    """

    pass


@register(BertForQuestionAnswering)
class PipelinedBertForQuestionAnswering(BertForQuestionAnswering, BertPipelineMixin):
    """
    BertForQuestionAnswering transformed to run in an IPU pipeline.

    Recommended usage:
    ```
    model = PipelinedBertForQuestionAnswering(config).parallelize().half()
    ```
    """

    @property
    def input_names(self):
        return ["input_ids", "attention_mask", "token_type_ids", "start_positions", "end_positions"]

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
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
