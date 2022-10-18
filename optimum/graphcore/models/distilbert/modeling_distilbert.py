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
from transformers import (
    DistilBertForMaskedLM,
    DistilBertForMultipleChoice,
    DistilBertForQuestionAnswering,
    DistilBertForSequenceClassification,
    DistilBertForTokenClassification,
)
from transformers.modeling_outputs import MaskedLMOutput, QuestionAnsweringModelOutput
from transformers.models.distilbert.modeling_distilbert import MultiHeadSelfAttention

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
]


# TODO: should we make a fx transformation for this?
class IPUMultiHeadSelfAttention(MultiHeadSelfAttention):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Parameters:
            query: torch.tensor(bs, seq_length, dim)
            key: torch.tensor(bs, seq_length, dim)
            value: torch.tensor(bs, seq_length, dim)
            mask: torch.tensor(bs, seq_length)

        Returns:
            weights: torch.tensor(bs, n_heads, seq_length, seq_length) Attention weights context: torch.tensor(bs,
            seq_length, dim) Contextualized layer. Optional: only if `output_attentions=True`
        """
        bs, q_length, dim = query.size()
        k_length = key.size(1)
        # assert dim == self.dim, f'Dimensions do not match: {dim} input vs {self.dim} configured'
        # assert key.size() == value.size()

        dim_per_head = self.dim // self.n_heads

        mask_reshp = (bs, 1, 1, k_length)

        def shape(x: torch.Tensor) -> torch.Tensor:
            """separate heads"""
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x: torch.Tensor) -> torch.Tensor:
            """group heads"""
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)

        q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)

        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0 for
        # positions we want to attend and -10000 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        mask = (1 - mask) * -10000
        mask = mask.view(mask_reshp).expand_as(scores)  # (bs, n_heads, q_length, k_length)
        scores = scores + mask  # (bs, n_heads, q_length, k_length)

        weights = nn.functional.softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length)
        weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
        context = unshape(context)  # (bs, q_length, dim)
        context = self.out_lin(context)  # (bs, q_length, dim)

        if output_attentions:
            return (context, weights)
        else:
            return (context,)


class DistilBertPipelineMixin(PipelineMixin):
    def get_transformations(self):
        log_insertions = self.ipu_config.log_insertions
        layer_ipu = get_layer_ipu(self.ipu_config.layers_per_ipu)
        last_ipu = len(self.ipu_config.layers_per_ipu) - 1
        transformations = [
            AddPoptorchBlock("Embedding", 0, "distilbert.embeddings", log_insertions=log_insertions),
            OutlineAttribute("distilbert.embeddings.LayerNorm", "Embedding"),
            AddPoptorchBlocksInSeries(
                "Encoder", layer_ipu, r"distilbert.transformer.layer.[0-9]+", log_insertions=log_insertions
            ),
            # Only one of the following AddPoptorchBlock, will actually add a block.
            AddPoptorchBlock("Classifier Output", last_ipu, "classifier", log_insertions=log_insertions),
            AddPoptorchBlock("QA Outputs", last_ipu, "qa_outputs", log_insertions=log_insertions),
        ]
        if self.ipu_config.recompute_checkpoint_every_layer:
            transformations.append(
                RecomputationCheckpoint(
                    "distilbert.transformer.layer.[0-9]+",
                    to_exclude=f"distilbert.transformer.layer.{self.config.num_hidden_layers - 1}",
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
        for mod in self.modules():
            if isinstance(mod, MultiHeadSelfAttention):
                mod.__class__ = IPUMultiHeadSelfAttention
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
        for mod in self.modules():
            if isinstance(mod, IPUMultiHeadSelfAttention):
                mod.__class__ = MultiHeadSelfAttention
        transformations = self.get_transformations()
        transformations += _OPTIMIZATION_TRANSFORMATIONS
        composition = compose(*transformations)
        self = composition(self, reverse=True)
        return self


@register(DistilBertForMaskedLM)
class PipelinedDistilBertForMaskedLM(DistilBertForMaskedLM, DistilBertPipelineMixin):
    def __init__(self, config):
        super().__init__(config)
        self.gather_indices = OnehotGather()

    # TODO: validate that.
    def get_transformations(self):
        log_insertions = self.ipu_config.log_insertions
        layer_ipu = get_layer_ipu(self.ipu_config.layers_per_ipu)
        transformations = [
            AddPoptorchBlock("Embedding", 0, "distilbert.embeddings", log_insertions=log_insertions),
            OutlineAttribute("distilbert.embeddings.LayerNorm", "Embedding"),
            AddPoptorchBlocksInSeries(
                "Encoder", layer_ipu, r"distilbert.encoder.layer.[0-9]+", log_insertions=log_insertions
            ),
            AddPoptorchBlock("Classifier Output", 0, "cls", log_insertions=log_insertions),
        ]
        if self.ipu_config.recompute_checkpoint_every_layer:
            transformations.append(
                RecomputationCheckpoint(
                    "distilbert.encoder.layer.[0-9]+",
                    to_exclude=f"distilbert.encoder.layer.{self.config.num_hidden_layers - 1}",
                )
            )
        if self.ipu_config.embedding_serialization_factor > 1:
            transformations += [
                LinearToSerializedLinear("cls.predictions.decoder"),
                TieWeights("distilbert.embeddings.word_embeddings", "cls.predictions.decoder"),
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
        for mod in self.modules():
            if isinstance(mod, MultiHeadSelfAttention):
                mod.__class__ = IPUMultiHeadSelfAttention
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
        for mod in self.modules():
            if isinstance(mod, IPUMultiHeadSelfAttention):
                mod.__class__ = MultiHeadSelfAttention
        return self

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[MaskedLMOutput, Tuple[torch.Tensor, ...]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            dlbrt_output = self.distilbert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = dlbrt_output[0]  # (bs, seq_length, dim)

            if hasattr(self.config, "max_num_masked_tokens"):
                # Select only the masked tokens for the classifier
                labels, positions = torch.topk(labels, k=self.config.max_num_masked_tokens, dim=1)
                hidden_states = self.gather_indices(hidden_states, positions)

            prediction_logits = self.vocab_transform(hidden_states)  # (bs, seq_length, dim)
            prediction_logits = self.activation(prediction_logits)  # (bs, seq_length, dim)
            prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
            prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)

            masked_lm_loss = F.cross_entropy(prediction_logits.view(-1, self.config.vocab_size), labels.view(-1))
            # When training only return the loss
            if return_dict:
                return MaskedLMOutput(loss=masked_lm_loss)
            else:
                return (masked_lm_loss,)
        else:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )


@register(DistilBertForSequenceClassification)
class PipelinedDistilBertForSequenceClassification(DistilBertForSequenceClassification, DistilBertPipelineMixin):
    pass


@register(DistilBertForQuestionAnswering)
class PipelinedDistilBertForQuestionAnswering(DistilBertForQuestionAnswering, DistilBertPipelineMixin):
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[QuestionAnsweringModelOutput, Tuple[torch.Tensor, ...]]:
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
            input_ids=input_ids,
            attention_mask=attention_mask,
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


@register(DistilBertForTokenClassification)
class PipelinedDistilBertForTokenClassification(DistilBertForTokenClassification, DistilBertPipelineMixin):
    pass


@register(DistilBertForMultipleChoice)
class PipelinedDistilBertForMultipleChoice(DistilBertForMultipleChoice, DistilBertPipelineMixin):
    pass
