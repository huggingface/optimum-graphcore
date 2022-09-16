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
from typing import Optional, Tuple

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
from transformers.models.distilbert.modeling_distilbert import MultiHeadSelfAttention

from ...modeling_utils import (
    OnehotGather,
    PipelineMixin,
    SerializedEmbedding,
    SerializedLinear,
    get_layer_ipu,
    recomputation_checkpoint,
    register,
)


logger = logging.get_logger(__name__)

FLOAT16_LIMIT = 1e4


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
        # Always use -1e4 to avoid NaN issues in fp16.
        mask = FLOAT16_LIMIT * (mask - 1)
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
    def parallelize(self):
        """
        Transform the model to run in an IPU pipeline.
        - Adds pipeline stages to the model
        - Adds recomputation checkpoints
        """
        super().parallelize()

        for layer in self.distilbert.transformer.layer:
            layer.attention.__class__ = IPUMultiHeadSelfAttention

        logger.info("-------------------- Device Allocation --------------------")
        logger.info("Embedding --> IPU 0")
        is_masked_lm = isinstance(self, DistilBertForMaskedLM)
        if self.ipu_config.embedding_serialization_factor > 1 and not is_masked_lm:
            self.distilbert.embeddings.word_embeddings = SerializedEmbedding(
                self.distilbert.embeddings.word_embeddings, self.ipu_config.embedding_serialization_factor
            )
        self.distilbert.embeddings = poptorch.BeginBlock(self.distilbert.embeddings, "Embedding", 0)

        layer_ipu = get_layer_ipu(self.ipu_config.layers_per_ipu)
        for index, layer in enumerate(self.distilbert.transformer.layer):
            ipu = layer_ipu[index]
            if self.ipu_config.recompute_checkpoint_every_layer and index != self.config.num_hidden_layers - 1:
                recomputation_checkpoint(layer)
            self.distilbert.transformer.layer[index] = poptorch.BeginBlock(layer, f"Encoder{index}", ipu_id=ipu)
            logger.info(f"Encoder {index:<2} --> IPU {ipu}")

        return self

    def deparallelize(self):
        """
        Undo the changes to the model done by `parallelize`.
        You should call this before doing `save_pretrained` so that the `model.state_dict` is
        compatible with the original model.
        """
        super().deparallelize()

        for layer in self.distilbert.transformer.layer:
            layer.attention.__class__ = MultiHeadSelfAttention

        is_masked_lm = isinstance(self, DistilBertForMaskedLM)
        if self.ipu_config.embedding_serialization_factor > 1 and not is_masked_lm:
            self.distilbert.embeddings.word_embeddings = self.distilbert.embeddings.word_embeddings.deserialize()

        return self


@register(DistilBertForMaskedLM)
class PipelinedDistilBertForMaskedLM(DistilBertForMaskedLM, DistilBertPipelineMixin):
    def __init__(self, config):
        super().__init__(config)
        self.gather_indices = OnehotGather()

    def parallelize(self):
        super().parallelize()

        if self.ipu_config.embedding_serialization_factor > 1:
            serialized_vocab_projector = SerializedLinear(
                self.config.dim,
                self.config.vocab_size,
                self.ipu_config.embedding_serialization_factor,
                bias=True,
                mode=poptorch.MatMulSerializationMode.OutputChannels,
            )
            serialized_vocab_projector.load_state_dict(self.vocab_projector.state_dict())
            self.vocab_projector = serialized_vocab_projector
            self.tie_weights()

        logger.info("LM Head --> IPU 0")
        self.vocab_transform = poptorch.BeginBlock(self.vocab_transform, "LM Head", ipu_id=0)
        self.vocab_layer_norm = poptorch.BeginBlock(self.vocab_layer_norm, "LM Head", ipu_id=0)
        self.vocab_projector = poptorch.BeginBlock(self.vocab_projector, "LM Head", ipu_id=0)
        logger.info("-----------------------------------------------------------")
        return self

    def deparallelize(self):
        super().deparallelize()

        if self.ipu_config.embedding_serialization_factor > 1:
            vocab_projector = nn.Linear(
                self.config.hidden_size,
                self.config.vocab_size,
                bias=True,
            )
            vocab_projector.load_state_dict(self.vocab_projector.state_dict())
            self.vocab_projector = vocab_projector
            self.tie_weights()

    def forward(self, input_ids, attention_mask, labels=None):
        if self.training:
            outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
            sequence_output = outputs[0]

            if hasattr(self.config, "max_num_masked_tokens"):
                # Select only the masked tokens for the classifier
                labels, positions = torch.topk(labels, k=self.config.max_num_masked_tokens, dim=1)
                sequence_output = self.gather_indices(sequence_output, positions)

            prediction_logits = self.vocab_transform(sequence_output)  # (bs, seq_length, dim)
            prediction_logits = self.activation(prediction_logits)  # (bs, seq_length, dim)
            prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
            prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)

            masked_lm_loss = F.cross_entropy(prediction_logits.view(-1, self.config.vocab_size), labels.view(-1))
            return (masked_lm_loss,)
        else:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=False,
            )


@register(DistilBertForSequenceClassification)
class PipelinedDistilBertForSequenceClassification(DistilBertForSequenceClassification, DistilBertPipelineMixin):
    def parallelize(self):
        super().parallelize()

        last_ipu = self.ipu_config.ipus_per_replica - 1
        logger.info(f"Classifier --> IPU {last_ipu}")
        self.pre_classifier = poptorch.BeginBlock(self.pre_classifier, "Classifier", ipu_id=last_ipu)
        self.classifier = poptorch.BeginBlock(self.classifier, "Classifier", ipu_id=last_ipu)
        logger.info("-----------------------------------------------------------")
        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        return super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=False)


@register(DistilBertForQuestionAnswering)
class PipelinedDistilBertForQuestionAnswering(DistilBertForQuestionAnswering, DistilBertPipelineMixin):
    def parallelize(self):
        super().parallelize()

        last_ipu = self.ipu_config.ipus_per_replica - 1
        logger.info(f"QA Outputs --> IPU {last_ipu}")
        self.qa_outputs = poptorch.BeginBlock(self.qa_outputs, "QA Outputs", ipu_id=last_ipu)
        logger.info("-----------------------------------------------------------")
        return self

    def forward(self, input_ids, attention_mask, start_positions=None, end_positions=None):
        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions,
            return_dict=False,
        )
        if start_positions is not None and end_positions is not None:
            output = (poptorch.identity_loss(output[0], reduction="none"),) + output[1:]
        return output


@register(DistilBertForTokenClassification)
class PipelinedDistilBertForTokenClassification(DistilBertForTokenClassification, DistilBertPipelineMixin):
    def parallelize(self):
        super().parallelize()

        last_ipu = self.ipu_config.ipus_per_replica - 1
        logger.info(f"Classifier --> IPU {last_ipu}")
        self.classifier = poptorch.BeginBlock(self.classifier, "Classifier", ipu_id=last_ipu)
        logger.info("-----------------------------------------------------------")
        return self

    def forward(self, input_ids, attention_mask, labels=None):
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=False,
        )


@register(DistilBertForMultipleChoice)
class PipelinedDistilBertForMultipleChoice(DistilBertForMultipleChoice, DistilBertPipelineMixin):
    def parallelize(self):
        super().parallelize()

        last_ipu = self.ipu_config.ipus_per_replica - 1
        logger.info(f"Classifier --> IPU {last_ipu}")
        self.pre_classifier = poptorch.BeginBlock(self.pre_classifier, "Classifier", ipu_id=last_ipu)
        self.classifier = poptorch.BeginBlock(self.classifier, "Classifier", ipu_id=last_ipu)
        logger.info("-----------------------------------------------------------")
        return self

    def forward(self, input_ids, attention_mask, labels=None):
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=False,
        )