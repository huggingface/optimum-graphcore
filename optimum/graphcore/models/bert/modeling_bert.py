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
from transformers.models.bert.modeling_bert import BertSelfAttention

from ...modeling_utils import (
    OnehotGather,
    PipelineMixin,
    SerializedEmbedding,
    SerializedLinear,
    get_layer_ipu,
    outline_attribute,
    recomputation_checkpoint,
    register,
)
from .bert_fused_attention import BertFusedSelfAttention


logger = logging.get_logger(__name__)


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

    def parallelize(self):
        """
        Transform the model to run in an IPU pipeline.
        - Adds pipeline stages to the model
        - Replaces self-attention layers with fused-qkv self-attention layers
        - (If enabled) Replaces the word embedding projection with a SerializedLinear layer
        - Adds recomputation checkpoints
        """
        super().parallelize()

        # Use faster fused-qkv self-attention
        for layer in self.bert.encoder.layer:
            layer.attention.self.__class__ = BertFusedSelfAttention

        if self.ipu_config.embedding_serialization_factor > 1:
            serialized_decoder = SerializedLinear(
                self.config.hidden_size,
                self.config.vocab_size,
                self.ipu_config.embedding_serialization_factor,
                bias=True,
                mode=poptorch.MatMulSerializationMode.OutputChannels,
            )
            serialized_decoder.load_state_dict(self.cls.predictions.decoder.state_dict())
            self.cls.predictions.decoder = serialized_decoder
            self.tie_weights()

        layer_ipu = get_layer_ipu(self.ipu_config.layers_per_ipu)

        logger.info("-------------------- Device Allocation --------------------")
        logger.info("Embedding --> IPU 0")
        self.bert.embeddings = poptorch.BeginBlock(self.bert.embeddings, "Embedding", ipu_id=0)
        # Preventing the embeddings.LayerNorm from being outlined with the encoder.layer.LayerNorm
        # improves the tile mapping of the pipeline stashes
        hs = outline_attribute(self.bert.embeddings.LayerNorm, "embeddings")
        self._hooks.extend(hs)

        for index, layer in enumerate(self.bert.encoder.layer):
            ipu = layer_ipu[index]
            if self.ipu_config.recompute_checkpoint_every_layer:
                h = recomputation_checkpoint(layer)
                self._hooks.append(h)
            self.bert.encoder.layer[index] = poptorch.BeginBlock(layer, f"Encoder{index}", ipu_id=ipu)
            logger.info(f"Encoder {index:<2} --> IPU {ipu}")

        logger.info("Pooler --> IPU 0")
        self.bert.pooler = poptorch.BeginBlock(self.bert.pooler, "Pooler", ipu_id=0)

        logger.info("Classifier --> IPU 0")
        self.cls = poptorch.BeginBlock(self.cls, "Classifier", ipu_id=0)
        logger.info("-----------------------------------------------------------")
        return self

    def deparallelize(self):
        """
        Undo the changes to the model done by `parallelize`.
        You should call this before doing `save_pretrained` so that the `model.state_dict` is
        compatible with the original model.
        """
        super().deparallelize()

        for layer in self.bert.encoder.layer:
            layer.attention.self.__class__ = BertSelfAttention

        if self.ipu_config.embedding_serialization_factor > 1:
            decoder = nn.Linear(
                self.config.hidden_size,
                self.config.vocab_size,
                bias=True,
            )
            decoder.load_state_dict(self.cls.predictions.decoder.state_dict())
            self.cls.predictions.decoder = decoder
            self.tie_weights()
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
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        next_sentence_label=None,
    ):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output, pooled_output = output[:2]

        if labels is not None:
            # Select only the masked tokens for the classifier
            max_number_of_masked_tokens = int(labels.size(1) * 0.25)
            masked_lm_labels, masked_lm_positions = torch.topk(labels, k=max_number_of_masked_tokens, dim=1)
            masked_output = self.gather_indices(sequence_output, masked_lm_positions)
        else:
            # This case should never happen during training
            masked_output = sequence_output

        prediction_scores, sequential_relationship_score = self.cls(masked_output, pooled_output)
        output = (
            prediction_scores,
            sequential_relationship_score,
        ) + output[2:]

        if labels is not None and next_sentence_label is not None:
            masked_lm_loss = F.cross_entropy(
                prediction_scores.view(-1, self.config.vocab_size),
                masked_lm_labels.view(-1),
                ignore_index=-100,
            ).float()
            next_sentence_loss = F.cross_entropy(
                sequential_relationship_score.view(-1, 2), next_sentence_label.view(-1)
            ).float()
            total_loss = poptorch.identity_loss(masked_lm_loss + next_sentence_loss, reduction="none")
            return (total_loss, masked_lm_loss, next_sentence_loss)

        return output


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

    def parallelize(self):
        """
        Transform the model to run in an IPU pipeline.
        - Adds pipeline stages to the model
        - Replaces self-attention layers with fused-qkv self-attention layers
        - (If enabled) Replaces the word embedding projection with a SerializedLinear layer
        - Adds recomputation checkpoints
        """
        super().parallelize()

        # Use faster fused-qkv self-attention
        for layer in self.bert.encoder.layer:
            layer.attention.self.__class__ = BertFusedSelfAttention

        if self.ipu_config.embedding_serialization_factor > 1:
            serialized_decoder = SerializedLinear(
                self.config.hidden_size,
                self.config.vocab_size,
                self.ipu_config.embedding_serialization_factor,
                bias=True,
                mode=poptorch.MatMulSerializationMode.OutputChannels,
            )
            serialized_decoder.load_state_dict(self.cls.predictions.decoder.state_dict())
            self.cls.predictions.decoder = serialized_decoder
            self.tie_weights()

        layer_ipu = get_layer_ipu(self.ipu_config.layers_per_ipu)

        logger.info("-------------------- Device Allocation --------------------")
        logger.info("Embedding  --> IPU 0")
        self.bert.embeddings = poptorch.BeginBlock(self.bert.embeddings, "Embedding", ipu_id=0)
        # Preventing the embeddings.LayerNorm from being outlined with the encoder.layer.LayerNorm
        # improves the tile mapping of the pipeline stashes
        hs = outline_attribute(self.bert.embeddings.LayerNorm, "embeddings")
        self._hooks.extend(hs)

        for index, layer in enumerate(self.bert.encoder.layer):
            ipu = layer_ipu[index]
            if self.ipu_config.recompute_checkpoint_every_layer:
                h = recomputation_checkpoint(layer)
                self._hooks.append(h)
            self.bert.encoder.layer[index] = poptorch.BeginBlock(layer, f"Encoder{index}", ipu_id=ipu)
            logger.info(f"Encoder {index:<2} --> IPU {ipu}")

        logger.info("Classifier --> IPU 0")
        self.cls = poptorch.BeginBlock(self.cls, "Classifier", ipu_id=0)
        logger.info("-----------------------------------------------------------")
        return self

    def deparallelize(self):
        """
        Undo the changes to the model done by `parallelize`.
        You should call this before doing `save_pretrained` so that the `model.state_dict` is
        compatible with the original model.
        """
        super().deparallelize()

        for layer in self.bert.encoder.layer:
            layer.attention.self.__class__ = BertSelfAttention

        if self.ipu_config.embedding_serialization_factor > 1:
            decoder = nn.Linear(
                self.config.hidden_size,
                self.config.vocab_size,
                bias=True,
            )
            decoder.load_state_dict(self.cls.predictions.decoder.state_dict())
            self.cls.predictions.decoder = decoder
            self.tie_weights()
        return self

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        if self.training:
            output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            sequence_output = output[0]

            # Select only the masked tokens for the classifier
            max_number_of_masked_tokens = int(labels.size(1) * 0.25)
            masked_lm_labels, masked_lm_positions = torch.topk(labels, k=max_number_of_masked_tokens, dim=1)
            masked_output = self.gather_indices(sequence_output, masked_lm_positions)

            prediction_scores = self.cls(masked_output)
            output = (prediction_scores,) + output[2:]

            masked_lm_loss = F.cross_entropy(
                prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1)
            ).float()
            return (masked_lm_loss,)

        else:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
                return_dict=False,
            )


class BertPipelineMixin(PipelineMixin):
    def parallelize(self):
        """
        Transform the model to run in an IPU pipeline.
        - Adds pipeline stages to the model
        - Replaces self-attention layers with fused-qkv self-attention layers
        - (If enabled) Replaces the word embedding with a SerializedEmbedding
        - Adds recomputation checkpoints
        """
        super().parallelize()

        # Use faster fused-qkv self-attention
        for layer in self.bert.encoder.layer:
            layer.attention.self.__class__ = BertFusedSelfAttention

        layer_ipu = get_layer_ipu(self.ipu_config.layers_per_ipu)

        logger.info("-------------------- Device Allocation --------------------")
        logger.info("Embedding --> IPU 0")
        if self.ipu_config.embedding_serialization_factor > 1:
            self.bert.embeddings.word_embeddings = SerializedEmbedding(
                self.bert.embeddings.word_embeddings, self.ipu_config.embedding_serialization_factor
            )
        self.bert.embeddings = poptorch.BeginBlock(self.bert.embeddings, "Embedding", ipu_id=0)
        hs = outline_attribute(self.bert.embeddings.LayerNorm, "embedding")
        self._hooks.extend(hs)

        for index, layer in enumerate(self.bert.encoder.layer):
            ipu = layer_ipu[index]
            if self.ipu_config.recompute_checkpoint_every_layer and index != self.config.num_hidden_layers - 1:
                h = recomputation_checkpoint(layer)
                self._hooks.append(h)
            self.bert.encoder.layer[index] = poptorch.BeginBlock(layer, f"Encoder{index}", ipu_id=ipu)
            logger.info(f"Encoder {index:<2} --> IPU {ipu}")
        return self

    def deparallelize(self):
        """
        Undo the changes to the model done by `parallelize`.
        You should call this before doing `save_pretrained` so that the `model.state_dict` is
        compatible with the original model.
        """
        super().deparallelize()

        for layer in self.bert.encoder.layer:
            layer.attention.self.__class__ = BertSelfAttention

        # Deserialize the serialized word embedding
        if self.ipu_config.embedding_serialization_factor > 1:
            self.bert.embeddings.word_embeddings = self.bert.embeddings.word_embeddings.deserialize()
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

    def parallelize(self):
        super().parallelize()
        last_ipu = self.ipu_config.ipus_per_replica - 1
        logger.info(f"Classifier Output --> IPU {last_ipu}")
        self.classifier = poptorch.BeginBlock(self.classifier, "Classifier Output", ipu_id=last_ipu)
        logger.info("-----------------------------------------------------------")
        return self

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            return_dict=False,
        )


@register(BertForMultipleChoice)
class PipelinedBertForMultipleChoice(BertForMultipleChoice, BertPipelineMixin):
    """
    BertForMultipleChoice transformed to run in an IPU pipeline.

    Recommended usage:
    ```
    model = PipelinedBertForMultipleChoice(config).parallelize().half()
    ```
    """

    def parallelize(self):
        super().parallelize()
        last_ipu = self.ipu_config.ipus_per_replica - 1
        logger.info(f"Classifier Output --> IPU {last_ipu}")
        self.classifier = poptorch.BeginBlock(self.classifier, "Classifier Output", ipu_id=last_ipu)
        logger.info("-----------------------------------------------------------")
        return self

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            return_dict=False,
        )


@register(BertForTokenClassification)
class PipelinedBertForTokenClassification(BertForTokenClassification, BertPipelineMixin):
    """
    BertForTokenClassification transformed to run in an IPU pipeline.

    Recommended usage:
    ```
    model = PipelinedBertForTokenClassification(config).parallelize().half()
    ```
    """

    def parallelize(self):
        super().parallelize()
        last_ipu = self.ipu_config.ipus_per_replica - 1
        logger.info(f"Classifier Output --> IPU {last_ipu}")
        self.classifier = poptorch.BeginBlock(self.classifier, "Classifier Output", ipu_id=last_ipu)
        logger.info("-----------------------------------------------------------")
        return self

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            return_dict=False,
        )


@register(BertForQuestionAnswering)
class PipelinedBertForQuestionAnswering(BertForQuestionAnswering, BertPipelineMixin):
    """
    BertForQuestionAnswering transformed to run in an IPU pipeline.

    Recommended usage:
    ```
    model = PipelinedBertForQuestionAnswering(config).parallelize().half()
    ```
    """

    def parallelize(self):
        super().parallelize()
        last_ipu = self.ipu_config.ipus_per_replica - 1
        logger.info(f"QA Outputs --> IPU {last_ipu}")
        self.qa_outputs = poptorch.BeginBlock(self.qa_outputs, "QA Outputs", ipu_id=last_ipu)
        logger.info("-----------------------------------------------------------")
        return self

    def forward(self, input_ids, attention_mask, token_type_ids, start_positions=None, end_positions=None):
        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            start_positions=start_positions,
            end_positions=end_positions,
            return_dict=False,
        )
        if start_positions is not None and end_positions is not None:
            output = (poptorch.identity_loss(output[0], reduction="none"),) + output[1:]
        return output
