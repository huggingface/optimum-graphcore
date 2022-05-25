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
from transformers import (
    RobertaForMaskedLM,
    RobertaForMultipleChoice,
    RobertaForQuestionAnswering,
    RobertaForSequenceClassification,
    RobertaForTokenClassification,
)

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


logger = logging.get_logger(__name__)


class RobertaPipelineMixin(PipelineMixin):
    def parallelize(self):
        """
        Transform the Roberta model body to run in an IPU pipeline.
        - Adds pipeline stages to the model
        - (If enabled) Replaces the word embedding with a SerializedEmbedding
        - Adds recomputation checkpoints
        """
        super().parallelize()
        layer_ipu = get_layer_ipu(self.ipu_config.layers_per_ipu)

        logger.info("-------------------- Device Allocation --------------------")
        logger.info("Embedding  --> IPU 0")
        if self.ipu_config.embedding_serialization_factor > 1:
            self.roberta.embeddings.word_embeddings = SerializedEmbedding(
                self.roberta.embeddings.word_embeddings, self.ipu_config.embedding_serialization_factor
            )
        self.roberta.embeddings = poptorch.BeginBlock(self.roberta.embeddings, "Embedding", ipu_id=0)
        hs = outline_attribute(self.roberta.embeddings.LayerNorm, "embedding")
        self._hooks.extend(hs)

        for index, layer in enumerate(self.roberta.encoder.layer):
            ipu = layer_ipu[index]
            if self.ipu_config.recompute_checkpoint_every_layer and index != self.config.num_hidden_layers - 1:
                h = recomputation_checkpoint(layer)
                self._hooks.append(h)
            self.roberta.encoder.layer[index] = poptorch.BeginBlock(layer, f"Encoder{index}", ipu_id=ipu)
            logger.info(f"Encoder {index:<2} --> IPU {ipu}")
        return self

    def deparallelize(self):
        """
        Undo the changes to the model done by `parallelize`.
        You should call this before doing `save_pretrained` so that the `model.state_dict` is
        fully compatible with `transformers.RobertaForSequenceClassification`.
        """
        super().deparallelize()
        # Deserialize the serialized word embedding
        if self.ipu_config.embedding_serialization_factor > 1:
            self.roberta.embeddings.word_embeddings = self.roberta.embeddings.word_embeddings.deserialize()
        return self


@register(RobertaForMaskedLM)
class PipelinedRobertaForMaskedLM(RobertaForMaskedLM, PipelineMixin):
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

    def parallelize(self):
        """
        Transform the model to run in an IPU pipeline.
        - Adds pipeline stages to the model
        - (If enabled) Replaces the word embedding projection with a SerializedLinear layer
        - Adds recomputation checkpoints
        """
        super().parallelize()

        if self.ipu_config.embedding_serialization_factor > 1:
            serialized_decoder = SerializedLinear(
                self.config.hidden_size,
                self.config.vocab_size,
                self.ipu_config.embedding_serialization_factor,
                bias=True,
                mode=poptorch.MatMulSerializationMode.OutputChannels,
            )
            serialized_decoder.load_state_dict(self.lm_head.decoder.state_dict())
            self.lm_head.decoder = serialized_decoder
            self.tie_weights()

        layer_ipu = get_layer_ipu(self.ipu_config.layers_per_ipu)

        logger.info("-------------------- Device Allocation --------------------")
        logger.info("Embedding  --> IPU 0")
        self.roberta.embeddings = poptorch.BeginBlock(self.roberta.embeddings, "Embedding", ipu_id=0)
        hs = outline_attribute(self.roberta.embeddings.LayerNorm, "embedding")
        self._hooks.extend(hs)

        for index, layer in enumerate(self.roberta.encoder.layer):
            ipu = layer_ipu[index]
            if self.ipu_config.recompute_checkpoint_every_layer:
                h = recomputation_checkpoint(layer)
                self._hooks.append(h)
            self.roberta.encoder.layer[index] = poptorch.BeginBlock(layer, f"Encoder{index}", ipu_id=ipu)
            logger.info(f"Encoder {index:<2} --> IPU {ipu}")

        logger.info("LM Head    --> IPU 0")
        self.lm_head = poptorch.BeginBlock(self.lm_head, "LM Head", ipu_id=0)
        logger.info("-----------------------------------------------------------")
        return self

    def deparallelize(self):
        """
        Undo the changes to the model done by `parallelize`.
        You should call this before doing `save_pretrained` so that the `model.state_dict` is
        compatible with the original model.
        """
        super().deparallelize()

        if self.ipu_config.embedding_serialization_factor > 1:
            decoder = nn.Linear(
                self.config.hidden_size,
                self.config.vocab_size,
                bias=True,
            )
            decoder.load_state_dict(self.lm_head.decoder.state_dict())
            self.lm_head.decoder = decoder
            self.tie_weights()
        return self

    def forward(self, input_ids, attention_mask, labels=None):
        if self.training:
            outputs = self.roberta(input_ids, attention_mask=attention_mask)
            sequence_output = outputs[0]

            # Select only the masked tokens for the classifier
            max_number_of_masked_tokens = int(labels.size(1) * 0.25)
            masked_lm_labels, masked_lm_positions = torch.topk(labels, k=max_number_of_masked_tokens, dim=1)
            masked_output = self.gather_indices(sequence_output, masked_lm_positions)

            prediction_scores = self.lm_head(masked_output)

            masked_lm_loss = F.cross_entropy(
                prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1)
            ).float()
            return (masked_lm_loss,)
        else:
            return super().forward(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=False
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

    def parallelize(self):
        super().parallelize()
        last_ipu = self.ipu_config.ipus_per_replica - 1
        logger.info(f"Classifier Output --> IPU {last_ipu}")
        self.classifier = poptorch.BeginBlock(self.classifier, "Classifier Output", ipu_id=last_ipu)
        logger.info("-----------------------------------------------------------")
        return self

    def forward(self, input_ids, attention_mask, labels=None):
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=False,
        )


@register(RobertaForMultipleChoice)
class PipelinedRobertaForMultipleChoice(RobertaForMultipleChoice, RobertaPipelineMixin):
    """
    RobertaForMultipleChoice transformed to run in an IPU pipeline.

    Recommended usage:
    ```
    model = PipelinedRobertaForMultipleChoice(config).parallelize().half()
    ```
    """

    def parallelize(self):
        super().parallelize()
        last_ipu = self.ipu_config.ipus_per_replica - 1
        logger.info(f"Classifier Output --> IPU {last_ipu}")
        self.classifier = poptorch.BeginBlock(self.classifier, "Classifier Output", ipu_id=last_ipu)
        logger.info("-----------------------------------------------------------")
        return self

    def forward(self, input_ids, attention_mask, labels=None):
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=False,
        )


@register(RobertaForTokenClassification)
class PipelinedRobertaForTokenClassification(RobertaForTokenClassification, RobertaPipelineMixin):
    """
    RobertaForTokenClassification transformed to run in an IPU pipeline.

    Recommended usage:
    ```
    model = PipelinedRobertaForTokenClassification(config).parallelize().half()
    ```
    """

    def parallelize(self):
        super().parallelize()
        last_ipu = self.ipu_config.ipus_per_replica - 1
        logger.info(f"Classifier Output --> IPU {last_ipu}")
        self.classifier = poptorch.BeginBlock(self.classifier, "Classifier Output", ipu_id=last_ipu)
        logger.info("-----------------------------------------------------------")
        return self

    def forward(self, input_ids, attention_mask, labels=None):
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=False,
        )


@register(RobertaForQuestionAnswering)
class PipelinedRobertaForQuestionAnswering(RobertaForQuestionAnswering, RobertaPipelineMixin):
    """
    RobertaForQuestionAnswering transformed to run in an IPU pipeline.

    Recommended usage:
    ```
    model = PipelinedRobertaForQuestionAnswering(config).parallelize().half()
    ```
    """

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
