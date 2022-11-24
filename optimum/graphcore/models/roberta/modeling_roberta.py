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
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import poptorch
from optimum.utils import logging
from transformers import (
    RobertaForMaskedLM,
    RobertaForMultipleChoice,
    RobertaForQuestionAnswering,
    RobertaForSequenceClassification,
    RobertaForTokenClassification,
)
from transformers.modeling_outputs import MaskedLMOutput, QuestionAnsweringModelOutput

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

        logger.info("-------------------- Device Allocation --------------------")
        logger.info("Embedding  --> IPU 0")
        if self.ipu_config.embedding_serialization_factor > 1:
            self.roberta.embeddings.word_embeddings = SerializedEmbedding(
                self.roberta.embeddings.word_embeddings, self.ipu_config.embedding_serialization_factor
            )
        self.roberta.embeddings = poptorch.BeginBlock(self.roberta.embeddings, "Embedding", ipu_id=0)
        hs = outline_attribute(self.roberta.embeddings.LayerNorm, "embedding")
        self._hooks.extend(hs)

        layer_ipu = get_layer_ipu(self.ipu_config.layers_per_ipu, self.roberta.encoder.layer)
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

        logger.info("-------------------- Device Allocation --------------------")
        logger.info("Embedding  --> IPU 0")
        self.roberta.embeddings = poptorch.BeginBlock(self.roberta.embeddings, "Embedding", ipu_id=0)
        hs = outline_attribute(self.roberta.embeddings.LayerNorm, "embedding")
        self._hooks.extend(hs)

        layer_ipu = get_layer_ipu(self.ipu_config.layers_per_ipu, self.roberta.encoder.layer)
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

    def parallelize(self):
        super().parallelize()
        last_ipu = self.ipu_config.ipus_per_replica - 1
        logger.info(f"Classifier Output --> IPU {last_ipu}")
        self.classifier = poptorch.BeginBlock(self.classifier, "Classifier Output", ipu_id=last_ipu)
        logger.info("-----------------------------------------------------------")
        return self


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
