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

from typing import Optional, Tuple, Union

import torch

import poptorch
from optimum.utils import logging
from transformers import MT5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput

from ...generation_utils import _IndexedInputLinear
from ...modeling_utils import (
    SerializedEmbedding,
    SplitLinear,
    register,
)
from ..t5.modeling_t5 import PipelinedT5ForConditionalGeneration


logger = logging.get_logger(__name__)


@register(MT5ForConditionalGeneration)
class PipelinedMT5ForConditionalGeneration(MT5ForConditionalGeneration, PipelinedT5ForConditionalGeneration):
    def parallelize(self, for_generation=False):
        """
        Transform the model to run in an IPU pipeline.
        - Adds pipeline stages to the model
        - (If enabled) Replaces the shared embedding with a SerializedEmbedding
        - Adds recomputation checkpoints

        Recommended usage:
        ```
        model = PipelinedMT5ForConditionalGeneration(config).parallelize().half()
        ```
        """
        # in training mode, this must be the case otherwise the model will not fit
        # this requires training and inference specific settings in the IPUConfig
        if self.ipu_config.embedding_serialization_factor < 2:
            raise ValueError("Embedding serialization factor must be greater than 1 for MT5 to run on the IPU.")

        self.shared = SerializedEmbedding(self.shared, self.ipu_config.embedding_serialization_factor)

        if for_generation:
            self.lm_head = SplitLinear(self.lm_head, splits=1, serialization_factor=self.ipu_config.embedding_serialization_factor)
        else:
            self.lm_head = SplitLinear(self.lm_head, splits=4, serialization_factor=self.ipu_config.embedding_serialization_factor)
        
        if self.config.tie_word_embeddings:
            raise ValueError("Tied input and output embeddings for MT5 are currently not supported on the IPU.")

        # Prevent `PipelinedT5ForConditionalGeneration.parallelize` from serialising the
        # embedding, we already split it above
        original_embedding_serialization_factor = self.ipu_config.embedding_serialization_factor
        self.ipu_config.embedding_serialization_factor = 1
        PipelinedT5ForConditionalGeneration.parallelize(self, for_generation=for_generation)
        self.ipu_config.embedding_serialization_factor = original_embedding_serialization_factor

        poptorch.removeBlocks(self.lm_head)
        if for_generation:
            last_ipu = self.decoder_ipu_config.ipus_per_replica - 1
            logger.info(f"Ammendment: LM Head Output --> IPU {last_ipu}")
            self.lm_head.split_linear_layers[0] = poptorch.BeginBlock(self.lm_head.split_linear_layers[0], f"LM Head Output {0}", ipu_id=last_ipu)
        else:
            last_ipu = self.ipu_config.ipus_per_replica - 1
            # TODO: need to make sure that splitting the lm head does not exceed ipus per replica for training
            logger.info(f"Ammendment: LM Head Output --> IPU {last_ipu - 1}-{last_ipu}")
            self.lm_head.split_linear_layers[0] = poptorch.BeginBlock(self.lm_head.split_linear_layers[0], f"LM Head Output {0}", ipu_id=last_ipu - 1)
            for i, _ in enumerate(self.lm_head.split_linear_layers[1:]):
                self.lm_head.split_linear_layers[i+1] = poptorch.BeginBlock(self.lm_head.split_linear_layers[i+1], f"LM Head Output {i+1}", ipu_id=last_ipu)
        
        logger.info("-----------------------------------------------------------")
        return self
    
    def deparallelize(self):
        """
        Undo the changes to the model done by `parallelize`.
        You should call this before doing `save_pretrained` so that the `model.state_dict` is
        fully compatible with `transformers.T5ForConditionalGeneration`.
        """
        # Prevent `PipelinedT5ForConditionalGeneration.deparallelize` from deserialising
        # the embedding, we will do it below
        original_embedding_serialization_factor = self.ipu_config.embedding_serialization_factor
        self.ipu_config.embedding_serialization_factor = 1
        PipelinedT5ForConditionalGeneration.deparallelize(self)
        self.ipu_config.embedding_serialization_factor = original_embedding_serialization_factor

        if self.lm_head.__class__ == _IndexedInputLinear:
            self.lm_head = self.lm_head.wrapped_linear
        self.lm_head = self.lm_head.deserialize()
        self.shared = self.shared.deserialize()

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
        return PipelinedT5ForConditionalGeneration.forward(
            self,
            input_ids,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            head_mask,
            decoder_head_mask,
            cross_attn_head_mask,
            encoder_outputs,
            past_key_values,
            inputs_embeds,
            decoder_inputs_embeds,
            labels,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
