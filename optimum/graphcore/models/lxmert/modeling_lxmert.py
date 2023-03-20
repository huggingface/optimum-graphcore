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
import torch.nn.functional as F

import poptorch
from optimum.utils import logging
from transformers import LxmertForQuestionAnswering
from transformers.models.lxmert.modeling_lxmert import LxmertForQuestionAnsweringOutput

from ...modeling_utils import PipelineMixin, recomputation_checkpoint, register


logger = logging.get_logger(__name__)


@register(LxmertForQuestionAnswering)
class PipelinedLxmertForQuestionAnswering(LxmertForQuestionAnswering, PipelineMixin):
    def parallelize(self):
        """
        Transform the model to run in an IPU pipeline.
        - Adds pipeline stages to the model
        - Adds recomputation checkpoints
        Recommended usage:
        ```
        model = PipelinedLxmertForQuestionAnswering(config).parallelize().half()
        ```
        """
        self._hooks = []

        logger.info("-------------------- Device Allocation --------------------")
        logger.info("Embedding               --> IPU 0")
        self.lxmert.embeddings = poptorch.BeginBlock(self.lxmert.embeddings, "Embedding", ipu_id=0)
        logger.info("Image embedding         --> IPU 0")
        self.lxmert.encoder.visn_fc = poptorch.BeginBlock(self.lxmert.encoder.visn_fc, "Image embedding", ipu_id=0)

        # Language layers
        for index, layer in enumerate(self.lxmert.encoder.layer):
            if self.ipu_config.recompute_checkpoint_every_layer:
                h = recomputation_checkpoint(layer)
                self._hooks.append(h)
            self.lxmert.encoder.layer[index] = poptorch.BeginBlock(layer, f"Language layer{index}", ipu_id=1)
            logger.info(f"Language layer {index:<2}       --> IPU 1")

        # Visual layers
        for index, layer in enumerate(self.lxmert.encoder.r_layers):
            if self.ipu_config.recompute_checkpoint_every_layer:
                h = recomputation_checkpoint(layer)
                self._hooks.append(h)
            self.lxmert.encoder.r_layers[index] = poptorch.BeginBlock(layer, f"Visual layer{index}", ipu_id=2)
            logger.info(f"Visual layer {index:<2}         --> IPU 2")

        # Cross modality layers
        for index, layer in enumerate(self.lxmert.encoder.x_layers):
            if self.ipu_config.recompute_checkpoint_every_layer:
                h = recomputation_checkpoint(layer)
                self._hooks.append(h)
            self.lxmert.encoder.x_layers[index] = poptorch.BeginBlock(layer, f"Cross modality layer{index}", ipu_id=3)
            logger.info(f"Cross modality layer {index:<2} --> IPU 3")

        logger.info(f"Pooler                  --> IPU 3")
        self.lxmert.pooler = poptorch.BeginBlock(self.lxmert.pooler, "Pooler", ipu_id=3)

        logger.info(f"Head                    --> IPU 3")
        self.answer_head = poptorch.BeginBlock(self.answer_head, "Head", ipu_id=3)
        logger.info("-----------------------------------------------------------")
        return self

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        visual_feats: Optional[torch.FloatTensor] = None,
        visual_pos: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        visual_attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[LxmertForQuestionAnsweringOutput, Tuple[torch.FloatTensor]]:
        r"""
        labels: (`Torch.Tensor` of shape `(batch_size)`, *optional*):
            A one-hot representation of the correct answer
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        lxmert_output = self.lxmert(
            input_ids=input_ids,
            visual_feats=visual_feats,
            visual_pos=visual_pos,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            visual_attention_mask=visual_attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        pooled_output = lxmert_output[2]
        answer_score = self.answer_head(pooled_output)
        loss = None
        if labels is not None:
            if labels.dim() == 1:
                loss = F.cross_entropy(answer_score.view(-1, self.num_qa_labels), labels.view(-1))
            # Soft labels for datasets such as VQA v2
            else:
                loss = F.binary_cross_entropy_with_logits(
                    answer_score.view(-1, self.num_qa_labels), labels.view(-1, self.num_qa_labels)
                )

        if not return_dict:
            output = (answer_score,) + lxmert_output[3:]
            return (loss,) + output if loss is not None else output

        return LxmertForQuestionAnsweringOutput(
            loss=loss,
            question_answering_score=answer_score,
            language_hidden_states=lxmert_output.language_hidden_states,
            vision_hidden_states=lxmert_output.vision_hidden_states,
            language_attentions=lxmert_output.language_attentions,
            vision_attentions=lxmert_output.vision_attentions,
            cross_encoder_attentions=lxmert_output.cross_encoder_attentions,
        )
