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
import transformers
from optimum.utils import logging

from ...modeling_utils import PipelineMixin, recomputation_checkpoint, register


logger = logging.get_logger(__name__)


@register(transformers.LxmertForQuestionAnswering)
class PipelinedLxmertForQuestionAnswering(transformers.LxmertForQuestionAnswering, PipelineMixin):
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
        input_ids,
        visual_feats,
        visual_pos,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        visual_attention_mask=None,
    ):
        lxmert_output = self.lxmert(
            input_ids=input_ids,
            visual_feats=visual_feats,
            visual_pos=visual_pos,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            visual_attention_mask=visual_attention_mask,
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

        output = (answer_score,) + lxmert_output[3:]
        return (loss,) + output if loss is not None else output
