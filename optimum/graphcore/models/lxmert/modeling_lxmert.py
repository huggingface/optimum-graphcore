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

import poptorch
import transformers
from optimum.utils import logging

from ...modeling_utils import PipelineMixin, register


logger = logging.get_logger(__name__)


def recomputation_checkpoint(module: nn.Module) -> torch.utils.hooks.RemovableHandle:
    """Annotates the output of a module to be checkpointed instead of
    recomputed"""

    def recompute_outputs(module, inputs, outputs):
        if type(outputs) is tuple:
            return tuple(poptorch.recomputationCheckpoint(y) for y in outputs)
        else:
            return poptorch.recomputationCheckpoint(outputs)

    return module.register_forward_hook(recompute_outputs)


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
        logger.info("Embedding  --> IPU 0")
        self.lxmert.embeddings = poptorch.BeginBlock(self.lxmert.embeddings, "Embedding", ipu_id=0)

        # Language layers
        for index, layer in enumerate(self.lxmert.encoder.layer):
            if self.config.recompute_checkpoint_every_layer:
                h = recomputation_checkpoint(layer)
                self._hooks.append(h)
            self.lxmert.encoder.layer[index] = poptorch.BeginBlock(layer, f"Language layer{index}", ipu_id=1)
            logger.info(f"Language layer {index:<2}   --> IPU {1}")

        # Visual layers
        self.lxmert.encoder.visn_fc = poptorch.BeginBlock(self.lxmert.encoder.visn_fc, "Image embedding", ipu_id=2)
        
        for index, layer in enumerate(self.lxmert.encoder.r_layers):
            if self.config.recompute_checkpoint_every_layer:
                h = recomputation_checkpoint(layer)
                self._hooks.append(h)
            self.lxmert.encoder.r_layers[index] = poptorch.BeginBlock(layer, f"Visual layer{index}", ipu_id=2)
            logger.info(f"Visual layer {index:<2}   --> IPU {2}")

        # Cross modality layers
        for index, layer in enumerate(self.lxmert.encoder.x_layers):
            if self.config.recompute_checkpoint_every_layer:
                h = recomputation_checkpoint(layer)
                self._hooks.append(h)
            self.lxmert.encoder.x_layers[index] = poptorch.BeginBlock(layer, f"Cross modality layer{index}", ipu_id=3)
            logger.info(f"Cross modality layer {index:<2}   --> IPU {3}")

        logger.info("Pooler     --> IPU {3}")
        self.lxmert.pooler = poptorch.BeginBlock(self.lxmert.pooler, "Pooler", ipu_id=3)

        print(f"Head       --> IPU {3}")
        self.answer_head = poptorch.BeginBlock(self.answer_head, "Head", ipu_id=3)
        logger.info("-----------------------------------------------------------")
        return self

    def forward(self, input_ids, visual_feats, visual_pos, attention_mask, visual_attention_mask, token_type_ids, labels=None):
        return super().forward(
            input_ids=input_ids,
            visual_feats=visual_feats,
            visual_pos=visual_pos,
            attention_mask=attention_mask,
            visual_attention_mask=visual_attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            return_dict=False,
        )