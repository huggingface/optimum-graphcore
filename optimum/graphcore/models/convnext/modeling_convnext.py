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

import poptorch
import transformers
from optimum.utils import logging
from transformers.models.convnext.modeling_convnext import ConvNextLayer, ConvNextLayerNorm

from ...modeling_utils import PipelineMixin, get_layer_ipu, recomputation_checkpoint, register
from .optimized_convnextlayer import OptimizedConvNextLayer


logger = logging.get_logger(__name__)


@register(transformers.ConvNextForImageClassification)
class PipelinedConvNextForImageClassification(transformers.ConvNextForImageClassification, PipelineMixin):
    def parallelize(self):
        super().parallelize()

        # Use optimized ConvNextLayer
        for stage in self.convnext.encoder.stages:
            for layer in stage.layers:
                layer.__class__ = OptimizedConvNextLayer

        # Enable autocast for ConvNextLayerNorm because computation cannot happen in fp16
        for mod in self.modules():
            if isinstance(mod, ConvNextLayerNorm):
                mod.forward = poptorch.autocast(enabled=True)(mod.forward)

        logger.info("---------- Device Allocation -----------")
        logger.info(f"Embedding  --> IPU 0")
        self.convnext.embeddings = poptorch.BeginBlock(self.convnext.embeddings, "Embedding", ipu_id=0)

        layer_ipu = get_layer_ipu(self.ipu_config.layers_per_ipu)
        global_layer_idx = 0
        for stage_idx, stage in enumerate(self.convnext.encoder.stages):
            for layer_idx, layer in enumerate(stage.layers):
                ipu = layer_ipu[global_layer_idx]
                logger.info(f"Encoder stage {stage_idx}, convnext layer {layer_idx} --> IPU {ipu}")
                layer = poptorch.BeginBlock(layer, f"Encoder_stage_{stage_idx}_layer_{layer_idx}", ipu_id=ipu)
                global_layer_idx += 1

        last_ipu = self.ipu_config.ipus_per_replica - 1
        logger.info(f"Head --> IPU {last_ipu}")
        logger.info("---------------------------------------")
        self.convnext.layernorm = poptorch.BeginBlock(self.convnext.layernorm, "LayerNorm", ipu_id=last_ipu)
        self.classifier = poptorch.BeginBlock(self.classifier, "Classifier", ipu_id=last_ipu)

        return self

    def deparallelize(self):
        super().deparallelize()

        for mod in self.modules():
            if isinstance(mod, ConvNextLayerNorm):
                mod.forward = ConvNextLayerNorm.forward.__get__(mod, ConvNextLayerNorm)

        # Switch back to non-optimized ConvNextLayer
        for stage in self.convnext.encoder.stages:
            for layer in stage.layers:
                layer.__class__ = ConvNextLayer

    def forward(self, pixel_values=None, labels=None):
        return super().forward(pixel_values=pixel_values, labels=labels, return_dict=False)
