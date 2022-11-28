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

from ...modeling_utils import PipelineMixin, get_layer_ipu, recomputation_checkpoint, register


logger = logging.get_logger(__name__)


@register(transformers.ViTForImageClassification)
class PipelinedViTForImageClassification(transformers.ViTForImageClassification, PipelineMixin):
    def parallelize(self):
        super().parallelize()
        logger.info("---------- Device Allocation -----------")
        logger.info("Embedding  --> IPU 0")
        self.vit.embeddings = poptorch.BeginBlock(self.vit.embeddings, "Embedding", ipu_id=0)

        layer_ipu = get_layer_ipu(self.ipu_config.layers_per_ipu, self.vit.encoder.layer)
        for index, layer in enumerate(self.vit.encoder.layer):
            if self.ipu_config.recompute_checkpoint_every_layer:
                # Put checkpoints on every encoder layer
                h = recomputation_checkpoint(layer)
                self._hooks.append(h)
            ipu = layer_ipu[index]
            logger.info(f"Encoder {index:<2} --> IPU {ipu}")
            self.vit.encoder.layer[index] = poptorch.BeginBlock(layer, f"Encoder{index}", ipu_id=ipu)

        last_ipu = self.ipu_config.ipus_per_replica - 1
        logger.info(f"Head       --> IPU {last_ipu}")
        logger.info("---------------------------------------")
        self.vit.layernorm = poptorch.BeginBlock(self.vit.layernorm, "LayerNorm", ipu_id=last_ipu)
        self.classifier = poptorch.BeginBlock(self.classifier, "Classifier", ipu_id=last_ipu)
        return self
