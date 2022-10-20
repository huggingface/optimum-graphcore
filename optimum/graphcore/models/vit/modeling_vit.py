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
import transformers

from ....fx.optimization import compose
from ....utils import logging
from ...fx import (
    AddPoptorchBlock,
    AddPoptorchBlocksInSeries,
    RecomputationCheckpoint,
    symbolic_trace_pipelined_model,
    DEFAULT_TRANSFORMATION_MANAGER,
)
from ...modeling_utils import PipelineMixin, get_layer_ipu, register


logger = logging.get_logger(__name__)


@register(transformers.ViTForImageClassification)
class PipelinedViTForImageClassification(transformers.ViTForImageClassification, PipelineMixin):
    def get_transformations(self):
        log_insertions = self.ipu_config.log_insertions
        layer_ipu = get_layer_ipu(self.ipu_config.layers_per_ipu)
        transformations = [
            AddPoptorchBlock("Embedding", 0, "vit.embeddings", log_insertions=log_insertions),
            AddPoptorchBlocksInSeries(
                "Encoder", layer_ipu, r"vit.encoder.layer.[0-9]+", log_insertions=log_insertions
            ),
            AddPoptorchBlock("LayerNorm Head Output", layer_ipu[-1], "vit.layernorm", log_insertions=log_insertions),
        ]
        if self.ipu_config.recompute_checkpoint_every_layer:
            transformations += [
                RecomputationCheckpoint(
                    "vit.encoder.layer.[0-9]+", to_exclude=f"vit.encoder.layer.{self.config.num_hidden_layers - 1}"
                ),
            ]
        return transformations

    def parallelize(self):
        super().parallelize()
        traced = symbolic_trace_pipelined_model(self)
        transformations = self.get_transformations()
        transformations += DEFAULT_TRANSFORMATION_MANAGER.get_reversible_transformations(self.ipu_config.optimization_level)
        composition = compose(*transformations)
        non_reversible_composition = DEFAULT_TRANSFORMATION_MANAGER.compose_non_reversible_transformations(self.ipu_config.optimization_level)
        traced = composition(traced)
        traced = non_reversible_composition(traced)
        return traced

    def deparallelize(self):
        super().deparallelize()
        transformations = self.get_transformations()
        transformations += DEFAULT_TRANSFORMATION_MANAGER.get_reversible_transformations(self.ipu_config.optimization_level)
        composition = compose(*transformations)
        self = composition(self, reverse=True)
        return self
