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

import transformers

from ....utils import logging
from ....fx.optimization import ChangeTrueDivToMulByInverse, MergeLinears, compose
from ...modeling_utils import PipelineMixin, get_layer_ipu, register
from ...fx.utils import symbolic_trace_pipelined_model
from ...fx.transformations import (
    AddPoptorchBlock,
    AddPoptorchBlocksInSeries,
    ClipValues,
    ClipValuesSymmetric,
    RecomputationCheckpoint,
    TupleOutput,
)


logger = logging.get_logger(__name__)

_OPTIMIZATION_TRANSFORMATIONS = [
    ChangeTrueDivToMulByInverse(),
    MergeLinears(),
    #    FuseBiasInLinear(),
]

_NON_REVERSIBLE_TRANSFORMATIONS = [
    ClipValuesSymmetric(1e4, exclude_targets=["view"]),
    ClipValues(1e-4, float("inf"), include_targets=[torch.nn.LayerNorm]),
    TupleOutput(),
]


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
                    "vit.encoder.layer.[0-9]+", to_exclude=f"vit.encoder.layer.{self.config.num_layers - 1}"
                ),
            ]
        return transformations

    def parallelize(self):
        super().parallelize()
        traced = symbolic_trace_pipelined_model(self)
        transformations = self.get_transformations()
        transformations += _OPTIMIZATION_TRANSFORMATIONS
        composition = compose(*transformations)
        non_reversible_composition = compose(*_NON_REVERSIBLE_TRANSFORMATIONS)
        traced = composition(traced)
        traced = non_reversible_composition(traced)
        return traced

    def deparallelize(self):
        super().deparallelize()
        transformations = self.get_transformations()
        transformations += _OPTIMIZATION_TRANSFORMATIONS
        composition = compose(*transformations)
        self = composition(self, reverse=True)
        return self
