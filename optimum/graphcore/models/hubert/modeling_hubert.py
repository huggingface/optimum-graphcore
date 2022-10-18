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

import poptorch
from transformers import HubertForSequenceClassification

from ....fx.optimization import ChangeTrueDivToMulByInverse, MergeLinears, compose
from ....utils import logging
from ...fx.transformations import (
    AddPoptorchBlock,
    AddPoptorchBlocksInSeries,
    ClipValues,
    ClipValuesSymmetric,
    RecomputationCheckpoint,
    TupleOutput,
)
from ...fx.utils import symbolic_trace_pipelined_model
from ...modeling_utils import PipelineMixin, get_layer_ipu, register


logger = logging.get_logger(__name__)

_OPTIMIZATION_TRANSFORMATIONS = [
    ChangeTrueDivToMulByInverse(),
    MergeLinears(),
    #    FuseBiasInLinear(),
]

_NON_REVERSIBLE_TRANSFORMATIONS = [
    ClipValuesSymmetric(1e4, exclude_targets=["view"]),
    ClipValues(1e-4, float("inf"), include_targets=[torch.nn.LayerNorm]),
]


@register(HubertForSequenceClassification)
class PipelinedHubertForSequenceClassification(HubertForSequenceClassification, PipelineMixin):
    def get_transformations(self):
        log_insertions = self.ipu_config.log_insertions
        layer_ipu = get_layer_ipu(self.ipu_config.layers_per_ipu)
        transformations = [
            AddPoptorchBlock("Feature Extractor", 0, "hubert.feature_extractor", log_insertions=log_insertions),
            AddPoptorchBlock("Feature Projection", 0, "hubert.feature_projection", log_insertions=log_insertions),
            AddPoptorchBlock("Encoder", 0, "hubert.encoder", log_insertions=log_insertions),
            AddPoptorchBlocksInSeries(
                "Encoder", layer_ipu, r"hubert.encoder.layers.[0-9]+", log_insertions=log_insertions
            ),
            AddPoptorchBlock("Projector", layer_ipu[-1], "projector", log_insertions=log_insertions),
            AddPoptorchBlock("Classifier", layer_ipu[-1], "classifier", log_insertions=log_insertions),
        ]
        if self.ipu_config.recompute_checkpoint_every_layer:
            transformations += [
                RecomputationCheckpoint(
                    "hubert.encoder.layers.[0-9]+",
                    to_exclude=f"hubert.encoder.layers.{self.config.num_hidden_layers - 1}",
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
