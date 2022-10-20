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
from transformers import HubertForSequenceClassification

from ....fx.optimization import MergeLinears, compose
from ....utils import logging
from ...fx import (
    DEFAULT_TRANSFORMATION_MANAGER,
    AddPoptorchBlock,
    AddPoptorchBlocksInSeries,
    RecomputationCheckpoint,
    symbolic_trace_pipelined_model,
)
from ...modeling_utils import PipelineMixin, get_layer_ipu, register


logger = logging.get_logger(__name__)

TRANSFORMATION_MANAGER = DEFAULT_TRANSFORMATION_MANAGER.without(MergeLinears())


@register(HubertForSequenceClassification)
class PipelinedHubertForSequenceClassification(HubertForSequenceClassification, PipelineMixin):
    def change_hubert_encoder_class(self, restore: bool):
        """Changes the encoder class to update its forward pass so that it uses our custom version.
        Args:
            restore: whether to restore the encoder to its original version or not.
        """
        from transformers.models.hubert.modeling_hubert import HubertEncoder, HubertEncoderStableLayerNorm

        from .ipu_layer_drop import IPUHubertEncoder, IPUHubertEncoderStableLayerNorm

        if self.config.do_stable_layer_norm:
            new_cls = HubertEncoderStableLayerNorm if restore else IPUHubertEncoderStableLayerNorm
        else:
            new_cls = HubertEncoder if restore else IPUHubertEncoder
        self.hubert.encoder.__class__ = new_cls

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
        self.change_hubert_encoder_class(False)
        traced = symbolic_trace_pipelined_model(self)
        transformations = self.get_transformations()
        transformations += TRANSFORMATION_MANAGER.get_reversible_transformations(self.ipu_config.optimization_level)
        composition = compose(*transformations)
        non_reversible_composition = TRANSFORMATION_MANAGER.compose_non_reversible_transformations(
            self.ipu_config.optimization_level
        )
        traced = composition(traced)
        traced = non_reversible_composition(traced)
        return traced

    def deparallelize(self):
        super().deparallelize()
        transformations = self.get_transformations()
        transformations += TRANSFORMATION_MANAGER.get_reversible_transformations(self.ipu_config.optimization_level)
        composition = compose(*transformations)
        self = composition(self, reverse=True)
        return self
