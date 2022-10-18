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
from torch import nn

from transformers.models.convnext.modeling_convnext import (
    ConvNextForImageClassification,
    ConvNextLayer,
    ConvNextLayerNorm,
)

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
from .optimized_convnextlayer import OptimizedConvNextLayer


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


class IPUConvNextLayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch_size, height,
    width, channels) while channels_first corresponds to inputs with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-4, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError(f"Unsupported data format: {self.data_format}")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            x = torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            input_dtype = x.dtype
            x = x.float()
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = x.to(dtype=input_dtype)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


@register(ConvNextForImageClassification)
class PipelinedConvNextForImageClassification(ConvNextForImageClassification, PipelineMixin):
    def get_transformations(self):
        log_insertions = self.ipu_config.log_insertions
        layer_ipu = get_layer_ipu(self.ipu_config.layers_per_ipu)
        transformations = [
            AddPoptorchBlock("Embedding", 0, r"convnext.embeddings", log_insertions=log_insertions),
            AddPoptorchBlocksInSeries(
                "Encoder", layer_ipu, r"convnext.encoder.stages.[0-9]+.layers.[0-9]+", log_insertions=log_insertions
            ),
            AddPoptorchBlock("LayerNorm", layer_ipu[-1], r"convnext.layernorm", log_insertions=log_insertions),
            AddPoptorchBlock("Classifier", layer_ipu[-1], r"classifier", log_insertions=log_insertions),
        ]
        if self.ipu_config.recompute_checkpoint_every_layer:
            transformations += [
                RecomputationCheckpoint(
                    "convnext.encoder.stages.[0-9]+.layers.[0-9]+",
                    to_exclude=f"convnext.encoder.stages.{self.config.num_stages - 1}.layers.{self.config.depths[-1] - 1}",
                ),
            ]
        return transformations

    def parallelize(self):
        super().parallelize()

        if not isinstance(self, torch.fx.GraphModule):
            # Use optimized ConvNextLayer
            for stage in self.convnext.encoder.stages:
                for layer in stage.layers:
                    layer.__class__ = OptimizedConvNextLayer

            # # Enable autocast for ConvNextLayerNorm because computation cannot happen in fp16
            # for mod in self.modules():
            #     if isinstance(mod, ConvNextLayerNorm):
            #         mod.__class__ = IPUConvNextLayerNorm

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

        # TODO: is that needed?
        for mod in self.modules():
            if isinstance(mod, IPUConvNextLayerNorm):
                mod.__class__ = ConvNextLayerNorm

        transformations = self.get_transformations()
        transformations += _OPTIMIZATION_TRANSFORMATIONS
        composition = compose(*transformations)
        self = composition(self, reverse=True)
        return self
