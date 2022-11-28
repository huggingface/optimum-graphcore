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
from optimum.utils import logging
from transformers.models.convnext.modeling_convnext import (
    ConvNextForImageClassification,
    ConvNextLayer,
    ConvNextLayerNorm,
)

from ...modeling_utils import PipelineMixin, get_layer_ipu, register
from .optimized_convnextlayer import OptimizedConvNextLayer


logger = logging.get_logger(__name__)


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
    def parallelize(self):
        super().parallelize()

        # Use optimized ConvNextLayer
        for stage in self.convnext.encoder.stages:
            for layer in stage.layers:
                layer.__class__ = OptimizedConvNextLayer

        # ConvNextLayerNorm does not correctly handle fp16.
        # TODO remove this in newer version of transformers that has https://github.com/huggingface/transformers/pull/18746
        for mod in self.modules():
            if isinstance(mod, ConvNextLayerNorm):
                mod.__class__ = IPUConvNextLayerNorm

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
            if isinstance(mod, IPUConvNextLayerNorm):
                mod.__class__ = ConvNextLayerNorm

        # Switch back to non-optimized ConvNextLayer
        for stage in self.convnext.encoder.stages:
            for layer in stage.layers:
                layer.__class__ = ConvNextLayer
