# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
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

import copy

import torch

import poptorch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from diffusers.models.attention import CrossAttention
from optimum.graphcore import IPUConfig
from optimum.graphcore.modeling_utils import PipelineMixin


def _sliced_attention(self, query, key, value, sequence_length, dim):
    """Overriding this implementation to use concatenation as slice assignment is not yet supported."""
    batch_size_attention = query.shape[0]
    hidden_states = []
    slice_size = self._slice_size if self._slice_size is not None else batch_size_attention
    for i in range(batch_size_attention // slice_size):
        start_idx = i * slice_size
        end_idx = (i + 1) * slice_size
        attn_slice = torch.matmul(query[start_idx:end_idx], key[start_idx:end_idx].transpose(1, 2)) * self.scale
        attn_slice = attn_slice.softmax(dim=-1)
        attn_slice = torch.matmul(attn_slice, value[start_idx:end_idx])

        hidden_states.append(attn_slice)

    hidden_states = torch.cat(hidden_states)

    # reshape hidden_states
    hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
    return hidden_states


def override_sliced_attention(unet):
    for module in unet.modules():
        if isinstance(module, CrossAttention):
            module._sliced_attention = _sliced_attention.__get__(module, CrossAttention)


class UNetCastingWrapper(torch.nn.Module):
    def __init__(self, unet, input_dtype=torch.float16, output_dtype=torch.float32):
        super().__init__()
        self.unet = unet
        self.input_dtype = input_dtype
        self.output_dtype = output_dtype

    def forward(self, sample, timestep, encoder_hidden_states, return_dict=True):
        sample = sample.to(self.input_dtype)
        encoder_hidden_states = encoder_hidden_states.to(self.input_dtype)

        ret = self.unet(sample, timestep, encoder_hidden_states)

        ret.sample = ret.sample.to(self.output_dtype)
        return ret

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "unet":
                raise AttributeError()
            return getattr(self.unet, name)


class IPUUNet2DConditionModel(UNet2DConditionModel, PipelineMixin):
    def parallelize(self):
        super().parallelize()

        self.conv_in = poptorch.BeginBlock(self.conv_in, "conv_in", ipu_id=0)
        self.down_blocks[2].downsamplers[0] = poptorch.BeginBlock(
            self.down_blocks[2].downsamplers[0], "down_blocks[2].downsamplers[0]", ipu_id=1
        )
        self.up_blocks[0].resnets[2] = poptorch.BeginBlock(
            self.up_blocks[0].resnets[2], "up_blocks[0].resnets[2]", ipu_id=2
        )
        self.up_blocks[1].attentions[2] = poptorch.BeginBlock(
            self.up_blocks[1].attentions[2], "up_blocks[1].attentions[2]", ipu_id=3
        )

        return self


def maybe_cast_module_to_float(module):
    if module is not None:
        module = module.float()
    return module


class IPUStableDiffusionPipeline(StableDiffusionPipeline):
    def __init__(
        self, vae, text_encoder, tokenizer, unet, scheduler, safety_checker, feature_extractor, ipu_config=None
    ):
        text_encoder = maybe_cast_module_to_float(text_encoder)
        vae = maybe_cast_module_to_float(vae)
        safety_checker = maybe_cast_module_to_float(safety_checker)

        if unet is not None:
            default_ipu_config_dict = {
                "enable_half_partials": True,
                "executable_cache_dir": "./exe_cache",
                "inference_device_iterations": 1,
                "inference_replication_factor": 1,
                "ipus_per_replica": 4,
                "matmul_proportion": [0.09, 0.1, 0.1, 0.08],
            }
            if ipu_config is not None:
                default_ipu_config_dict.update(ipu_config)
            unet_ipu_config = IPUConfig.from_dict(default_ipu_config_dict)

            unet_ipu = copy.deepcopy(unet)
            unet_ipu.__class__ = IPUUNet2DConditionModel
            unet_ipu.ipu_config = unet_ipu_config
            unet_ipu.parallelize()
            override_sliced_attention(unet_ipu)

            opts = unet_ipu_config.to_options(for_inference=True)
            opts._Popart.set("saveInitializersToFile", "weights.onnx")
            opts._Popart.set("enableExplicitIR", True)
            unet_ipu = poptorch.inferenceModel(unet_ipu.eval(), opts)

            unet = UNetCastingWrapper(unet_ipu)

        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, ipu_config=None, **kwargs):
        return super().from_pretrained(pretrained_model_name_or_path, ipu_config=ipu_config, **kwargs)
