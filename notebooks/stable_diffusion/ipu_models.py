import copy
from functools import partial
import inspect

from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from optimum.graphcore import IPUConfig
from optimum.graphcore.modeling_utils import PipelineMixin
import torch

import poptorch


def _sliced_attention(self, query, key, value, sequence_length, dim):
    batch_size_attention = query.shape[0]
    hidden_states = []
    slice_size = self._slice_size if self._slice_size is not None else batch_size_attention
    for i in range(batch_size_attention // slice_size):
        start_idx = i * slice_size
        end_idx = (i + 1) * slice_size
        if query.device.type == "mps":
            # Better performance on mps (~20-25%)
            attn_slice = (
                torch.einsum("b i d, b j d -> b i j", query[start_idx:end_idx], key[start_idx:end_idx]) * self.scale
            )
        else:
            attn_slice = (
                torch.matmul(query[start_idx:end_idx], key[start_idx:end_idx].transpose(1, 2)) * self.scale
            )  # TODO: use baddbmm for better performance
        attn_slice = attn_slice.softmax(dim=-1)
        if query.device.type == "mps":
            attn_slice = torch.einsum("b i j, b j d -> b i d", attn_slice, value[start_idx:end_idx])
        else:
            attn_slice = torch.matmul(attn_slice, value[start_idx:end_idx])

        hidden_states.append(attn_slice)

    hidden_states = torch.cat(hidden_states)

    # reshape hidden_states
    hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
    return hidden_states


def override_sliced_attention(unet):
    for module in unet.modules():
        if module.__class__.__name__ == "CrossAttention":
            assert hasattr(module, "_sliced_attention")
            setattr(module, "_sliced_attention", partial(_sliced_attention, module))


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


def copy_signature(base):
    def decorator(func):
        func.__signature__ = inspect.signature(base)
        return func

    return decorator


class IPUStableDiffusionPipeline(StableDiffusionPipeline):
    @copy_signature(StableDiffusionPipeline.__init__)
    def __init__(self, **kwargs):
        for module_name in ["text_encoder", "vae", "safety_checker"]:
            module = kwargs.get(module_name, None)
            if module is not None:
                module = module.float()
                kwargs[module_name] = module

        unet = kwargs.get("unet", None)
        if unet is not None:
            IPU_CONFIG_DICT = {
                "inference_device_iterations": 1,
                "inference_replication_factor": {"default": 1},
                "executable_cache_dir": "./exe_cache",
                "ipus_per_replica": 4,
                "matmul_proportion": [0.09, 0.1, 0.1, 0.08],
                "enable_half_partials": True,
            }
            ipu_config = IPUConfig.from_dict(IPU_CONFIG_DICT)
            unet_ipu = copy.deepcopy(unet)
            unet_ipu.__class__ = IPUUNet2DConditionModel
            override_sliced_attention(unet_ipu)
            unet_ipu.ipu_config = ipu_config
            unet_ipu.parallelize()
            opts = ipu_config.to_options(for_inference=True)
            opts.setExecutionStrategy(poptorch.ShardedExecution())
            opts._Popart.set("saveInitializersToFile", "weights.onnx")
            opts._Popart.set("enableExplicitIR", True)
            unet_ipu = poptorch.inferenceModel(unet_ipu.eval(), opts)

            unet = UNetCastingWrapper(unet_ipu)

            kwargs["unet"] = unet

        super().__init__(**kwargs)
