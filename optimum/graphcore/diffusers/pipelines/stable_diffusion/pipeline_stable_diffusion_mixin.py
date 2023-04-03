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
from typing import Optional, Tuple, Union

import torch

import poptorch
from diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.models.autoencoder_kl import AutoencoderKLOutput
from diffusers.models.cross_attention import CrossAttention
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from diffusers.models.vae import DecoderOutput, DiagonalGaussianDistribution
from optimum.utils import logging
from transformers import CLIPTextModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import CLIPTextTransformer

from ....ipu_configuration import IPUConfig
from ....modeling_utils import PipelineMixin
from .safety_checker import IPUStableDiffusionSafetyChecker


logger = logging.get_logger(__name__)


class IPUSlicedAttnProcessor:
    """
    SlicedAttnProcessor but we slice across the query sequence length instead of across heads.
    NB: this ignores the `slice_size` factor since we interpret it differently and use a value that is
    derived from the sequence length based on an empirical attention matrix memory target.
    """

    def __init__(self, attn_matrix_target_mem_mb: int):
        if attn_matrix_target_mem_mb < 1:
            raise ValueError(f"`attn_matrix_target_mem_mb` {attn_matrix_target_mem_mb} must be a positive integer.")

        self._attn_matrix_target_mem_mb = attn_matrix_target_mem_mb

    @staticmethod
    def _nearest_divisor(target, start, end):
        for divisor in range(start, end + 1):
            if target % divisor == 0:
                return divisor
        raise ValueError(f"No divisor found in range [{start}, {end}].")

    def __call__(self, attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape

        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length)

        query = attn.to_q(hidden_states)
        dim = query.shape[-1]
        query = attn.head_to_batch_dim(query)

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # Begin IPU modifications.
        attn_matrix_mem = query.element_size() * query.shape[0] * query.shape[1] * key.shape[1]
        num_slices = attn_matrix_mem // (self._attn_matrix_target_mem_mb * 1024 * 1024)
        num_slices = max(num_slices, 1)
        num_slices = self._nearest_divisor(query.shape[1], num_slices, 2 * num_slices)
        slice_size = query.shape[1] // num_slices

        hidden_states = []

        key = key.transpose(1, 2)
        for i in range(num_slices):
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size

            attn_slice = torch.matmul(query[:, start_idx:end_idx], key) * attn.scale
            if attention_mask is not None:
                attn_slice = attn_slice + attention_mask[:, start_idx:end_idx]
            attn_slice = attn_slice.softmax(dim=-1)
            attn_slice = torch.matmul(attn_slice, value)

            hidden_states.append(attn_slice)

        hidden_states = torch.cat(hidden_states, dim=1)
        # End IPU modifications.

        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class IPUCLIPTextModel(CLIPTextModel, PipelineMixin):
    def parallelize(self):
        super().parallelize()

        def _build_causal_attention_mask(self, bsz, seq_len, dtype):
            # lazily create causal attention mask, with full attention between the vision tokens
            # pytorch uses additive attention mask; fill with -inf
            # IPU MOD
            # mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype)
            # mask.fill_(torch.tensor(torch.finfo(dtype).min))
            mask = torch.ones((bsz, seq_len, seq_len), dtype=dtype) * torch.finfo(dtype).min
            mask.triu_(1)  # zero out the lower diagonal
            mask = mask.unsqueeze(1)  # expand mask
            return mask

        self.text_model._build_causal_attention_mask = _build_causal_attention_mask.__get__(
            self.text_model, CLIPTextTransformer
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        output.last_hidden_state = output.last_hidden_state.to(torch.float32)
        return output


class IPUUNet2DConditionModel(UNet2DConditionModel, PipelineMixin):
    def change_cross_attention_processor(self, attn_matrix_target_mem_mb):
        for module in self.modules():
            if isinstance(module, CrossAttention):
                module.set_processor(IPUSlicedAttnProcessor(attn_matrix_target_mem_mb))

    def parallelize(self, attn_matrix_target_mem_mb=None):
        super().parallelize()

        self.change_cross_attention_processor(attn_matrix_target_mem_mb)

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


class UNetCastingWrapper(torch.nn.Module):
    """Schedulers differ in the dtype they store the timesteps in, so changing a
    scheduler would trigger a recompilation as the input dtype would change. This wrapper
    simply ensures that the timestep dtype provided to the PoplarExecutor is consistent."""

    def __init__(self, unet, input_dtype=torch.float16, output_dtype=torch.float32):
        super().__init__()
        self.unet = unet
        self.input_dtype = input_dtype
        self.output_dtype = output_dtype

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        timestep = timestep.repeat(self.unet.options.device_iterations)
        output = self.unet(
            sample.to(self.input_dtype), timestep.to(torch.float32), encoder_hidden_states.to(self.input_dtype)
        )
        output.sample = output.sample.to(self.output_dtype)
        return output

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "unet":
                raise AttributeError()
            return getattr(self.unet, name)


class IPUAutoencoderKL(AutoencoderKL, PipelineMixin):
    def parallelize(self):
        super().parallelize()

        self.post_quant_conv = poptorch.BeginBlock(self.post_quant_conv, "post_quant_conv", ipu_id=0)
        self.decoder.conv_in = poptorch.BeginBlock(self.decoder.conv_in, "decoder.conv_in", ipu_id=0)
        self.decoder.up_blocks[2].resnets[2] = poptorch.BeginBlock(
            self.decoder.up_blocks[2].resnets[2], "decoder.up_blocks[2].resnets[2]", ipu_id=1
        )

        return self

    def forward(self, z: torch.FloatTensor, return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        output = super().decode(z.to(self.decoder.conv_in.weight.dtype), return_dict=return_dict)
        output.sample = output.sample.to(torch.float32)
        return output

    def encode(self, x: torch.FloatTensor, return_dict: bool = True) -> AutoencoderKLOutput:
        h = self.encoder(x.to(self.encoder.conv_in.weight.dtype))
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments.to(torch.float32))

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)


def maybe_cast_module_to_float(module):
    if module is not None:
        module = module.float()
    return module


def override_module_eps(module, eps=1e-5):
    for child_module in module.modules():
        if hasattr(child_module, "eps"):
            setattr(child_module, "eps", eps)


class IPUStableDiffusionPipelineMixin:
    def __init__(
        self,
        vae,
        text_encoder,
        tokenizer,
        unet,
        scheduler,
        safety_checker,
        feature_extractor,
        requires_safety_checker=True,
        unet_ipu_config=None,
        text_encoder_ipu_config=None,
        vae_ipu_config=None,
        safety_checker_ipu_config=None,
    ):
        common_ipu_config = {
            "enable_half_partials": True,
            "executable_cache_dir": "./exe_cache",
            "inference_device_iterations": 1,
            "inference_replication_factor": 1,
        }

        def _get_poplar_executor(model, ipu_model_class, ipu_config):
            model_ipu_config = {**common_ipu_config, **ipu_config}
            model_ipu_config = IPUConfig.from_dict(model_ipu_config)

            model_ipu = copy.deepcopy(model).half()
            model_ipu.__class__ = ipu_model_class
            model_ipu.ipu_config = model_ipu_config
            model_ipu.parallelize()
            override_module_eps(model_ipu)

            opts = model_ipu_config.to_options(for_inference=True)
            return poptorch.inferenceModel(model_ipu.eval(), opts)

        if text_encoder_ipu_config is not None:
            logger.info("Running text_encoder on IPU.")
            text_encoder = _get_poplar_executor(text_encoder, IPUCLIPTextModel, text_encoder_ipu_config)
        else:
            logger.info("Running text_encoder on CPU.")
            text_encoder = maybe_cast_module_to_float(text_encoder)

        if vae_ipu_config is not None:
            logger.info("Running VAE decoder on IPU.")

            # Originally, latents are decoded by calling the decode method on the VAE. This isn't compatible with
            # Poptorch since the PoplarExecutor compilation and execution go via the __call__ special method.
            # We modify upstream such that VAE forward calls decode instead.
            # TODO: improve to also be compatible with encode.
            # Adapted from StableDiffusionPipeline.decode_latents.
            def decode_latents(self, latents):
                latents = 1 / 0.18215 * latents
                # IPU MOD
                # image = self.vae.decode(latents).sample
                image = self.vae(latents).sample
                image = (image / 2 + 0.5).clamp(0, 1)
                # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
                image = image.cpu().permute(0, 2, 3, 1).float().numpy()
                return image

            self.decode_latents = decode_latents.__get__(self)

            vae = _get_poplar_executor(vae, IPUAutoencoderKL, vae_ipu_config)

            if not isinstance(self, StableDiffusionPipeline):
                # Img2Img and Inpaint pipelines encode context images via vae.encode.
                # For now, we run the VAE encoder on CPU.
                logger.info("Running VAE encoder on CPU.")
                vae.encoder.float()
                vae.quant_conv.float()
        else:
            logger.info("Running VAE on CPU.")
            vae = maybe_cast_module_to_float(vae)

        if safety_checker is not None and safety_checker_ipu_config is not None:
            logger.info("Running safety_checker on IPU.")

            # The image coming out of decode_latents is a numpy ndarray. Before being passed to
            # the safety_checker running on IPU, we convert it to a torch tensor, and
            # convert it back to a numpy ndarray before returning it.
            # Adapted from StableDiffusionPipeline.run_safety_checker.
            def run_safety_checker(self, image, device, dtype):
                if self.safety_checker is not None:
                    safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(
                        device
                    )
                    image, has_nsfw_concept = self.safety_checker(
                        images=torch.tensor(image), clip_input=safety_checker_input.pixel_values.to(dtype)
                    )
                    if any(has_nsfw_concept):
                        logger.warning(
                            "Potential NSFW content was detected in one or more images. A black image will be returned instead. "
                            "Try again with a different prompt and/or seed."
                        )
                    image = image.numpy()
                else:
                    has_nsfw_concept = None
                return image, has_nsfw_concept

            self.run_safety_checker = run_safety_checker.__get__(self)

            safety_checker = _get_poplar_executor(
                safety_checker, IPUStableDiffusionSafetyChecker, safety_checker_ipu_config
            )
        else:
            logger.info("Running safety_checker on CPU.")
            safety_checker = maybe_cast_module_to_float(safety_checker)

        if unet_ipu_config is not None:
            logger.info("Running UNet on IPU.")

            attn_matrix_target_mem_mb = unet_ipu_config["attn_matrix_target_mem_mb"]

            unet_ipu_config = {**common_ipu_config, **unet_ipu_config}
            unet_ipu_config = IPUConfig.from_dict(unet_ipu_config)

            unet_ipu = copy.deepcopy(unet)
            unet_ipu.__class__ = IPUUNet2DConditionModel
            unet_ipu.ipu_config = unet_ipu_config
            unet_ipu.parallelize(attn_matrix_target_mem_mb=attn_matrix_target_mem_mb)
            override_module_eps(unet_ipu)

            opts = unet_ipu_config.to_options(for_inference=True)
            opts._Popart.set("saveInitializersToFile", "weights.onnx")
            opts._Popart.set("enableExplicitIR", True)
            unet_ipu = poptorch.inferenceModel(unet_ipu.eval(), opts)
            unet = UNetCastingWrapper(unet_ipu)
        else:
            logger.info("Running UNet on CPU.")
            unet = maybe_cast_module_to_float(unet)

        super().__init__(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            requires_safety_checker=requires_safety_checker,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        unet_ipu_config=None,
        text_encoder_ipu_config=None,
        vae_ipu_config=None,
        safety_checker_ipu_config=None,
        **kwargs,
    ):
        return super().from_pretrained(
            pretrained_model_name_or_path,
            unet_ipu_config=unet_ipu_config,
            text_encoder_ipu_config=text_encoder_ipu_config,
            vae_ipu_config=vae_ipu_config,
            safety_checker_ipu_config=safety_checker_ipu_config,
            **kwargs,
        )

    def set_attention_slice(self, slice_size: Optional[int]):
        # Another side effect of letting this go through is that CrossAttention could set
        # a different processor than what we intended, so do the simple thing for now.
        logger.warn(
            "Attention slicing is enabled by default. Specifying a custom value "
            "for the `slice_size` is currently unsupported."
        )

    def detach_from_device(self):
        for module in [self.text_encoder, self.unet.unet, self.vae, self.safety_checker]:
            if not isinstance(module, poptorch.PoplarExecutor):
                continue
            if module.isAttachedToDevice():
                module.detachFromDevice()
