from diffusers import StableDiffusionImg2ImgPipeline

from .pipeline_stable_diffusion_mixin import IPUStableDiffusionPipelineMixin


class IPUStableDiffusionImg2ImgPipeline(IPUStableDiffusionPipelineMixin, StableDiffusionImg2ImgPipeline):
    pass
