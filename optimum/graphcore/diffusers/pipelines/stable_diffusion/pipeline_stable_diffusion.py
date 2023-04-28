from diffusers import StableDiffusionPipeline

from .pipeline_stable_diffusion_mixin import IPUStableDiffusionPipelineMixin


class IPUStableDiffusionPipeline(IPUStableDiffusionPipelineMixin, StableDiffusionPipeline):
    pass
