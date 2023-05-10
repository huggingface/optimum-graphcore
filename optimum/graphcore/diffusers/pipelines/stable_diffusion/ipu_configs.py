from optimum.utils import logging

from ....training_args import ALLOWED_N_IPU


logger = logging.get_logger(__name__)


INFERENCE_ENGINES_TO_MODEL_NAMES = {
    "stable-diffusion-v1": "CompVis/stable-diffusion-v1-4",  # this is a guess
    "stable-diffusion-v1-5": "runwayml/stable-diffusion-v1-5",
    "stable-diffusion-512-v2-0": "stabilityai/stable-diffusion-2-base",
    "stable-diffusion-768-v2-0": "stabilityai/stable-diffusion-2",
    "stable-diffusion-512-v2-1": "stabilityai/stable-diffusion-2-1-base",
    "stable-diffusion-768-v2-1": "stabilityai/stable-diffusion-2-1",
    "stable-inpainting-v1-0": "runwayml/stable-diffusion-inpainting",
    "stable-inpainting-512-v2-0": "stabilityai/stable-diffusion-2-inpainting",
}


STABLE_DIFFUSION_V1_512_IPU_CONFIG = {
    "text_encoder": {
        "ipus_per_replica": 1,
        "matmul_proportion": 0.6,
    },
    "unet": {
        "ipus_per_replica": 4,
        "matmul_proportion": [0.6, 0.6, 0.1, 0.3],
        "attn_matrix_target_mem_mb": 100,
    },
    "vae": {"ipus_per_replica": 2, "matmul_proportion": 0.3},
    "safety_checker": {"ipus_per_replica": 1, "matmul_proportion": 0.6},
}


STABLE_DIFFUSION_V2_512_IPU_CONFIG = {
    "text_encoder": {
        "ipus_per_replica": 1,
        "matmul_proportion": 0.6,
    },
    "unet": {
        "ipus_per_replica": 4,
        "matmul_proportion": [0.6, 0.6, 0.1, 0.3],
        "attn_matrix_target_mem_mb": 100,
    },
    "vae": {"ipus_per_replica": 2, "matmul_proportion": 0.3},
    "safety_checker": {"ipus_per_replica": 1, "matmul_proportion": 0.6},
}


STABLE_DIFFUSION_V2_768_IPU_CONFIG = {
    "text_encoder": {
        "ipus_per_replica": 1,
        "matmul_proportion": 0.6,
    },
    "unet": {
        "ipus_per_replica": 4,
        "matmul_proportion": [0.06, 0.1, 0.1, 0.1],
        "attn_matrix_target_mem_mb": 45,
    },
    "vae": None,  # not supported yet
    "safety_checker": {"ipus_per_replica": 1, "matmul_proportion": 0.6},
}


INFERENCE_ENGINES_TO_IPU_CONFIGS = {
    "stable-diffusion-v1": STABLE_DIFFUSION_V1_512_IPU_CONFIG,  # this is a guess
    "stable-diffusion-v1-5": STABLE_DIFFUSION_V1_512_IPU_CONFIG,
    "stable-diffusion-512-v2-0": STABLE_DIFFUSION_V2_512_IPU_CONFIG,
    "stable-diffusion-768-v2-0": STABLE_DIFFUSION_V2_768_IPU_CONFIG,
    "stable-diffusion-512-v2-1": STABLE_DIFFUSION_V2_512_IPU_CONFIG,
    "stable-diffusion-768-v2-1": STABLE_DIFFUSION_V2_768_IPU_CONFIG,
    "stable-inpainting-v1-0": STABLE_DIFFUSION_V1_512_IPU_CONFIG,
    "stable-inpainting-512-v2-0": STABLE_DIFFUSION_V2_512_IPU_CONFIG,
}


def get_default_ipu_configs(
    engine: str = "stable-diffusion-512-v2-0",
    height: int = 512,
    width: int = 512,
    num_prompts: int = 1,
    num_images_per_prompt: int = 1,
    n_ipu: int = 4,
    **common_kwargs,
):
    if engine not in INFERENCE_ENGINES_TO_MODEL_NAMES:
        raise ValueError(f"{engine} should be one of {', '.join(INFERENCE_ENGINES_TO_MODEL_NAMES)}")
    if n_ipu not in ALLOWED_N_IPU:
        raise ValueError(
            f"{n_ipu=} is not a correct value for a POD type, supported POD types: {', '.join(ALLOWED_N_IPU)}"
        )

    default_image_dim = 768 if "768" in engine else 512
    if default_image_dim == 768 and height < default_image_dim and width < default_image_dim:
        logger.warn(
            "Generating an image of a size smaller than 768x768 with a checkpoint fine-tuned for 768x768 "
            "can lead to images of poor quality."
        )

    model_ipu_configs = INFERENCE_ENGINES_TO_IPU_CONFIGS[engine]

    unet_ipu_config = model_ipu_configs["unet"]
    text_encoder_ipu_config = model_ipu_configs["text_encoder"] if n_ipu > 4 else None
    vae_ipu_config = model_ipu_configs["vae"] if n_ipu > 4 else None
    safety_checker_ipu_config = model_ipu_configs["safety_checker"] if n_ipu > 4 else None

    # Set the micro batch size at 1 for now.
    common_kwargs["inference_device_iterations"] = num_prompts * num_images_per_prompt

    unet_ipu_config = {**unet_ipu_config, **common_kwargs}
    if text_encoder_ipu_config:
        text_encoder_ipu_config = {**text_encoder_ipu_config, **common_kwargs}
        # The text encoder is only run once per single prompt or batch of prompts,
        # then outputs are duplicated by num_images_per_prompt.
        text_encoder_ipu_config["inference_device_iterations"] = num_prompts
    if vae_ipu_config:
        vae_ipu_config = {**vae_ipu_config, **common_kwargs}
    if safety_checker_ipu_config:
        safety_checker_ipu_config = {**safety_checker_ipu_config, **common_kwargs}

    return unet_ipu_config, text_encoder_ipu_config, vae_ipu_config, safety_checker_ipu_config
