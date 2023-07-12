# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict

from optimum.utils import logging

from ....training_args import ALLOWED_N_IPU


logger = logging.get_logger(__name__)


# Deprecated.
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


# Deprecated.
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
    unet_config: Dict[str, Any],
    n_ipu: int = 4,
    num_prompts: int = 1,
    num_images_per_prompt: int = 1,
    **common_kwargs,
):
    if n_ipu not in ALLOWED_N_IPU:
        raise ValueError(
            f"{n_ipu=} is not a valid value for a Pod type, supported Pod types: {', '.join(ALLOWED_N_IPU)}"
        )

    # Infer base checkpoint model size for instantiating default IPU configs.
    cross_attention_dim = unet_config.cross_attention_dim
    sample_size = unet_config.sample_size
    model_ipu_configs = None
    if cross_attention_dim == 768:
        model_ipu_configs = STABLE_DIFFUSION_V1_512_IPU_CONFIG
    elif cross_attention_dim == 1024:
        if sample_size == 64:
            model_ipu_configs = STABLE_DIFFUSION_V2_512_IPU_CONFIG
        elif sample_size == 96:
            model_ipu_configs = STABLE_DIFFUSION_V2_768_IPU_CONFIG
    if model_ipu_configs is None:
        logger.warn(
            f"UNet config has a combination of `{cross_attention_dim=}` and `{sample_size=}` which we do not "
            "have known configs for (SD1 = (768, 64), SD2 512x512 = (1024, 64), SD2 768 x 768 = (1024, 96). "
            "Defaulting to the SD1 config."
        )

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
