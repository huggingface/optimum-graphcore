#  Copyright 2021 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os
from typing import Union

import poptorch
import transformers
from transformers import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

CONFIG_NAME = "ipu_config.json"


class IPUConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        self.use_popdist = kwargs.pop("use_popdist", False)
        self.auto_round_num_ipus = kwargs.pop("auto_round_num_ipus", True)
        self.random_seed = kwargs.pop("random_seed", 42)
        self.replication_factor = kwargs.pop("replication_factor", 1)
        self.gradient_accumulation = kwargs.pop("gradient_accumulation", 1)
        self.device_iterations = kwargs.pop("device_iterations", 1)
        self.ipus_per_replica = kwargs.pop("ipus_per_replica", 1)
        # if self.ipus_per_replica is None:
        #     raise KeyError("ipus_per_replica must be provided")
        self.matmul_proportion = kwargs.pop("matmul_proportion", 0.6)
        if isinstance(self.matmul_proportion, float):
            self.matmul_proportion = [self.matmul_proportion]
        if len(self.matmul_proportion) == 1:
            self.matmul_proportion = self.matmul_proportion * self.ipus_per_replica

        self.profile_dir = kwargs.pop("profile_dir", None)

        # TODO: set default config attributes.
        for k, v in kwargs.items():
            setattr(self, k, v)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        orig_transformers_config_name = transformers.file_utils.CONFIG_NAME
        transformers.configuration_utils.CONFIG_NAME = CONFIG_NAME
        super().save_pretrained(save_directory, push_to_hub=push_to_hub, **kwargs)
        transformers.configuration_utils.CONFIG_NAME = orig_transformers_config_name

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs):
        orig_transformers_config_name = transformers.file_utils.CONFIG_NAME
        transformers.configuration_utils.CONFIG_NAME = CONFIG_NAME
        ipu_config = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        transformers.configuration_utils.CONFIG_NAME = orig_transformers_config_name
        return ipu_config

    def to_options(self) -> poptorch.Options:
        raise NotImplementedError()

    @property
    def batch_size_factor(self) -> int:
        return self.replication_factor * self.gradient_accumulation_steps * self.device_iterations
