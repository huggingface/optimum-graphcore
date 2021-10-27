# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

import copy
import yaml
import os
from typing import Any, Dict, Tuple, Union

import poptorch

import transformers
from transformers.file_utils import (
    PushToHubMixin,
    cached_path,
    copy_func,
    hf_bucket_url,
    is_offline_mode,
    is_remote_url,
    is_torch_available,
)
from transformers import PretrainedConfig
from transformers.utils import logging

# from . import __version__

logger = logging.get_logger(__name__)

CONFIG_NAME = "ipu_config.json"


class IPUConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        self.use_popdist = kwargs.pop("use_popdist", False)
        self.auto_round_num_ipus = kwargs.pop("auto_round_num_ipus", True)
        self.random_seed = kwargs.pop("random_seed", 42)
        self.replication_factor = kwargs.pop("replication_factor", 1)
        self.gradient_accumulation = kwargs.pop("gradient_accumulation", 1)
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


# class IPUConfig(PushToHubMixin):
#
#     def __init__(self, **kwargs):
#         """
#         Args:
#             config_path (:obj:`str`):
#                 Path to the YAML configuration file used to control the tuning behavior.
#         Returns:
#             config: IPUConfig object.
#         """
#         self.use_popdist = kwargs.pop("use_popdist", False)
#         self.auto_round_num_ipus = kwargs.pop("auto_round_num_ipus", True)
#         self.random_seed = kwargs.pop("random_seed", 42)
#         self.replication_factor = kwargs.pop("replication_factor", 1)
#         self.gradient_accumulation = kwargs.pop("gradient_accumulation", 1)
#
#         # TODO: set default config attributes.
#         for k, v in kwargs.items():
#             setattr(self, k, v)
#
#     def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
#         """
#         Save a configuration object to the directory ``save_directory``, so that it can be re-loaded using the
#         :func:`~transformers.PretrainedConfig.from_pretrained` class method.
#
#         Args:
#             save_directory (:obj:`str` or :obj:`os.PathLike`):
#                 Directory where the configuration YAML file will be saved (will be created if it does not exist).
#             push_to_hub (:obj:`bool`, `optional`, defaults to :obj:`False`):
#                 Whether or not to push your model to the Hugging Face model hub after saving it.
#
#                 .. warning::
#
#                     Using :obj:`push_to_hub=True` will synchronize the repository you are pushing to with
#                     :obj:`save_directory`, which requires :obj:`save_directory` to be a local clone of the repo you are
#                     pushing to if it's an existing folder. Pass along :obj:`temp_dir=True` to use a temporary directory
#                     instead.
#
#             kwargs:
#                 Additional key word arguments passed along to the
#                 :meth:`~transformers.file_utils.PushToHubMixin.push_to_hub` method.
#         """
#         if os.path.isfile(save_directory):
#             raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")
#
#         if push_to_hub:
#             commit_message = kwargs.pop("commit_message", None)
#             repo = self._create_or_get_repo(save_directory, **kwargs)
#
#         os.makedirs(save_directory, exist_ok=True)
#         # If we save using the predefined names, we can load using `from_pretrained`
#         output_config_file = os.path.join(save_directory, CONFIG_NAME)
#
#         self.to_json_file(output_config_file, use_diff=True)
#         logger.info(f"Configuration saved in {output_config_file}")
#
#         if push_to_hub:
#             url = self._push_to_hub(repo, commit_message=commit_message)
#             logger.info(f"Configuration pushed to the hub in this commit: {url}")
#
#     @classmethod
#     def from_pretrained(cls, config_name_or_path: Union[str, os.PathLike], **kwargs):
#         """
#         Instantiate an IPUConfig object from a configuration file which can either be hosted on
#         huggingface.co or from a local directory path.
#         Args:
#             config_name_or_path (:obj:`str`):
#                 Repository name in the Hugging Face Hub or path to a local directory containing the configuration file.
#             cache_dir (:obj:`str`, `optional`):
#                 Path to a directory in which a downloaded configuration should be cached if the standard cache should
#                 not be used.
#             force_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
#                 Whether or not to force to (re-)download the configuration files and override the cached versions if
#                 they exist.
#             resume_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
#                 Whether or not to delete incompletely received file. Attempts to resume the download if such a file
#                 exists.
#             revision(:obj:`str`, `optional`):
#                 The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
#                 git-based system for storing models and other artifacts on huggingface.co, so ``revision`` can be any
#                 identifier allowed by git.
#         Returns:
#             config: IPUConfig object.
#         """
#         config_dict, kwargs = cls.get_config_dict(config_name_or_path, **kwargs)
#         return cls.from_dict(config_dict, **kwargs)
#
#     @classmethod
#     def get_config_dict(
#         cls, config_name_or_path: Union[str, os.PathLike], **kwargs
#     ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
#         """
#         From a ``config_name_or_path``, resolve to a dictionary of parameters, to be used for instantiating a
#         :class:`~transformers.PretrainedConfig` using ``from_dict``.
#
#
#
#         Parameters:
#             config_name_or_path (:obj:`str` or :obj:`os.PathLike`):
#                 The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.
#
#         Returns:
#             :obj:`Tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the configuration object.
#
#         """
#         cache_dir = kwargs.pop("cache_dir", None)
#         force_download = kwargs.pop("force_download", False)
#         resume_download = kwargs.pop("resume_download", False)
#         proxies = kwargs.pop("proxies", None)
#         use_auth_token = kwargs.pop("use_auth_token", None)
#         local_files_only = kwargs.pop("local_files_only", False)
#         revision = kwargs.pop("revision", None)
#         from_pipeline = kwargs.pop("_from_pipeline", None)
#         from_auto_class = kwargs.pop("_from_auto", False)
#
#         user_agent = {"file_type": "config", "from_auto_class": from_auto_class}
#         if from_pipeline is not None:
#             user_agent["using_pipeline"] = from_pipeline
#
#         if is_offline_mode() and not local_files_only:
#             logger.info("Offline mode: forcing local_files_only=True")
#             local_files_only = True
#
#         config_name_or_path = str(config_name_or_path)
#         if os.path.isdir(config_name_or_path):
#             config_file = os.path.join(config_name_or_path, CONFIG_NAME)
#         elif os.path.isfile(config_name_or_path) or is_remote_url(config_name_or_path):
#             config_file = config_name_or_path
#         else:
#             config_file = hf_bucket_url(
#                 config_name_or_path, filename=CONFIG_NAME, revision=revision, mirror=None
#             )
#
#         try:
#             # Load from URL or cache if already cached
#             resolved_config_file = cached_path(
#                 config_file,
#                 cache_dir=cache_dir,
#                 force_download=force_download,
#                 proxies=proxies,
#                 resume_download=resume_download,
#                 local_files_only=local_files_only,
#                 use_auth_token=use_auth_token,
#                 user_agent=user_agent,
#             )
#             # Load config dict
#             config_dict = cls._dict_from_yaml_file(resolved_config_file)
#
#         except EnvironmentError as err:
#             logger.error(err)
#             msg = (
#                 f"Can't load config for '{config_name_or_path}'. Make sure that:\n\n"
#                 f"- '{config_name_or_path}' is a correct model identifier listed on 'https://huggingface.co/models'\n"
#                 f"  (make sure '{config_name_or_path}' is not a path to a local directory with something else, in that case)\n\n"
#                 f"- or '{config_name_or_path}' is the correct path to a directory containing a {CONFIG_NAME} file\n\n"
#             )
#
#             if revision is not None:
#                 msg += f"- or '{revision}' is a valid git identifier (branch name, a tag name, or a commit id) that exists for this model name as listed on its model page on 'https://huggingface.co/models'\n\n"
#
#             raise EnvironmentError(msg)
#
#         except (yaml.YAMLError, UnicodeDecodeError):
#             msg = (
#                 f"Couldn't reach server at '{config_file}' to download configuration file or "
#                 "configuration file is not a valid YAML file. "
#                 f"Please check network or file content here: {resolved_config_file}."
#             )
#             raise EnvironmentError(msg)
#
#         if resolved_config_file == config_file:
#             logger.info(f"loading configuration file {config_file}")
#         else:
#             logger.info(f"loading configuration file {config_file} from cache at {resolved_config_file}")
#
#         return config_dict, kwargs
#
#     @classmethod
#     def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> "IPUConfig":
#         """
#         Instantiates a :class:`~transformers.PretrainedConfig` from a Python dictionary of parameters.
#
#         Args:
#             config_dict (:obj:`Dict[str, Any]`):
#                 Dictionary that will be used to instantiate the configuration object. Such a dictionary can be
#                 retrieved from a pretrained checkpoint by leveraging the
#                 :func:`~transformers.PretrainedConfig.get_config_dict` method.
#             kwargs (:obj:`Dict[str, Any]`):
#                 Additional parameters from which to initialize the configuration object.
#
#         Returns:
#             :class:`PretrainedConfig`: The configuration object instantiated from those parameters.
#         """
#         return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
#
#         config = cls(**config_dict)
#
#         # Update config with kwargs if needed
#         to_remove = []
#         for key, value in kwargs.items():
#             if hasattr(config, key):
#                 setattr(config, key, value)
#                 if key != "torch_dtype":
#                     to_remove.append(key)
#         for key in to_remove:
#             kwargs.pop(key, None)
#
#         logger.info(f"Model config {config}")
#         if return_unused_kwargs:
#             return config, kwargs
#         else:
#             return config
#
#     @classmethod
#     def _dict_from_yaml_file(cls, yaml_file: Union[str, os.PathLike]):
#         with open(yaml_file, "r", encoding="utf-8") as reader:
#             d = yaml.safe_load(reader)
#         return d
#
#     def to_dict(self) -> Dict[str, Any]:
#         """
#         Serializes this instance to a Python dictionary.
#
#         Returns:
#             :obj:`Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
#         """
#         output = copy.deepcopy(self.__dict__)
#         if hasattr(self.__class__, "model_type"):
#             output["model_type"] = self.__class__.model_type
#
#         # Transformers version when serializing the model
#         # output["transformers_version"] = __version__
#
#         self.dict_torch_dtype_to_str(output)
#
#         return output
#
#     def update(self, config_dict: Dict[str, Any]):
#         """
#         Updates attributes of this class with attributes from ``config_dict``.
#
#         Args:
#             config_dict (:obj:`Dict[str, Any]`): Dictionary of attributes that should be updated for this class.
#         """
#         for key, value in config_dict.items():
#             setattr(self, key, value)
#
#     def dict_torch_dtype_to_str(self, d: Dict[str, Any]) -> None:
#         """
#         Checks whether the passed dictionary has a `torch_dtype` key and if it's not None, converts torch.dtype to a
#         string of just the type. For example, :obj:`torch.float32` get converted into `"float32"` string, which can
#         then be stored in the json format.
#         """
#         if d.get("torch_dtype", None) is not None and not isinstance(d["torch_dtype"], str):
#             d["torch_dtype"] = str(d["torch_dtype"]).split(".")[1]
