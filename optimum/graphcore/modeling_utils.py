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

import copy

from torch import nn

import poptorch
from optimum.utils import logging
from transformers import AutoConfig, PreTrainedModel

from .ipu_configuration import IPUConfig


logger = logging.get_logger(__name__)

_PRETRAINED_TO_PIPELINED_REGISTRY = {}


def register(transformers_cls=None):
    def wrapper(cls):
        orig_cls = transformers_cls
        if orig_cls is None:
            found = False
            for base_cls in cls.__bases__:
                if base_cls != PipelineMixin:
                    orig_cls = base_cls
                    found = True
                    break
            if not found:
                raise ValueError(f"Was not able to find original transformers class for {cls}")
        _PRETRAINED_TO_PIPELINED_REGISTRY[orig_cls] = cls
        return cls

    return wrapper


def to_pipelined(model: nn.Module, ipu_config: IPUConfig, force: bool = False):
    model_cls = model.__class__
    pipelined_cls = _PRETRAINED_TO_PIPELINED_REGISTRY.get(model_cls, None)
    if pipelined_cls is not None:
        return pipelined_cls.from_transformers(model, ipu_config)
    else:
        if force:
            logger.warning(
                f"No pipelined version exists for {model_cls.__name__}, creating it dynamically, it might not work as expected."
            )
            pipelined_cls = type(f"Pipelined{model_cls.__name__}", (model_cls, PipelineMixin), {})
            return pipelined_cls.from_model(model)

        else:
            raise KeyError(f"{model_cls.__name__} pipelined version not found in registry.")


class PipelineMixin:
    @classmethod
    def from_transformers(cls, model: PreTrainedModel, ipu_config: IPUConfig):
        # TODO: make this cleaner.
        # For now, everything is put in the model config to make things simpler.
        config = copy.deepcopy(model.config)
        config.update(ipu_config.to_dict())
        pipelined_model = cls(config)
        pipelined_model.load_state_dict(model.state_dict())
        return pipelined_model

    @classmethod
    def from_pretrained_transformers(cls, model_name_or_path: str, ipu_config: IPUConfig):
        config = AutoConfig.from_pretrained(model_name_or_path)
        config.update(ipu_config.to_dict())
        return cls.from_pretrained(model_name_or_path, config=config)

    @classmethod
    def from_model(cls, model: nn.Module):
        clone = copy.deepcopy(model)
        # It is fine because PipelineMixin only adds functionality, it does not add any attribute.
        clone.__class__ = cls
        return clone

    def parallelize(self):
        """Transform the model to run in an IPU pipeline."""
        self._hooks = []
        return self

    def deparallelize(self):
        """
        Undo the changes to the model done by `parallelize`.
        You should call this before doing `save_pretrained` so that the `model.state_dict` is fully compatible with the
        original model.
        """
        # Remove hooks
        if hasattr(self, "_hooks"):
            for h in self._hooks:
                h.remove()
        # Remove poptorch Blocks
        for m in self.modules():
            if m is not self:
                poptorch.removeBlocks(m)
        return self

    def num_parameters(self, only_trainable: bool = False, exclude_embeddings: bool = False) -> int:
        """
        Get number of (optionally, trainable or non-embeddings) parameters in the module.

        Args:
            only_trainable (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to return only the number of trainable parameters

            exclude_embeddings (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to return only the number of non-embeddings parameters

        Returns:
            :obj:`int`: The number of parameters.
        """

        # TODO: actually overwrite this to handle SerializedEmbedding.
        if exclude_embeddings:
            embedding_param_names = [
                f"{name}.weight" for name, module_type in self.named_modules() if isinstance(module_type, nn.Embedding)
            ]
            non_embedding_parameters = [
                parameter for name, parameter in self.named_parameters() if name not in embedding_param_names
            ]
            return sum(p.numel() for p in non_embedding_parameters if p.requires_grad or not only_trainable)
        else:
            return sum(p.numel() for p in self.parameters() if p.requires_grad or not only_trainable)
