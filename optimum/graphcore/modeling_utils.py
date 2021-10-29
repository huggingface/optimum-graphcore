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

from transformers import PreTrainedModel

from .ipu_configuration import IPUConfig

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


def to_pipelined(model: PreTrainedModel, ipu_config: IPUConfig):
    model_cls = model.__class__
    pipelined_cls = _PRETRAINED_TO_PIPELINED_REGISTRY.get(model_cls, None)
    if pipelined_cls is None:
        raise KeyError(f"Pipelined version of {model_cls} not found in registry.")
    return pipelined_cls.from_transformers(model, ipu_config)


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
