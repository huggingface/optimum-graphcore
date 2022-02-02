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

    from typing import Any, Dict, Optional

    import torch

    from transformers.modeling_outputs import ModelOutput

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
    ) -> Dict[str, Any]:
        # 2. prepare encoder args and encoder kwargs from model kwargs
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        # 1. get encoder
        compiled_encoder = getattr(self, "_compiled_encoder", None)
        if compiled_encoder is None:
            encoder = self.get_encoder()
            # TODO: how to pass the poptorch options?
            compiled_encoder = poptorch.inferenceModel(encoder.eval())
            compiled_encoder.compile(**encoder_kwargs)

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        model_kwargs["encoder_outputs"]: ModelOutput = compiled_encoder(**encoder_kwargs)

        return model_kwargs


def _get_layer_ipu(layers_per_ipu):
    # List of the IPU Id for each encoder layer
    layer_ipu = []
    for ipu, n_layers in enumerate(layers_per_ipu):
        layer_ipu += [ipu] * n_layers
    return layer_ipu


def recomputation_checkpoint(module: nn.Module):
    """Annotates the output of a module to be checkpointed instead of
    recomputed"""

    def recompute_outputs(module, inputs, outputs):
        return tuple(poptorch.recomputationCheckpoint(y) for y in outputs)

    module.register_forward_hook(recompute_outputs)


def outline_attribute(module: nn.Module, value: str):
    """Adds an attribute to a module. This attribute will be used
    when comparing operation equivalence in outlining. For example:

    layer1 = nn.Linear(...)
    layer2 = nn.Linear(...)
    layer3 = nn.Linear(...)
    layer4 = nn.Linear(...)
    outline_attribute(layer1, "A")
    outline_attribute(layer2, "A")
    outline_attribute(layer3, "B")

    The code for layer1 can be reused for layer2.
    But it can't be used for layer3 or layer4.
    """
    context = poptorch.Attribute(__outline={"layer": value})

    def enable(*args):
        context.__enter__()

    def disable(*args):
        context.__exit__(None, None, None)

    module.register_forward_pre_hook(enable)
    module.register_forward_hook(disable)


class SerializedEmbedding(nn.Module):
    """
    Wrapper for `nn.Embedding` layer that performs the embedding look-up into
    smaller serialized steps in order to reduce memory in the embedding gradient
    calculation.

    Args:
        embedding: A `nn.Embedding` to wrap
        serialization_factor: The number of serialized embedding look-ups
    """

    def __init__(self, embedding: nn.Embedding, serialization_factor: int):
        super().__init__()
        self.serialization_factor = serialization_factor
        self.num_embeddings = embedding.num_embeddings

        # Num embeddings should be divisible by the serialization factor
        assert self.num_embeddings % self.serialization_factor == 0
        self.split_size = self.num_embeddings // self.serialization_factor
        self.split_embeddings = nn.ModuleList(
            [
                nn.Embedding.from_pretrained(
                    embedding.weight[i * self.split_size : (i + 1) * self.split_size, :].detach(),
                    freeze=False,
                    padding_idx=embedding.padding_idx if i == 0 else None,
                )
                for i in range(self.serialization_factor)
            ]
        )

    def deserialize(self):
        """
        Deserialize the internal wrapped embedding layer and return it as a
        `nn.Embedding` object.

        Returns:
            `nn.Embedding` layer
        """
        return nn.Embedding.from_pretrained(torch.vstack([l.weight for l in self.split_embeddings]), padding_idx=0)

    def forward(self, indices):
        # iterate through the splits
        x_sum = None
        for i in range(self.serialization_factor):
            # mask out the indices not in this split
            split_indices = indices - i * self.split_size
            mask = (split_indices >= 0) * (split_indices < self.split_size)
            mask = mask.detach()
            split_indices *= mask

            # do the embedding lookup
            x = self.split_embeddings[i](split_indices)

            # multiply the output by mask
            x *= mask.unsqueeze(-1)

            # add to partial
            if x_sum is not None:
                x_sum += x
            else:
                x_sum = x
        return x_sum


class SerializedLinear(nn.Linear):
    """
    Exactly equivalent to `nn.Linear` layer, but with the matrix multiplication replaced with
    a serialized matrix multiplication: `poptorch.serializedMatMul`.
    The matrix multiplication is split into separate smaller multiplications, calculated one after the other,
    to reduce the memory requirements of the multiplication and its gradient calculation.

    Args:
        in_features: Size of each input sample
        out_features: Size of each output sample
        factor: Number of serialized multiplications. Must be a factor of
            the dimension to serialize on.
        bias: If set to False, the layer will not learn an additive bias.
            Default: True
        mode: Which dimension of the matmul to serialize on:
            for matrix A (m by n) multiplied by matrix B (n by p).
            * InputChannels: Split across the input channels (dimension m).
            * ReducingDim: Split across the reducing dimension (n).
            * OutputChannels: Split across the output channels (dimension p).
            * Disabled: Same as an ordinary matrix multiplication.
    """

    def __init__(
        self, in_features, out_features, factor, bias=False, mode=poptorch.MatMulSerializationMode.OutputChannels
    ):
        super().__init__(in_features, out_features, bias)
        self.mode = mode
        self.factor = factor

    def forward(self, x):
        output = poptorch.serializedMatMul(x, self.weight.t(), self.mode, self.factor)
        if self.bias is not None:
            output += self.bias
        return output
