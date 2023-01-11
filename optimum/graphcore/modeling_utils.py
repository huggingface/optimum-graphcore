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
from inspect import signature
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

import poptorch
from optimum.utils import logging
from transformers import AutoConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

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
    # If the user defined his/her own model and already subclassed from PipelineMixin. I.e., the model is already pipelined.
    elif isinstance(model, PipelineMixin):
        clone = copy.deepcopy(model)
        clone.ipu_config = copy.deepcopy(ipu_config)
        return clone
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
        """
        Creates a pipeline model from a [`~transformers.PreTrainedModel`].

        Args:
            model ([`~transformers.PreTrainedModel`]):
                The model to convert to a pipelined model.
            ipu_config ([`IPUConfig`]):
                The IPUConfig of the pipelined model.

        Returns:
            The pipelined version of the model.
        """
        config = copy.deepcopy(model.config)
        pipelined_model = cls(config)
        pipelined_model.load_state_dict(model.state_dict())
        pipelined_model.ipu_config = copy.deepcopy(ipu_config)
        return pipelined_model

    @classmethod
    def from_pretrained_transformers(cls, model_name_or_path: str, ipu_config: IPUConfig, *model_args, **kwargs):
        """
        Creates a pipeline model by using `from_pretrained`.

        Args:
            model_name_or_path (`str`):
                The model name or path.
            ipu_config ([`IPUConfig`]):
                The IPUConfig of the pipelined model.
            model_args (`Tuple[Any]`):
                The positional arguments to use when instantiating the model.
            kwargs (`Dict[str, Any]`):
                The keyword arguments to use when instantiating the model.

        Returns:
            The pipelined model.
        """
        pipelined_model = cls.from_pretrained(model_name_or_path, *model_args, **kwargs)
        pipelined_model.ipu_config = copy.deepcopy(ipu_config)
        return pipelined_model

    @classmethod
    def from_model(cls, model: nn.Module):
        clone = copy.deepcopy(model)
        clone.__class__ = cls
        # Just needed so that .parallelize() does not throw an error
        clone.ipu_config = IPUConfig()
        return clone

    def _has_ipu_config_check(self):
        _ipu_config = getattr(self, "_ipu_config", None)
        if _ipu_config is None:
            raise AttributeError("No IPUConfig was found, please set the ipu_config attribute")

    @property
    def ipu_config(self):
        """Property that checks that the model has an [`IPUConfig`] attached, and returns it."""
        self._has_ipu_config_check()
        return self._ipu_config

    @ipu_config.setter
    def ipu_config(self, value: IPUConfig):
        if not isinstance(value, IPUConfig):
            raise TypeError(f"ipu_config must be an instance of IPUConfig, but {type(value)} was provided")
        self._ipu_config = value

    def parallelize(self):
        """Transforms the model to run in an IPU pipeline."""
        self._hooks = []
        self._has_ipu_config_check()
        return self

    def deparallelize(self):
        """
        Undoes the changes to the model done by `parallelize`.
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


class GenerationMethodsMixin:
    def get_encoder(
        self,
        device_iterations: Optional[int] = None,
        replication_factor: Optional[int] = None,
        for_inference: bool = True,
    ):
        if not hasattr(self, "_wrapped_encoder"):
            encoder = super().get_encoder()
            if self.ipu_config.execute_encoder_on_cpu_for_generation:
                self._wrapped_encoder = encoder.to(torch.float32)
            else:
                self.eval_opts = self.ipu_config.to_options(for_inference=True)
                self._wrapped_encoder = poptorch.inferenceModel(
                    encoder, options=self.ipu_config.to_options(for_inference=True)
                )
        return self._wrapped_encoder

    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, **kwargs) -> Dict[str, Any]:
        inputs = super().prepare_inputs_for_generation(input_ids, **kwargs)
        return {k: v for k, v in inputs.items() if k in signature(self._forward_for_generate).parameters}


def get_layer_ipu(layers_per_ipu: List[int], target_number_of_layers: Optional[Union[int, List]] = None):
    # List of the IPU Id for each encoder layer
    layer_ipu: List[int] = []
    for ipu, n_layers in enumerate(layers_per_ipu):
        layer_ipu += [ipu] * n_layers
    if target_number_of_layers is not None:
        if not isinstance(target_number_of_layers, (int, float)):
            target_number_of_layers = len(target_number_of_layers)
        if len(layer_ipu) < target_number_of_layers:
            raise ValueError(
                "layers_per_ipu does not define enough layers for the current model."
                " The current IPU Config specifies IPU assignments for "
                f"{len(layer_ipu)} layers but there are {target_number_of_layers} layers "
                f"in the model. layers_per_ipu={layers_per_ipu}"
            )
    return layer_ipu


def recomputation_checkpoint(module: nn.Module) -> torch.utils.hooks.RemovableHandle:
    """Annotates the output of a module to be checkpointed instead of
    recomputed"""

    def recompute_outputs(module, inputs, outputs):
        if isinstance(outputs, torch.Tensor):
            return poptorch.recomputationCheckpoint(outputs)
        elif isinstance(outputs, tuple):
            return tuple(poptorch.recomputationCheckpoint(y) for y in outputs)

    return module.register_forward_hook(recompute_outputs)


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

    handles = []
    handles.append(module.register_forward_pre_hook(enable))
    handles.append(module.register_forward_hook(disable))
    return handles


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
        self,
        in_features,
        out_features,
        factor,
        bias=False,
        mode=poptorch.MatMulSerializationMode.OutputChannels,
    ):
        super().__init__(in_features, out_features, bias)
        self.mode = mode
        self.factor = factor

    def forward(self, x):
        if not self.training:
            output = super().forward(x)
        else:
            output = poptorch.serializedMatMul(x, self.weight.t(), self.mode, self.factor)
            if self.bias is not None:
                output += self.bias
        return output


class SharedEmbedding(nn.Module):
    """Wrapper around the shared embedding between the encoder and the decoder stacks.

    Attributes:
        shared: The shared embedding layer.
    """

    def __init__(self, shared: nn.Embedding):
        super().__init__()
        self.shared = shared

    def _combine_inputs(self, input_ids: torch.Tensor, decoder_input_ids: torch.Tensor) -> Tuple[int, torch.Tensor]:
        idx = input_ids.size(1)
        return idx, torch.cat([input_ids, decoder_input_ids], dim=1)

    def _separate_inputs(self, idx: int, embeds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return embeds[:, :idx, :], embeds[:, idx:, :]

    def forward(
        self,
        input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        encoder_embed_scale: Optional[float] = None,
        decoder_embed_scale: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: use this once the TiedGather pattern issue is solved.
        # encoder_inputs_embeds, decoder_inputs_embeds = None, None
        # if input_ids is not None and encoder_embed_scale is not None:
        #     encoder_inputs_embeds = self.shared(input_ids) * encoder_embed_scale
        # if decoder_input_ids is not None and decoder_embed_scale is not None:
        #     decoder_inputs_embeds = self.shared(decoder_input_ids) * decoder_embed_scale
        # combined, n1, n2 = self._combine_inputs(input_ids, decoder_input_ids)
        # encoder_inputs_embeds, decoder_inputs_embeds = self._separate_inputs(self.shared(combined), n1, n2)
        idx, combined = self._combine_inputs(input_ids, decoder_input_ids)
        encoder_inputs_embeds, decoder_inputs_embeds = self._separate_inputs(idx, self.shared(combined))

        if encoder_embed_scale:
            encoder_inputs_embeds = encoder_inputs_embeds * encoder_embed_scale
        if decoder_embed_scale:
            decoder_inputs_embeds = decoder_inputs_embeds * decoder_embed_scale

        return encoder_inputs_embeds, decoder_inputs_embeds


class OnehotGather(nn.Module):
    """
    Gathers selected indices from a tensor by transforming the list of indices
    into a one-hot matrix and then multiplying the tensor by that matrix.
    """

    def forward(self, sequence, positions):
        """
        Gather the vectors at the specific positions over a batch.
        """
        num_classes = int(sequence.shape[1])
        one_hot_positions = F.one_hot(positions, num_classes).to(dtype=sequence.dtype)
        return torch.matmul(one_hot_positions.detach(), sequence)
