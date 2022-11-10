# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
"""Utilties related to FX."""
import inspect
import math
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import torch

import transformers
from transformers.models.auto import get_values
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_CTC_MAPPING_NAMES,
)
from transformers.utils.fx import HFAttribute, HFProxy, HFTracer, get_concrete_args

from ..modeling_utils import PipelineMixin


if TYPE_CHECKING:
    from transformers import PreTrainedModel


# TODO: keep this until transformers >= 4.23.2
class GCProxy(HFProxy):
    @property
    def dtype(self):
        return self.__getattr__("dtype")

    def __getattr__(self, k):
        if k == "_metadata":
            return self.__getattribute__(k)
        # note: not added to the graph yet, if this is a method call
        # we peephole optimize to the method invocation
        hf_attribute = HFAttribute(self, k)
        if hasattr(self, "_metadata"):
            hf_attribute.install_metadata(getattr(self._metadata, k))
        return hf_attribute


class PipelinedTracer(HFTracer):
    # TODO: keep this until transformers >= 4.23.2
    _TORCH_METHODS_TO_PATCH = list(HFTracer._TORCH_METHODS_TO_PATCH)
    _TORCH_METHODS_TO_PATCH.append("clamp")
    _TORCH_METHODS_TO_PATCH.append("rand")
    _TORCH_METHODS_TO_PATCH.append("finfo")
    """
    Tracer that enables tracing and transforming models to run them on IPUs.
    Compared to the HFTracer, this one adds the following features:
        - Ops can be wrapped (not only attributes of the torch module) to enable tracing.
        - Each node contains the "parent_module_qualified_name" attribute, specifying under which module the node was
        created. This is useful because some transformations need that, for instance RecomputationCheckpoint.
    """

    def __init__(self, autowrap_modules=(math,), autowrap_functions=()):
        super().__init__(autowrap_modules=autowrap_modules, autowrap_functions=autowrap_functions)
        self.ops_to_wrap = []
        self.current_module_qualified_name = ["root"]
        self.current_module_type = ["root"]
        self.root_is_in_half_precision = False

    def register_op_to_wrap(self, name, wrapper, orig_op):
        self.ops_to_wrap.append((name, wrapper, orig_op))

    def _patch_op(self, op_name: str, op_patched: "Callable"):
        names = op_name.split(".")
        module_names = names[1:-1]
        attr_name = names[-1]
        mod = torch
        for module_name in module_names:
            mod = getattr(mod, module_name)
        setattr(mod, attr_name, op_patched)

    def wrap_ops(self):
        for name, wrapper, _ in self.ops_to_wrap:
            self._patch_op(name, wrapper)

    def unwrap_ops(self):
        for name, _, orig_op in self.ops_to_wrap:
            self._patch_op(name, orig_op)

    def proxy(self, node):
        # Would be better to update the created node in TracerBase.create_node, but this method has less arguments, so
        # it is easier to use this one, and equivalent.
        node.parent_module_qualified_name = self.current_module_qualified_name[-1]
        node.parent_module_type = self.current_module_type[-1]
        return GCProxy(node, self)
        # return gc_proxy
        return super().proxy(node)

    def call_module(self, m, forward, args, kwargs):
        # Could be done in a "cleaner" fashion by inlining the content of Tracer.call_module.
        # Preferred to inherint from it and do it that way instead.
        module_qualified_name = self.path_of_module(m)
        is_leaf_module = self.is_leaf_module(m, module_qualified_name)
        if not is_leaf_module:
            self.current_module_qualified_name.append(module_qualified_name)
            self.current_module_type.append(type(m))
        self.orig_forward = forward
        proxy = super().call_module(m, forward, args, kwargs)
        if not is_leaf_module:
            self.current_module_qualified_name.pop(-1)
            self.current_module_type.pop(-1)
        return proxy

    def create_proxy(self, kind, target, args, kwargs, name=None, type_expr=None, proxy_factory_fn=None):
        # if self.root_is_in_half_precision:
        #     float32_dtype_in_args = any(a is torch.float32 for a in args)
        #     float32_dtype_in_kwargs = kwargs.get("dtype", None) is torch.float32
        #     node_types_to_inspect = [
        #         ("call_method", "to"),
        #         ("call_function", torch.full),
        #     ]
        #     torch_methods_to_patched_version = {
        #         orig: wrapped for (orig, wrapped) in self.patched_torch_methods.values()
        #     }
        #     for (k, t) in node_types_to_inspect:
        #         if kind == k and target == torch_methods_to_patched_version.get(t, t):
        #             if float32_dtype_in_args:
        #                 args = tuple(a if a is not torch.float32 else torch.float16 for a in args)
        #             if float32_dtype_in_kwargs:
        #                 kwargs["dtype"] = torch.float16
        return super().create_proxy(
            kind, target, args, kwargs, name=name, type_expr=type_expr, proxy_factory_fn=proxy_factory_fn
        )

    # TODO: keep until transformers 4.23.2 is released.
    def _generate_dummy_input(
        self, model: "PreTrainedModel", input_name: str, shape: List[int]
    ) -> Dict[str, torch.Tensor]:
        input_dict = {}
        model_class_name = getattr(model, "class_for_deserialization", model.__class__).__name__
        if input_name == "labels":
            if model_class_name in get_values(MODEL_FOR_CTC_MAPPING_NAMES):
                input_dict["labels"] = torch.zeros(*shape, dtype=torch.float, device=model.device)
            if model_class_name in get_values(MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES):
                input_dict["labels"] = torch.zeros(shape[0], dtype=torch.long, device=model.device)
        if "labels" not in input_dict:
            input_dict = super()._generate_dummy_input(model, input_name, shape)
        return input_dict

    def trace(self, *args, **kwargs) -> torch.fx.Graph:
        root = args[0]
        if not isinstance(root, torch.nn.Module):
            # Cannot infer easily.
            self.root_is_in_half_precision = False
        else:
            self.root_is_in_half_precision = any(p.dtype is torch.float16 for p in root.parameters())
        graph = super().trace(*args, **kwargs)
        self.root_is_in_half_precision = False
        return graph


def symbolic_trace_with_pipelined_tracer(
    model: PipelineMixin,
    input_names: Optional[List[str]] = None,
) -> torch.fx.GraphModule:
    """
    Performs symbolic tracing on the model.

    Args:
        model ([`PretrainedModel`]):
            The model to trace.
        input_names (`List[str]`, *optional*):
            The names of the inputs of the traced model. If unset, model.dummy_inputs.keys() are used instead.
    Returns:
        `torch.fx.GraphModule`: A GraphModule constructed by recording operations seen while tracing the model.
    """
    if input_names is None:
        input_names = model.dummy_inputs.keys()

    input_names = list(input_names)
    concrete_args = get_concrete_args(model, input_names)

    # Tracing.
    tracer = PipelinedTracer()
    for wrap_info in model.get_ops_to_wrap_for_tracing():
        tracer.register_op_to_wrap(*wrap_info)
    tracer.wrap_ops()
    traced_graph = tracer.trace(model, concrete_args=concrete_args)
    tracer.unwrap_ops()

    traced = torch.fx.GraphModule(model, traced_graph)

    # The model class must be stored as an attribute to allow model deserialization, which uses trace, and thus
    # _generate_dummy_input, where the model class is needed.
    traced.class_for_deserialization = model.__class__
    traced.device = model.device

    for name, attr in vars(model).items():
        setattr(traced, name, getattr(traced, name, attr))

    return traced


def cast_traced_model_to_proper_class(model: torch.nn.Module, traced: torch.fx.GraphModule):
    """Casts the traced `torch.fx.GraphModule` to the original class of the traced model."""
    type_ = type(f"Traced{model.__class__.__name__}", (torch.fx.GraphModule, model.__class__), {})
    traced.__class__ = type_
    traced.recompile()


def symbolic_trace_pipelined_model(pipelined_model: PipelineMixin) -> PipelineMixin:
    """
    Traces a pipelined model and casts the traced model to the original class of the model.

    Args:
        pipelined_model ([`~PipelineMixin`]):
            The pipelined model.

    Returns:
        [`~PipelineMixin`]: The traced model.
    """
    if isinstance(pipelined_model, torch.fx.GraphModule):
        return pipelined_model

    transformers_class = None
    bases = list(pipelined_model.__class__.__bases__)

    while bases:
        base = bases.pop(0)
        if inspect.getmodule(base).__name__.startswith("transformers") and transformers.PreTrainedModel in base.mro():
            transformers_class = base
            break
        bases += list(base.__bases__)

    # Trick to make HFTracer._generate_dummy_input work with the pipelined class.
    # This attribute will be set properly in symbolic_trace_with_pipelined_tracer once tracing is done.
    pipelined_model.class_for_deserialization = transformers_class
    traced = symbolic_trace_with_pipelined_tracer(
        pipelined_model, input_names=pipelined_model.input_names_for_symbolic_trace
    )
    cast_traced_model_to_proper_class(pipelined_model, traced)
    return traced