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
import math
from typing import Callable, List, Optional

import torch

import transformers
from transformers.utils.fx import HFTracer, get_concrete_args

from ..modeling_utils import PipelineMixin


class PipelinedTracer(HFTracer):
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
        proxy = super().proxy(node)
        return proxy

    def call_module(self, m, forward, args, kwargs):
        # Could be done in a "cleaner" fashion by inlining the content of Tracer.call_module.
        # Preferred to inherint from it and do it that way instead.
        module_qualified_name = self.path_of_module(m)
        is_leaf_module = self.is_leaf_module(m, module_qualified_name)
        if not is_leaf_module:
            self.current_module_qualified_name.append(module_qualified_name)
        self.orig_forward = forward
        proxy = super().call_module(m, forward, args, kwargs)
        if not is_leaf_module:
            self.current_module_qualified_name.pop(-1)
        return proxy


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


def symbolic_trace_pipelined_model(pipelined_model: PipelineMixin) -> PipelineMixin:
    if isinstance(pipelined_model, torch.fx.GraphModule):
        return pipelined_model

    transformers_class = None
    for base in pipelined_model.__class__.__bases__:
        if transformers.PreTrainedModel in base.__mro__:
            transformers_class = base
            break

    # Trick to make HFTracer._generate_dummy_input work with the pipelined class.
    # This attribute will be set properly in symbolic_trace_with_pipelined_tracer once tracing is done.
    pipelined_model.class_for_deserialization = transformers_class
    traced = symbolic_trace_with_pipelined_tracer(pipelined_model, input_names=pipelined_model.input_names)
    type_ = type(f"Traced{pipelined_model.__class__.__name__}", (torch.fx.GraphModule, pipelined_model.__class__), {})
    traced.__class__ = type_
    return traced
