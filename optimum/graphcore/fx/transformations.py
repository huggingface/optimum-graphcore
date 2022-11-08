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
"""All the parallelization-related transformations."""

import collections
import operator
import re
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

import torch

import poptorch
from optimum.utils import logging

from ...fx.optimization import ReversibleTransformation, Transformation
from ..modeling_utils import SerializedEmbedding, SerializedLinear, SharedEmbedding


if TYPE_CHECKING:
    from torch.fx import GraphModule, Node

logger = logging.get_logger(__name__)


def node_matches_pattern(pattern, node: "Node"):
    # TODO: validate that.
    name = node.target if isinstance(node.target, str) else node.name.replace("_", ".")
    return re.match(pattern, name)


def parent_module_qualified_name(node: "Node") -> str:
    name = getattr(node, "parent_module_qualified_name", "")
    return name if name != "root" else ""


def parent_module_type(node: "Node") -> Union[str, Type]:
    return getattr(node, "parent_module_type", None)


class AddPoptorchBlockBase(ReversibleTransformation):
    """
    Base class that provide useful methods for inserting poptorch blocks in the model.
    """

    def __init__(
        self, block_name: str, layer_ipu: Union[int, List[int]], name_regex: str, log_insertions: bool = False
    ):
        self.block_name = block_name
        self.layer_ipu = layer_ipu
        self.name_regex = re.compile(name_regex)
        self.log_insertions = log_insertions

    def find_start_nodes(self, graph_module: "GraphModule") -> List["Node"]:
        nodes = []
        prefixes = set()
        for node in graph_module.graph.nodes:
            # If module under which the node was created is root, we use the node name to match.
            name = node.name
            if parent_module_qualified_name(node) != "":
                name = f"{parent_module_qualified_name(node)}.{name}"
            match = re.match(self.name_regex, name)
            if match:
                prefix = match.group(0)
                if prefix not in prefixes:
                    nodes.append(node)
                    prefixes.add(match.group(0))
        return nodes

    def insert_start_block_node(self, graph_module: "GraphModule", node: "Node", block_name: str, ipu_id: int):

        if node.op in ["placeholder", "output"]:
            raise RuntimeError("You cannot insert a start block op before a placeholder or an output.")

        with graph_module.graph.inserting_before(node):
            new_node = graph_module.graph.call_function(poptorch.Block.start, (block_name,), {"ipu_id": ipu_id})
            new_node.parent_module_qualified_name = node.parent_module_qualified_name
        new_node.was_transformed = f"{self.__class__.__name__}"

    def get_ipu_for_index(self, index: Optional[int] = None) -> int:
        if isinstance(self.layer_ipu, list):
            if index is None:
                raise ValueError("You must provide an index when layer_ipu is a list.")
            return self.layer_ipu[index]
        return self.layer_ipu

    def reverse(self, graph_module: "GraphModule") -> "GraphModule":
        for node in graph_module.graph.nodes:
            if getattr(node, "was_transformed", "") == self.__class__.__name__:
                graph_module.graph.erase_node(node)
        return graph_module


class AddPoptorchBlocksInSeries(AddPoptorchBlockBase):
    """
    Adds poptorch blocks in series, to all the layers matching name_regex.
    """

    def transform(self, graph_module: "GraphModule") -> "GraphModule":
        nodes = self.find_start_nodes(graph_module)
        for index, node in enumerate(nodes):
            ipu_id = self.get_ipu_for_index(index)
            name = f"{self.block_name} {index}"
            if self.log_insertions:
                logger.info(f"{name} --> IPU {ipu_id}")
            self.insert_start_block_node(graph_module, node, name, ipu_id)
        return graph_module


class AddPoptorchBlock(AddPoptorchBlockBase):
    """
    Adds a poptorch block before the first node (layer) matching name_regex.
    """

    def transform(self, graph_module: "GraphModule") -> "GraphModule":
        start_nodes = self.find_start_nodes(graph_module)
        if not start_nodes:
            return graph_module
        node = start_nodes[0]
        ipu_id = self.get_ipu_for_index()
        if self.log_insertions:
            logger.info(f"{self.block_name} --> IPU {ipu_id}")
        self.insert_start_block_node(graph_module, node, f"{self.block_name}", ipu_id)
        return graph_module


class AutoParallelizeAutoEncoder(ReversibleTransformation):
    pass


class TupleOutput(Transformation):
    """
    Transforms the output of the model to a tuple, if it is a dict, and does not nothing otherwise.
    """

    def transform(self, graph_module: "GraphModule") -> "GraphModule":
        for node in graph_module.graph.nodes:
            if node.op == "output":
                if isinstance(node.args[0], dict):
                    node.args = (tuple(node.args[0].values()),)
        return graph_module


class ClipValues(Transformation):
    """
    Clips values to make them fall into [min_value, max_value].
    This is useful for fp16 for instance.
    """

    def __init__(
        self,
        min_value: float,
        max_value: float,
        include_targets: Optional[Tuple[Union[str, Callable]]] = None,
        exclude_targets: Optional[Tuple[Union[str, Callable]]] = None,
    ):
        self.min_value = min_value
        self.max_value = max_value
        self.include_targets = include_targets if include_targets is not None else ()
        self.exclude_targets = exclude_targets if exclude_targets is not None else ()

    def _clip_node_args(self, args):
        if isinstance(args, (tuple, list, set)):
            return type(args)(self._clip_node_args(arg) for arg in args)
        elif isinstance(args, dict):
            return {name: self._clip_node_args(arg) for name, arg in args.items()}
        elif isinstance(args, (float, int)):
            return min(max(args, self.min_value), self.max_value)
        else:
            return args

    def transform(self, graph_module: "GraphModule") -> "GraphModule":
        for node in graph_module.graph.nodes:
            if self.include_targets and node.target not in self.include_targets:
                continue
            if node.target in self.exclude_targets:
                continue
            node.args = self._clip_node_args(node.args)
        return graph_module


class ClipValuesSymmetric(ClipValues):
    """
    Clips values to make them fall into [-clip_value, clip_value].
    This is useful for fp16 for instance.
    """

    def __init__(
        self,
        clip_value: float,
        include_targets: Optional[Tuple[Union[str, Callable]]] = None,
        exclude_targets: Optional[Tuple[Union[str, Callable]]] = None,
    ):
        if clip_value < 0:
            raise ValueError(f"The provided clip value must be equal or greater than 0, but here {clip_value}.")
        return super().__init__(
            -clip_value, clip_value, include_targets=include_targets, exclude_targets=exclude_targets
        )


class OutlineAttribute(ReversibleTransformation):
    """
    Adds an attribute to a module. This attribute will be used when comparing operation equivalence in outlining.
    """

    def __init__(self, name_regex: str, value: str):
        self.name_regex = re.compile(name_regex)
        self.value = value

    def transform(self, graph_module: "GraphModule") -> "GraphModule":
        first_match, last_match = None, None
        for node in graph_module.graph.nodes:
            match = re.match(self.name_regex, node.target) if isinstance(node.target, str) else False
            if match:
                if first_match is None:
                    first_match = node
                last_match = node
        if first_match is None:
            raise RuntimeError(f"Could not find any op matching {self.name_regex} to outline.")

        with graph_module.graph.inserting_before(first_match):
            new_node = graph_module.graph.call_function(
                torch.ops.poptorch.set_attribute, ("__outline", "layer", self.value)
            )
            new_node.parent_module_qualified_name = first_match.parent_module_qualified_name
        with graph_module.graph.inserting_after(last_match):
            new_node = graph_module.graph.call_function(torch.ops.poptorch.clear_attribute, ("__outline", "layer"))
            new_node.parent_module_qualified_name = first_match.parent_module_qualified_name
        return graph_module

    def reverse(self, graph_module: "GraphModule") -> "GraphModule":
        has_clear_attribute_to_erase = False
        for node in graph_module.graph.nodes:
            if node.target is torch.ops.poptorch.set_attribute:
                if node.args[2] == self.value:
                    graph_module.graph.erase_node(node)
                    has_clear_attribute_to_erase = True
            if node.target is torch.ops.poptorch.clear_attribute and has_clear_attribute_to_erase:
                graph_module.graph.erase_node(node)
                has_clear_attribute_to_erase = False
        return graph_module


class RecomputationCheckpoint(ReversibleTransformation):
    """
    Annotates the output of a module to be checkpointed instead of recomputed.
    """

    def __init__(
        self, name_regex: str, to_exclude: Optional[str] = None, output_nodes_specs: Dict[str, List[Any]] = None
    ):
        self.name_regex = re.compile(name_regex)
        self.to_exclude = re.compile(to_exclude) if to_exclude is not None else None
        self.output_nodes_specs = None
        if output_nodes_specs is not None:
            self.output_nodes_specs = collections.defaultdict(set)
            for k, v in output_nodes_specs.items():
                self.output_nodes_specs[k] = v

    def find_output_nodes_for_module_name(self, graph_module: "GraphModule", module_qualified_name: str):
        nodes_in_module = set()
        first_match = False
        # Some nodes are created by calling a module that was created before in the model. This means that these nodes
        # parent_module_qualified_name attributes will be "from the past", but we still want to consider them inside
        # the current module, since they are called here.
        modules_from_the_past = set()
        for node in graph_module.graph.nodes:
            name = parent_module_qualified_name(node)
            starts_with_module_qualified_name = name.startswith(module_qualified_name)
            if not first_match and not starts_with_module_qualified_name:
                pass
            elif not first_match and starts_with_module_qualified_name:
                first_match = True
                nodes_in_module.add(node)
            elif first_match and starts_with_module_qualified_name:
                nodes_in_module.add(node)
            elif first_match and name in modules_from_the_past:
                # The module under which this node was created belongs to somewhere before in the hierarchy, but we
                # consider this node to be part of this module since it's being used here.
                nodes_in_module.add(node)
            else:
                break
            modules_from_the_past.add(name)
        output_nodes = [n for n in nodes_in_module if set(n.users.keys()) - nodes_in_module]
        if self.output_nodes_specs:
            output_nodes = [n for n in output_nodes if n.target in self.output_nodes_specs[n.op]]
        return output_nodes

    def transform(self, graph_module: "GraphModule") -> "GraphModule":
        matched_module_names = collections.OrderedDict()
        for node in graph_module.graph.nodes:
            match = re.match(self.name_regex, parent_module_qualified_name(node))
            to_exclude = False
            if self.to_exclude is not None:
                to_exclude = re.match(self.to_exclude, parent_module_qualified_name(node))
            if match and not to_exclude:
                matched_module_names[match.group(0)] = None

        output_nodes = []
        for qualified_name in matched_module_names.keys():
            output_nodes += self.find_output_nodes_for_module_name(graph_module, qualified_name)

        for output in output_nodes:
            with graph_module.graph.inserting_after(output):
                recomputation_node = graph_module.graph.call_function(poptorch.recomputationCheckpoint)
            output.replace_all_uses_with(recomputation_node)
            recomputation_node.args = (output,)

        return graph_module

    def reverse(self, graph_module: "GraphModule") -> "GraphModule":
        for node in graph_module.graph.nodes:
            if node.target == poptorch.recomputationCheckpoint:
                node.replace_all_uses_with(node.args[0])
                graph_module.graph.erase_node(node)
        return graph_module


class AutoCast(ReversibleTransformation):
    def __init__(self, module_types: Union[Type, Set[Type]], name_regex: Optional[str] = None):
        self.module_types = module_types
        if not isinstance(self.module_types, (list, tuple, set)):
            self.module_types = {self.module_types}
        if not isinstance(self.module_types, set):
            self.module_types = set(self.module_types)
        self.name_regex = re.compile(name_regex) if name_regex is not None else None

    def find_start_and_end_nodes(self, graph_module: "GraphModule") -> List[Tuple["Node", "Node"]]:
        start_and_end_nodes = []
        start_node = None
        end_node = None
        for node in graph_module.graph.nodes:
            name = parent_module_qualified_name(node)
            if self.name_regex is not None and not re.match(self.name_regex, name):
                continue
            type_ = parent_module_type(node)
            if type_ in self.module_types and start_node is None:
                start_node = node
            elif type_ in self.module_types:
                end_node = node
            elif start_node is not None:
                start_and_end_nodes.append((start_node, end_node))
                start_node = None
        return start_and_end_nodes

    def transform(self, graph_module: "GraphModule") -> "GraphModule":
        start_and_end_nodes = self.find_start_and_end_nodes(graph_module)
        for start_node, end_node in start_and_end_nodes:
            with graph_module.graph.inserting_before(start_node):
                graph_module.graph.call_function(torch.ops.poptorch.begin_autocast)
            with graph_module.graph.inserting_after(end_node):
                graph_module.graph.call_function(torch.ops.poptorch.restore_autocast)
        return graph_module

    def reverse(self, graph_module: "GraphModule") -> "GraphModule":
        for node in graph_module.graph.nodes:
            if node.target in [torch.ops.poptorch.begin_autocast, torch.ops.poptorch.restore_autocast]:
                graph_module.graph.erase_node(node)
        return graph_module


class VocabEmbeddingToSerializedEmbedding(ReversibleTransformation):
    """
    Transforms the embedding layer matching name_regex to a SerializedEmbedding layer.
    If no name_regex is provided, all the embeddings will be detected, but in any case, only the embedding with the
    biggest number of embeddings will be transformed (this is usually the one containing the vocabulary).
    """

    def __init__(self, name_regex: Optional[str] = None):
        self.name_regex_for_module = re.compile(name_regex) if name_regex else None
        self.name_regex_for_function = re.compile(name_regex.replace(".", "_")) if name_regex else None

    def transform(self, graph_module: "GraphModule") -> "GraphModule":
        embedding_nodes = []
        for node in graph_module.graph.nodes:
            if node.op == "call_module":
                if self.name_regex_for_module is not None and not re.match(self.name_regex_for_module, node.target):
                    continue
                elif not isinstance(graph_module.get_submodule(node.target), torch.nn.Embedding):
                    continue
                embedding_nodes.append(node)
            elif node.op == "call_function":
                if self.name_regex_for_function is not None and not re.match(self.name_regex_for_function, node.name):
                    continue
                elif node.target is not torch.nn.functional.embedding:
                    continue
                embedding_nodes.append(node)

        # We assume the vocab embedding to be the embedding with the maximum number of embeddings.
        if not embedding_nodes:
            raise RuntimeError("Could not find any embedding node")

        def sort_nodes_function(node):
            if node.op == "call_module":
                return graph_module.get_submodule(node.target).num_embeddings
            return node.args[1].shape[1]

        embedding_node = max(embedding_nodes, key=sort_nodes_function)
        if embedding_node.op == "call_function":
            raise NotImplementedError(
                "VocabEmbeddingToSerializedEmbedding does not support torch.nn.functional.embedding yet."
            )

        split = embedding_node.target.rsplit(".", maxsplit=1)
        if len(split) == 1:
            split = [None] + split
        parent_fully_qualified_name, embedding_name = split

        new_embedding = SerializedEmbedding(
            graph_module.get_submodule(embedding_node.target), graph_module.ipu_config.embedding_serialization_factor
        )
        submodule = graph_module
        if parent_fully_qualified_name is not None:
            submodule = graph_module.get_submodule(parent_fully_qualified_name)
        setattr(submodule, embedding_name, new_embedding)
        embedding_node.was_transformed = "VocabEmbeddingToSerializedEmbedding"

        return graph_module

    def reverse(self, graph_module: "GraphModule") -> "GraphModule":
        for node in graph_module.graph.nodes:
            if getattr(node, "was_transformed", "") == "VocabEmbeddingToSerializedEmbedding":
                split = node.target.rsplit(".", maxsplit=1)
                if len(split) == 1:
                    split = [None] + split
                parent_fully_qualified_name, embedding_name = split
                submodule = graph_module
                if parent_fully_qualified_name is not None:
                    submodule = graph_module.get_submodule(parent_fully_qualified_name)
                setattr(
                    submodule,
                    embedding_name,
                    graph_module.get_submodule(node.target).deserialize(),
                )
                break
        return graph_module


class LinearToSerializedLinear(ReversibleTransformation):
    """
    Transforms the linear layers matching name_regex to SerializedLinear layers.
    """

    def __init__(self, name_regex: str):
        self.name_regex = re.compile(name_regex) if name_regex else None

    def transform(self, graph_module: "GraphModule") -> "GraphModule":
        for node in graph_module.graph.nodes:
            if node.op != "call_module":
                continue
            match = re.match(self.name_regex, node.target) if self.name_regex is not None else True
            if match and isinstance(graph_module.get_submodule(node.target), torch.nn.Linear):
                linear = graph_module.get_submodule(node.target)
                serialized_linear = SerializedLinear(
                    graph_module.config.hidden_size,
                    graph_module.config.vocab_size,
                    graph_module.ipu_config.embedding_serialization_factor,
                    bias=linear.bias is not None,
                    mode=poptorch.MatMulSerializationMode.OutputChannels,
                )
                serialized_linear.load_state_dict(linear.state_dict())
                split = node.target.rsplit(".", maxsplit=1)
                if len(split) == 1:
                    split = [None] + split
                parent_fully_qualified_name, linear_name = split
                submodule = graph_module
                if parent_fully_qualified_name is not None:
                    submodule = graph_module.get_submodule(parent_fully_qualified_name)
                setattr(submodule, linear_name, serialized_linear)
        return graph_module

    def reverse(self, graph_module: "GraphModule") -> "GraphModule":
        for node in graph_module.graph.nodes:
            if node.op == "call_module" and isinstance(graph_module.get_submodule(node.target), SerializedLinear):
                graph_module.get_submodule(node.target).__class__ = torch.nn.Linear
        return graph_module


class TieWeights(Transformation):
    def __init__(
        self,
        layer_a: str,
        layer_b: str,
        weight_attribute_name_for_a: Optional[str] = "weight",
        weight_attribute_name_for_b: Optional[str] = "weight",
    ):
        self.layer_a = layer_a
        self.layer_b = layer_b
        self.layer_b = layer_b
        self.weight_attribute_name_for_a = weight_attribute_name_for_a
        self.weight_attribute_name_for_b = weight_attribute_name_for_b

    def transform(self, graph_module: "GraphModule") -> "GraphModule":
        layer_a, layer_b = None, None
        for node in graph_module.graph.nodes:
            if node.op == "call_module":
                if node.target == self.layer_a:
                    layer_a = graph_module.get_submodule(node.target)
                if node.target == self.layer_b:
                    layer_b = graph_module.get_submodule(node.target)

        if layer_a is None or layer_b is None:
            raise ValueError(
                f"Could not find both layers {self.layer_a} and {self.layer_b} to tie their weights together"
            )
        if not hasattr(layer_a, self.weight_attribute_name_for_a):
            raise AttributeError(f"{layer_a} does not have an attribute called {self.weight_attribute_name_for_a}")
        if not hasattr(layer_b, self.weight_attribute_name_for_b):
            raise AttributeError(f"{layer_b} does not have an attribute called {self.weight_attribute_name_for_b}")

        setattr(layer_b, self.weight_attribute_name_for_b, getattr(layer_a, self.weight_attribute_name_for_a))
        return graph_module


class ShareEmbeddingComputation(Transformation):
    def __init__(
        self,
        name_regex: Optional[str] = None,
        allowed_embedding_classes: Union[Tuple[Type], Type] = (torch.nn.Embedding, SerializedEmbedding),
    ):
        self.name_regex = re.compile(name_regex) if name_regex else None
        self.allowed_embedding_classes = allowed_embedding_classes
        if not isinstance(self.allowed_embedding_classes, tuple):
            self.allowed_embedding_classes = (self.allowed_embedding_classes,)

    def _find_nodes_to_move(self, graph_module, embedding_input_node, shared_embedding_node):
        nodes_before_embedding_input_node = set()
        for node in graph_module.graph.nodes:
            if node is shared_embedding_node:
                break
            nodes_before_embedding_input_node.add(node)

        to_visit = [embedding_input_node]
        to_move = set()
        while to_visit:
            node = to_visit.pop(0)
            if node not in nodes_before_embedding_input_node:
                to_move.add(node)
                to_visit += node.all_input_nodes
        ordered_to_move = []
        for node in graph_module.graph.nodes:
            if node in to_move:
                ordered_to_move.append(node)
        return ordered_to_move

    def _move_nodes_after_node(self, graph_module, nodes_to_move, node):
        old_to_new_mapping = {}
        with graph_module.graph.inserting_after(node):
            for n in reversed(nodes_to_move):
                old_to_new_mapping[n] = graph_module.graph.create_node(n.op, n.target, n.args, n.kwargs, n.name)
        return old_to_new_mapping

    def transform(self, graph_module: "GraphModule") -> "GraphModule":
        candidates = collections.defaultdict(list)
        embedding_nodes = collections.defaultdict(list)
        for node in graph_module.graph.nodes:
            if node.op == "call_module":
                if self.name_regex is not None and not re.match(self.name_regex, node.target):
                    continue
                elif not isinstance(graph_module.get_submodule(node.target), self.allowed_embedding_classes):
                    continue
                candidates[node.target].append(node.args[0])
                embedding_nodes[node.target].append(node)

        candidates = {k: v for k, v in candidates.items() if len(v) > 1}
        embedding_nodes = {k: v for k, v in embedding_nodes.items() if k in candidates}

        for target, embedding_input_nodes in candidates.items():
            if len(embedding_input_nodes) > 2:
                raise NotImplementedError("Currently support embedding computation sharing for 2.")
            new_input_nodes = []
            for input_node in reversed(embedding_input_nodes[1:]):
                nodes_to_move = self._find_nodes_to_move(graph_module, input_node, embedding_nodes[target][0])
                old_to_new_mapping = self._move_nodes_after_node(graph_module, nodes_to_move, embedding_input_nodes[0])
                for old_node, new_node in old_to_new_mapping.items():
                    old_node.replace_all_uses_with(new_node)
                    graph_module.graph.erase_node(old_node)
                new_input_nodes.append(old_to_new_mapping[nodes_to_move[-1]])

            graph_module.add_submodule(target, SharedEmbedding(graph_module.get_submodule(target)))
            shared_node = embedding_nodes[target][0]
            shared_node.args = tuple(embedding_input_nodes[0:1] + new_input_nodes)
            with graph_module.graph.inserting_after(shared_node):
                getitem = graph_module.graph.call_function(operator.getitem, ())
                shared_node.replace_all_uses_with(getitem)
                getitem.args = (shared_node, 0)
                with graph_module.graph.inserting_after(getitem):
                    for idx in reversed(range(len(embedding_nodes[target][1:]))):
                        embedding_node = embedding_nodes[target][idx + 1]
                        getitem = graph_module.graph.call_function(operator.getitem, (shared_node, idx + 1))
                        embedding_node.replace_all_uses_with(getitem)
                        graph_module.graph.erase_node(embedding_node)
        return graph_module
