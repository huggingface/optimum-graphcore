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
import collections
import re
from typing import TYPE_CHECKING, List, Optional, Union

import torch

import poptorch
from optimum.utils import logging

from ...fx.optimization import ReversibleTransformation, Transformation
from ..modeling_utils import (
    SerializedEmbedding,
    SerializedLinear,
    get_layer_ipu,
    outline_attribute,
)


if TYPE_CHECKING:
    from torch.fx import GraphModule, Node

logger = logging.get_logger(__name__)


def node_matches_pattern(pattern, node: "Node"):
    # TODO: validate that.
    name = node.target if isinstance(node.target, str) else node.name.replace("_", ".")
    return re.match(pattern, name)


class AddPoptorchBlockBase(ReversibleTransformation):
    def __init__(
        self, block_name: str, layer_ipu: Union[int, List[int]], module_name_regex: str, log_insertions: bool = False
    ):
        self.block_name = block_name
        self.layer_ipu = layer_ipu
        self.module_name_regex = re.compile(module_name_regex) if module_name_regex is not None else None
        self.log_insertions = log_insertions

    def find_start_nodes(self, graph_module: "GraphModule") -> List["Node"]:
        nodes = []
        prefixes = set()
        for node in graph_module.graph.nodes:
            # TODO: how to match the case where node.target is str
            match = re.match(self.module_name_regex, node.target) if isinstance(node.target, str) else None
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

        # def start_block(inputs_to_forward, name, ipu_id):
        #     poptorch.Block.start(name, ipu_id=ipu_id)
        #     if len(inputs_to_forward) != 1:
        #         return inputs_to_forward
        #     return inputs_to_forward[0]

        # with graph_module.graph.inserting_before(node):
        #     new_node = graph_module.graph.call_function(start_block, (node.args, block_name, ipu_id))
        #     if node.op != "get_attr":
        #         new_args = []
        #         if len(node.args) > 1:
        #             for idx, _ in enumerate(node.args):
        #                 new_args.append(graph_module.graph.call_function(operator.getitem, (new_node, idx)))
        #         elif node.args:
        #             new_args.append(new_node)
        #         else:
        #             raise NotImplementedError(
        #                 f"Inserting start block op before a {node.op} that does not take any argument is not supported."
        #             )
        #         node.args = tuple(new_args)

        #     new_node.was_transformed = f"{self.__class__.__name__}"
        #     new_node.orig_node = node

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
    def transform(self, graph_module: "GraphModule") -> "GraphModule":
        for node in graph_module.graph.nodes:
            if node.op == "output":
                if isinstance(node.args[0], dict):
                    node.args = (tuple(node.args[0].values()),)
        return graph_module


class ClipValues(Transformation):
    def __init__(self, clip_value: float):
        self.clip_value = clip_value

    def _clip_node_args(self, args):
        if isinstance(args, (tuple, list, set)):
            return type(args)(self._clip_node_args(arg) for arg in args)
        elif isinstance(args, dict):
            return {name: self._clip_node_args(arg) for name, arg in args.items()}
        elif isinstance(args, (float, int)):
            return min(max(args, -self.clip_value), self.clip_value)
        else:
            return args

    def transform(self, graph_module: "GraphModule") -> "GraphModule":
        for node in graph_module.graph.nodes:
            if node.op == "call_method" and node.target == "view":
                continue
            node.args = self._clip_node_args(node.args)
        return graph_module


class OutlineAttribute(ReversibleTransformation):
    def __init__(self, name_regex: str, value: str):
        self.name_regex = re.compile(name_regex)
        self.value = value

    def transform(self, graph_module: "GraphModule") -> "GraphModule":
        first_match, last_match = None, None
        for node in graph_module.graph.nodes:
            # TODO: how to match the case where node.target is str
            match = re.match(self.name_regex, node.target) if isinstance(node.target, str) else False
            if match:
                if first_match is None:
                    first_match = node
                last_match = node
        if first_match is None:
            raise RuntimeError(f"Could not find any op matching {self.name_regex} to outline.")

        with graph_module.graph.inserting_before(first_match):
            new_node = graph_module.graph.call_function(torch.ops.poptorch.set_attribute, ("__outline", "layer", self.value))
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
    def __init__(self, name_regex: str, to_exclude: Optional[str] = None):
        self.name_regex = re.compile(name_regex)
        self.to_exclude = re.compile(to_exclude) if to_exclude is not None else None

    def find_output_nodes_for_module_name(self, graph_module: "GraphModule", module_qualified_name: str):
        nodes_in_module = set()
        first_match = False
        for n in graph_module.graph.nodes:
            starts_with_module_qualified_name = getattr(n, "parent_module_qualified_name", "").startswith(module_qualified_name)
            print(getattr(n, "parent_module_qualified_name", ""))
            print(starts_with_module_qualified_name)
            if not first_match and not starts_with_module_qualified_name:
                continue
            elif not first_match and starts_with_module_qualified_name:
                first_match = True
                nodes_in_module.add(n)
            elif first_match and starts_with_module_qualified_name:
                nodes_in_module.add(n)
            else:
                break
        nodes_in_module = {n for n in nodes_in_module if set(n.users.keys()) & nodes_in_module}
        return [n for n in nodes_in_module if set(n.users.keys()) - nodes_in_module]

    def transform(self, graph_module: "GraphModule") -> "GraphModule":
        matched_module_names = collections.OrderedDict()
        for node in graph_module.graph.nodes:
            match = re.match(self.name_regex, getattr(node, "parent_module_qualified_name", ""))
            to_exclude = False
            if self.to_exclude is not None:
                to_exclude = re.match(self.to_exclude, getattr(node, "parent_module_qualified_name", ""))
            if match and not to_exclude:
                matched_module_names[match.group(0)] = None

        output_nodes = []
        for qualified_name in matched_module_names.keys():
            print(qualified_name)
            output_nodes += self.find_output_nodes_for_module_name(graph_module, qualified_name)

        for output in output_nodes:
            with graph_module.graph.inserting_after(output):
                print("output", output)
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


class VocabEmbeddingToSerializedEmbedding(ReversibleTransformation):
    def __init__(self, name_regex: Optional[str] = None):
        self.name_regex = re.compile(name_regex) if name_regex else None

    def transform(self, graph_module: "GraphModule") -> "GraphModule":
        embedding_nodes = []
        for node in graph_module.graph.nodes:
            if node.op != "call_module":
                continue
            match = re.match(self.name_regex, node.target) if self.name_regex is not None else True
            if match and isinstance(graph_module.get_submodule(node.target), torch.nn.Embedding):
                embedding_nodes.append(node)

        # We assume the vocab embedding to be the embedding with the maximum number of embeddings.
        if not embedding_nodes:
            raise RuntimeError("Could not find any embedding node")

        embedding_node = max(embedding_nodes, key=lambda node: graph_module.get_submodule(node.target).num_embeddings)
        parent_fully_qualified_name, embedding_name = embedding_node.target.rsplit(".", maxsplit=1)
        new_embedding = SerializedEmbedding(
            graph_module.get_submodule(embedding_node.target), graph_module.ipu_config.embedding_serialization_factor
        )
        setattr(graph_module.get_submodule(parent_fully_qualified_name), embedding_name, new_embedding)
        embedding_node.was_transformed = "VocabEmbeddingToSerializedEmbedding"

        return graph_module

    def reverse(self, graph_module: "GraphModule") -> "GraphModule":
        for node in graph_module.graph.nodes:
            if getattr(node, "was_transformed", "") == "VocabEmbeddingToSerializedEmbedding":
                parent_fully_qualified_name, embedding_name = node.target.rsplit(".", maxsplit=1)
                setattr(
                    graph_module.get_submodule(parent_fully_qualified_name),
                    embedding_name,
                    graph_module.get_submodule(node.target).deserialize(),
                )
                break
        return graph_module


class LinearToSerializedLinear(ReversibleTransformation):
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
                parent_fully_qualified_name, linear_name = node.target.rsplit(".", maxsplit=1)
                setattr(graph_module.get_submodule(parent_fully_qualified_name), linear_name, serialized_linear)
        graph_module.tie_weights()
        return graph_module

    def reverse(self, graph_module: "GraphModule") -> "GraphModule":
        for node in graph_module.graph.nodes:
            if node.op == "call_module" and isinstance(graph_module.get_submodule(node.target), SerializedLinear):
                graph_module.get_submodule(node.target).__class__ = torch.nn.Linear
        return graph_module
