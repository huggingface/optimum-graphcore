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
"""Defines the class that manages which transformations to apply according to some optimization level."""

import copy
import functools
import operator
from typing import Iterator, List, Tuple, Union

import torch

from ...fx.optimization import (
    ChangeTrueDivToMulByInverse,
    MergeLinears,
    ReversibleTransformation,
    Transformation,
    compose,
)
from .transformations import ClipValues, ClipValuesSymmetric


class TransformationManager:
    def __init__(self, *transformations: Tuple[int, "Transformation"]):
        self._signatures = {
            0: set(),
            1: set(),
            2: set(),
        }
        self._transformations = {
            0: [],
            1: [],
            2: [],
        }
        self.register(*transformations)

    def without(self, *args: Transformation) -> "TransformationManager":
        clone = copy.deepcopy(self)
        clone.unregister(*args)
        return clone

    def register(self, *transformations: Tuple[int, Transformation]):
        for (opt_level, t) in transformations:
            for k, signatures in self._signatures.items():
                if t.signature in signatures:
                    raise RuntimeError(
                        f"The transformation {t} has already been registered at optimization level = {k}."
                    )
            self._signatures[opt_level].add(t.signature)
            self._transformations[opt_level].append(t)

    def unregister(self, *transformations: Transformation):
        for transformation_to_unregister in transformations:
            level = None
            sig = transformation_to_unregister.signature
            for opt_level, signatures in self._signatures.items():
                if sig in signatures:
                    level = opt_level
                    signatures.remove(sig)
            if level is not None:
                idx_to_pop = None
                for idx, t in enumerate(self._transformations[level]):
                    if t.signature == sig:
                        idx_to_pop = idx
                        break
                self._transformations[level].pop(idx_to_pop)

    def _check_optimization_level(self, optimization_level):
        if optimization_level not in [0, 1, 2]:
            raise ValueError(f"The optimization level must be either 0, 1 or 2, but {optimization_level} was given.")

    def _get_transformations(
        self, optimization_level: int, as_list: bool = False
    ) -> Union[Iterator[Transformation], List[Transformation]]:
        self._check_optimization_level(optimization_level)
        iterator = functools.reduce(
            lambda x, y: x + y, (self._transformations[i] for i in range(optimization_level + 1)), []
        )
        return iterator if as_list is False else list(iterator)

    def get_transformations(self, optimization_level: int) -> List[Transformation]:
        return self._get_transformations(optimization_level, as_list=True)

    def get_non_reversible_transformations(self, optimization_level: int) -> List[Transformation]:
        return [
            t for t in self._get_transformations(optimization_level) if not isinstance(t, ReversibleTransformation)
        ]

    def get_reversible_transformations(self, optimization_level: int) -> List[ReversibleTransformation]:
        return [t for t in self._get_transformations(optimization_level) if isinstance(t, ReversibleTransformation)]

    def _compose_transformations(
        self, optimization_level: int, transformations: List[Transformation]
    ) -> Transformation:
        return compose(*transformations) if transformations else lambda x: x

    def compose_transformations(self, optimization_level: int) -> Transformation:
        return self._compose_transformations(optimization_level, self.get_transformations(optimization_level))

    def compose_non_reversible_transformations(self, optimization_level: int) -> Transformation:
        return self._compose_transformations(
            optimization_level, self.get_non_reversible_transformations(optimization_level)
        )

    def compose_reversible_transformations(self, optimization_level: int) -> ReversibleTransformation:
        return self._compose_transformations(
            optimization_level, self.get_reversible_transformations(optimization_level)
        )


DEFAULT_TRANSFORMATION_MANAGER = TransformationManager(
    (1, ChangeTrueDivToMulByInverse()),
    (1, MergeLinears()),
    # (1, FuseBiasInLinear()),
    # Those change the computation, but are actually needed for fp16 stability.
    (0, ClipValuesSymmetric(1e4, include_targets=(torch.add, torch.mul, operator.add, operator.mul))),
    (0, ClipValues(1e-4, float("inf"), include_targets=(torch.nn.LayerNorm,))),
)
