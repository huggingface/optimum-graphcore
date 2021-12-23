# coding=utf-8
# Copyright 2021 HuggingFace Inc.
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
import random
import string
import unittest
from collections import Iterable
from typing import Any, Dict, Optional, Set

import pytest

from optimum.graphcore import IPUConfig
from optimum.graphcore.ipu_configuration import ALLOWED_POD_TYPES


def random_value_from_initial_value(initial_value):
    if isinstance(initial_value, bool):
        return bool(random.randint(0, 1))
    elif isinstance(initial_value, int):
        return random.randint(1, 10)
    elif isinstance(initial_value, float):
        return random.random()
    elif isinstance(initial_value, str):
        return "".join(random.choices(string.ascii_lowercase, k=len(initial_value)))
    elif isinstance(initial_value, Iterable):
        return type(initial_value)([random_value_from_initial_value(x) for x in initial_value])
    elif initial_value is None:
        return None
    else:
        raise TypeError(f"cannot create random value from initial value of type {type(initial_value)}")


def create_pod_specific_attribute(
    initial_value: Any, add_default_value: bool = False, remove_pod_types: Optional[Set[str]] = None
) -> Dict[str, Any]:
    p = random.random()
    if p < 0.5:
        return initial_value
    if remove_pod_types is None:
        remove_pod_types = set()
    values = {k: random_value_from_initial_value(initial_value) for k in set(ALLOWED_POD_TYPES) - remove_pod_types}
    if add_default_value:
        values["default"] = random_value_from_initial_value(initial_value)
    return values


def create_ipu_config(with_default_values: bool = False, remove_pod_types: Optional[Set[str]] = None) -> IPUConfig:
    initial_dict = IPUConfig().to_dict()
    initial_dict = {
        k: create_pod_specific_attribute(v, add_default_value=with_default_values, remove_pod_types=remove_pod_types)
        for k, v in initial_dict.items()
    }
    return IPUConfig.from_dict(initial_dict)


class IPUConfigTester(unittest.TestCase):
    def test_for_pod_type(self):
        ipu_config = create_ipu_config()
        for pod_type in ALLOWED_POD_TYPES:
            pod_type_dict = {k: v[pod_type] if isinstance(v, dict) else v for k, v in ipu_config.to_dict().items()}
            ipu_config_for_pod_type = ipu_config.for_pod_type(pod_type)
            self.assertEqual(pod_type_dict, ipu_config_for_pod_type.to_dict())

    def test_for_pod_type_with_default(self):
        ipu_config = create_ipu_config(with_default_values=True)
        pod_type_dict = {k: v["default"] if isinstance(v, dict) else v for k, v in ipu_config.to_dict().items()}
        ipu_config_for_pod_type = ipu_config.for_pod_type()
        print(pod_type_dict)
        print(ipu_config_for_pod_type.to_dict())
        self.assertEqual(pod_type_dict, ipu_config_for_pod_type.to_dict())

    def test_for_pod_type_with_unallowed_pod_type(self):
        ipu_config = create_ipu_config()
        with pytest.raises(ValueError):
            ipu_config.for_pod_type("blablabla")

    def test_for_pod_type_not_in_config_attribute(self):
        pod_type_to_remove = random.choice(ALLOWED_POD_TYPES)
        ipu_config = create_ipu_config(remove_pod_types={pod_type_to_remove})
        with pytest.raises(KeyError):
            ipu_config.for_pod_type(pod_type_to_remove)

    # TODO: test that later, question: how to access poptorch.Options attributes?
    # def test_to_options(self):
    #     ipu_config = create_ipu_config()
    #     pod_type = random.choice(ALLOWED_POD_TYPES)
    #     options = ipu_config.to_options(pod_type=pod_type)

    # def test_to_options_for_inference(self):
    #     ipu_config = create_ipu_config()
    #     pod_type = random.choice(ALLOWED_POD_TYPES)
    #     options = ipu_config.to_options(for_inference=True, pod_type=pod_type)
    #     self.assertEqual(options.replicationFactor, ipu_config.inference_replication_factor)
    #     self.assertEqual(options.deviceIterations, ipu_config.inference_device_iterations)

    def _test_batch_size_factor(self, for_inference: bool):
        pod_type = random.choice(ALLOWED_POD_TYPES)
        # Case 1: the IPUConfig is not "specialized" and contains values for many pod types.
        ipu_config = create_ipu_config()
        batch_size_factor = ipu_config.batch_size_factor(for_inference=for_inference, pod_type=pod_type)
        ipu_config = ipu_config.for_pod_type(pod_type)
        replication_factor = (
            ipu_config.inference_replication_factor if for_inference else ipu_config.replication_factor
        )
        gradient_accumulation_steps = 1 if for_inference else ipu_config.gradient_accumulation_steps
        device_iterations = ipu_config.inference_device_iterations if for_inference else ipu_config.device_iterations
        self.assertEqual(
            replication_factor * gradient_accumulation_steps * device_iterations,
            batch_size_factor,
        )
        # Case 2: the IPUConfig is specialized, no pod type needs to be specified to compute the batch size factor.
        ipu_config = create_ipu_config().for_pod_type(pod_type)
        batch_size_factor = ipu_config.batch_size_factor(for_inference=for_inference)
        replication_factor = (
            ipu_config.inference_replication_factor if for_inference else ipu_config.replication_factor
        )
        gradient_accumulation_steps = 1 if for_inference else ipu_config.gradient_accumulation_steps
        device_iterations = ipu_config.inference_device_iterations if for_inference else ipu_config.device_iterations
        self.assertEqual(
            replication_factor * gradient_accumulation_steps * device_iterations,
            batch_size_factor,
        )

    def test_batch_size_factor(self):
        self._test_batch_size_factor(False)

    def test_batch_size_factor_for_inference(self):
        self._test_batch_size_factor(True)
