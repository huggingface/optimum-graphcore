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
import copy
import random
import string
import unittest
from collections import Iterable
from typing import Any, Dict, Optional, Set

import pytest

from optimum.graphcore import IPUConfig
from optimum.graphcore.ipu_configuration import ALLOWED_POD_TYPES
from optimum.graphcore.modeling_utils import get_layer_ipu
from poptorch import OutputMode


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
    allowed_output_modes = ["all", "sum", "final"]
    if isinstance(initial_dict["output_mode"], dict):
        initial_dict["output_mode"] = {k: random.choice(allowed_output_modes) for k in initial_dict["output_mode"]}
    else:
        initial_dict["output_mode"] = random.choice(allowed_output_modes)
    # Setting this setting to False as it is currently not supported and will throw an error.
    initial_dict["execute_encoder_on_cpu_for_generation"] = False
    # Edge case where replication_factor=1 and replicated_tensor_sharding=True which will get overriden to
    # replicated_tensor_sharding=False.
    if isinstance(initial_dict["replication_factor"], dict) and isinstance(
        initial_dict["replicated_tensor_sharding"], dict
    ):
        for pod_type in initial_dict["replication_factor"].keys():
            if initial_dict["replication_factor"][pod_type] == 1:
                initial_dict["replicated_tensor_sharding"][pod_type] = False
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

    def _test_to_options(self, for_inference: bool):
        def make_poptorch_options_comparable_to_ipu_config(options_dict):
            options_dict = copy.deepcopy(options_dict)
            # mapping specifies how to transform an options dict entry to something that can be compared to an IPUConfig.
            # It maps a string to either another string or a function:
            #   1. String -> string: it specifies how to align values for attribute names that differ between
            #       poptorch.Options and IPUConfig.
            #   2. String -> function: the function must return a tuple of type (str, Any), the first element being the
            #       name of the attribute to update, and the second one being the actual value to use for the update.
            mapping = {
                "gradient_accumulation": "gradient_accumulation_steps",
                # Seed?
                "location_optimizer.onChip": lambda d: (
                    "optimizer_state_offchip",
                    not d["location_optimizer"]["onChip"],
                ),
                "location_optimizer.useReplicatedTensorSharding": lambda d: (
                    "replicated_tensor_sharding",
                    bool(d["location_optimizer"]["useReplicatedTensorSharding"]),
                ),
                # This works for matmul_proportion because the randomly generated IPUConfig will set matmul_proportion
                # to some float value, meaning that this value will be replicated for all the IPUs, so there is only
                # one value in d["available_memory_proportion"].values(), which should be equal to
                # ipu_config.matmul_proportion
                "available_memory_proportion": lambda d: (
                    "matmul_proportion",
                    set(d["available_memory_proportion"].values()).pop(),
                ),
                "cachePath": "executable_cache_dir",
                "output_mode": lambda d: ("output_mode", OutputMode(d["output_mode"]).name.lower()),
            }
            for k, v in mapping.items():
                try:
                    if isinstance(v, str):
                        value = options_dict
                        keys = k.split(".")
                        for key in keys:
                            value = value[key]
                        options_dict[v] = value
                    else:
                        key_name, value = v(options_dict)
                        options_dict[key_name] = value
                except KeyError:
                    continue

            return options_dict

        def intersection_of_dicts(d1, d2):
            d1 = copy.deepcopy(d1)
            d2 = copy.deepcopy(d2)
            d1 = {k: v for k, v in d1.items() if k in d2}
            d2 = {k: v for k, v in d2.items() if k in d1}
            return d1, d2

        pod_type = random.choice(ALLOWED_POD_TYPES)
        # Case 1: the IPUConfig is not "specialized" and contains values for many pod types.
        ipu_config = create_ipu_config()
        options = ipu_config.to_options(for_inference=for_inference, pod_type=pod_type)
        ipu_config_dict = ipu_config.for_pod_type(pod_type).to_dict()
        if for_inference:
            ipu_config_dict["replication_factor"] = ipu_config_dict["inference_replication_factor"]
            ipu_config_dict["device_iterations"] = ipu_config_dict["inference_device_iterations"]
            ipu_config_dict["gradient_accumulation_steps"] = 1
            ipu_config_dict["output_mode"] = "all"
        ipu_config_dict, options_dict = intersection_of_dicts(
            ipu_config_dict, make_poptorch_options_comparable_to_ipu_config(options.toDict())
        )
        self.assertEqual(ipu_config_dict, options_dict)
        # Case 2: the IPUConfig is specialized, no pod type needs to be specified to create the poptorch.Options.
        ipu_config = create_ipu_config().for_pod_type(pod_type)
        options = ipu_config.to_options(for_inference=for_inference)
        ipu_config_dict = ipu_config.to_dict()
        if for_inference:
            ipu_config_dict["replication_factor"] = ipu_config_dict["inference_replication_factor"]
            ipu_config_dict["device_iterations"] = ipu_config_dict["inference_device_iterations"]
            ipu_config_dict["gradient_accumulation_steps"] = 1
            ipu_config_dict["output_mode"] = "all"
        ipu_config_dict, options_dict = intersection_of_dicts(
            ipu_config_dict, make_poptorch_options_comparable_to_ipu_config(options.toDict())
        )
        self.assertEqual(ipu_config_dict, options_dict)

    def test_to_options(self):
        return self._test_to_options(False)

    def test_to_options_for_inference(self):
        return self._test_to_options(True)

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

    def test_layers_per_ipu(self):
        # ipus_per_replica inferred from layers_per_ipu
        ipu_config = IPUConfig(layers_per_ipu=[1, 2])
        self.assertEqual(ipu_config.ipus_per_replica, 2)

        # get_layer_ipu with specific layers_per_ipu
        ipu_config = IPUConfig(layers_per_ipu=[1, 2])
        layer_ipu = get_layer_ipu(ipu_config, 3)
        self.assertEqual(layer_ipu, [0, 1, 1])

        # No target number of layers specified
        ipu_config = IPUConfig(layers_per_ipu=[1, 2])
        layer_ipu = get_layer_ipu(ipu_config)
        self.assertEqual(layer_ipu, [0, 1, 1])

        # Raises exception if number of layers is too few
        ipu_config = IPUConfig(layers_per_ipu=[1, 2])
        with pytest.raises(
            ValueError, match="layers_per_ipu does not define the correct number of layers for the current model"
        ):
            layer_ipu = get_layer_ipu(ipu_config, 2)

        # Raises exception if number of layers is too many
        ipu_config = IPUConfig(layers_per_ipu=[1, 2])
        with pytest.raises(
            ValueError, match="layers_per_ipu does not define the correct number of layers for the current model"
        ):
            layer_ipu = get_layer_ipu(ipu_config, 4)

        # layers_per_ipu and ipus_per_replica mismatch raises
        ipu_config = IPUConfig(layers_per_ipu=[1, 2], ipus_per_replica=4)
        with pytest.raises(
            ValueError,
            match=r"layers_per_ipu has non-default value set, but its length does not match ipus_per_replica",
        ):
            layer_ipu = get_layer_ipu(ipu_config, 3)
        ipu_config = IPUConfig(layers_per_ipu=[1, -1], ipus_per_replica=4)
        with pytest.raises(
            ValueError,
            match=r"layers_per_ipu has non-default value set, but its length does not match ipus_per_replica",
        ):
            layer_ipu = get_layer_ipu(ipu_config, 3)

        # Default config everything should be on single IPU
        ipu_config = IPUConfig()
        layer_ipu = get_layer_ipu(ipu_config, 9)
        self.assertEqual(layer_ipu, [0] * 9)

        # Spread layers across 2 IPUs
        ipu_config = IPUConfig(ipus_per_replica=2)
        layer_ipu = get_layer_ipu(ipu_config, 6)  # even
        self.assertEqual(layer_ipu, [0, 0, 0, 1, 1, 1])
        layer_ipu = get_layer_ipu(ipu_config, 7)  # odd
        self.assertEqual(layer_ipu, [0, 0, 0, 1, 1, 1, 1])

        # Wild card
        ipu_config = IPUConfig(layers_per_ipu=[2, -1])
        layer_ipu = get_layer_ipu(ipu_config, 6)
        self.assertEqual(layer_ipu, [0, 0, 1, 1, 1, 1])
        ipu_config = IPUConfig(layers_per_ipu=[-1, 2])
        layer_ipu = get_layer_ipu(ipu_config, 6)
        self.assertEqual(layer_ipu, [0, 0, 0, 0, 1, 1])
        ipu_config = IPUConfig(layers_per_ipu=[-1, 2, -1, 2])
        layer_ipu = get_layer_ipu(ipu_config, 7)
        self.assertEqual(layer_ipu, [0, 1, 1, 2, 2, 3, 3])

        # Invalid values
        ipu_config = IPUConfig(layers_per_ipu=[2, -2])
        with pytest.raises(ValueError, match=r"Invalid values in layers_per_ipu"):
            layer_ipu = get_layer_ipu(ipu_config, 6)
        ipu_config = IPUConfig(ipus_per_replica=0)
        with pytest.raises(ValueError, match=r"Invalid value for ipus_per_replica"):
            layer_ipu = get_layer_ipu(ipu_config, 6)
