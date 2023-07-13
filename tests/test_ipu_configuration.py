# coding=utf-8
# Copyright 2021 HuggingFace Inc.
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
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
import re
import unittest
from typing import Any, Dict

import pytest
from parameterized import parameterized
from poptorch import OutputMode

from optimum.graphcore import IPUConfig
from optimum.graphcore.ipu_configuration import IncompatibleIPUConfigError
from optimum.graphcore.modeling_utils import get_layer_ipu, split_encoder_decoder_ipu_config


def create_ipu_config() -> IPUConfig:
    initial_dict = IPUConfig().to_dict()
    allowed_output_modes = ["all", "sum", "final"]
    initial_dict["output_mode"] = random.choice(allowed_output_modes)
    # Setting this setting to False as it is currently not supported and will throw an error.
    initial_dict["execute_encoder_on_cpu_for_generation"] = False
    return IPUConfig.from_dict(initial_dict)


def create_mode_ipu_config(test_attributes: Dict[str, Any], mode):
    helper_config = IPUConfig()
    helper_config.mode = mode
    ipu_config = IPUConfig(
        **{helper_config._get_managed_attr_mode_name(attr): value for attr, value in test_attributes.items()}
    )
    ipu_config.mode = mode
    return ipu_config


class IPUConfigTester(unittest.TestCase):
    def test_attribute_default_values(self):
        # ipus_per_replica should equal to len(layers_per_ipu) if not provided
        ipu_config = IPUConfig(layers_per_ipu=[1, 2, 3])
        self.assertEqual(ipu_config.ipus_per_replica, 3)

        # inference_matmul_proportion not specified but matmul_proportion is
        ipu_config = IPUConfig(
            layers_per_ipu=[1, 2, 3, 4],
            matmul_proportion=[0.1, 0.2, 0.3, 0.4],
        )
        self.assertEqual(ipu_config.inference_matmul_proportion, [0.1, 0.2, 0.3, 0.4])

        # inference_matmul_proportion not specified but inference_layers_per_ipu != len(matmul_proportion)
        ipu_config = IPUConfig(
            layers_per_ipu=[1, 2, 3, 4], matmul_proportion=[0.1, 0.2, 0.3, 0.4], inference_layers_per_ipu=[3, 7]
        )
        self.assertEqual(ipu_config.inference_matmul_proportion, 0.2)

        # inference_ipus_per_replica not specified but ipus_per_replica is
        ipu_config = IPUConfig(
            ipus_per_replica=10,
        )
        self.assertEqual(ipu_config.inference_ipus_per_replica, 10)

        # inference_ipus_per_replica not specified but inference_layers_per_ipu is
        ipu_config = IPUConfig(ipus_per_replica=10, inference_layers_per_ipu=[1, 2, 3])
        self.assertEqual(ipu_config.inference_ipus_per_replica, 3)

        # If the user has not provided either {projection/embedding}_serialization_factor
        # or serialized_{projection/embedding}_splits_per_ipu, {projection/embedding}_serialization_factor}
        # should default to 1
        ipu_config = IPUConfig()
        self.assertEqual(ipu_config.projection_serialization_factor, 1)
        self.assertEqual(ipu_config.inference_projection_serialization_factor, 1)
        self.assertEqual(ipu_config.embedding_serialization_factor, 1)
        self.assertEqual(ipu_config.inference_embedding_serialization_factor, 1)

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

        ipu_config = create_ipu_config()
        options = ipu_config.to_options(for_inference=for_inference)
        ipu_config_dict = ipu_config.to_dict()
        if for_inference:
            ipu_config_dict["replication_factor"] = ipu_config_dict["inference_replication_factor"]
            ipu_config_dict["device_iterations"] = ipu_config_dict["inference_device_iterations"]
            ipu_config_dict["matmul_proportion"] = ipu_config_dict["inference_matmul_proportion"]
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
        # Case 1: the IPUConfig is not "specialized" and contains values for many pod types.
        ipu_config = create_ipu_config()
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
        # Case 2: the IPUConfig is specialized, no pod type needs to be specified to compute the batch size factor.
        ipu_config = create_ipu_config()
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
            IncompatibleIPUConfigError,
            match="layers_per_ipu does not define the correct number of layers for the current model",
        ):
            layer_ipu = get_layer_ipu(ipu_config, 2)

        # Raises exception if number of layers is too many
        ipu_config = IPUConfig(layers_per_ipu=[1, 2])
        with pytest.raises(
            IncompatibleIPUConfigError,
            match="layers_per_ipu does not define the correct number of layers for the current model",
        ):
            layer_ipu = get_layer_ipu(ipu_config, 4)

        # layers_per_ipu and ipus_per_replica mismatch raises
        ipu_config = IPUConfig(layers_per_ipu=[1, -1], ipus_per_replica=4)
        with pytest.raises(
            IncompatibleIPUConfigError,
            match=r"layers_per_ipu has a non-default value set, but its length does not match ipus_per_replica",
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

    def test_split_encoder_decoder_ipu_config(self):
        # Test splitting two IPUs
        ipu_config = IPUConfig(layers_per_ipu=[1, 2])
        e_ipu_config, d_ipu_config = split_encoder_decoder_ipu_config(ipu_config, 1, 2)
        self.assertEqual(e_ipu_config.layers_per_ipu, [1])
        self.assertEqual(e_ipu_config.ipus_per_replica, 1)
        self.assertEqual(d_ipu_config.layers_per_ipu, [2])
        self.assertEqual(d_ipu_config.ipus_per_replica, 1)

        # Test splitting matmul_proportion
        ipu_config = IPUConfig(layers_per_ipu=[2, 2, 2, 2], matmul_proportion=[0.1, 0.2, 0.3, 0.4])
        e_ipu_config, d_ipu_config = split_encoder_decoder_ipu_config(ipu_config, 4, 4)
        self.assertEqual(e_ipu_config.matmul_proportion, [0.1, 0.2])
        self.assertEqual(d_ipu_config.matmul_proportion, [0.3, 0.4])

        # For generation using encoder decoder models and layers `SplitProjection`
        # and `SerializedEmbedding` placed on different IPUs, cannot
        # have serialized_{linear/embedding}_splits_per_ipu present in
        # both the encoder and decoder
        with pytest.raises(ValueError, match="must have all splits placed on the"):
            failing_ipu_config = IPUConfig(
                layers_per_ipu=[0, 2, 2, 0], serialized_projection_splits_per_ipu=[0, 2, 2, 0]
            )
            split_encoder_decoder_ipu_config(failing_ipu_config, 2, 2)

        with pytest.raises(ValueError, match="must have all splits placed on the"):
            failing_ipu_config = IPUConfig(
                layers_per_ipu=[0, 2, 2, 0], serialized_embedding_splits_per_ipu=[0, 2, 2, 0]
            )
            split_encoder_decoder_ipu_config(failing_ipu_config, 2, 2)

        # Test that all the other values from the original ipu_config are intact
        ipu_config = ipu_config.to_dict()
        e_ipu_config = e_ipu_config.to_dict()
        d_ipu_config = d_ipu_config.to_dict()
        skip = {"layers_per_ipu", "ipus_per_replica", "matmul_proportion"}
        for k in ipu_config.keys():
            if re.search("|".join(skip), k):
                continue
            self.assertEqual(ipu_config[k], e_ipu_config[k], k)
            self.assertEqual(ipu_config[k], d_ipu_config[k], k)

        # Test that wildcards work
        ipu_config = IPUConfig(layers_per_ipu=[-1, -1])
        e_ipu_config, d_ipu_config = split_encoder_decoder_ipu_config(ipu_config, 2, 3)
        self.assertEqual(e_ipu_config.layers_per_ipu, [2])
        self.assertEqual(d_ipu_config.layers_per_ipu, [3])

        # Wrong number of layers should raise an exception
        ipu_config = IPUConfig(layers_per_ipu=[1, 2])
        with pytest.raises(
            IncompatibleIPUConfigError,
            match=r"layers_per_ipu does not define the correct number of layers for the current model",
        ):
            e_ipu_config, d_ipu_config = split_encoder_decoder_ipu_config(ipu_config, 2, 2)

        # Encoder and decoder layers defined on same IPU should raise an exception
        ipu_config = IPUConfig(layers_per_ipu=[4, 3])
        with pytest.raises(
            IncompatibleIPUConfigError, match=r"Unable to find a valid split of ipu_config.layers_per_ipu"
        ):
            e_ipu_config, d_ipu_config = split_encoder_decoder_ipu_config(ipu_config, 3, 4)

        # If ipu_config only has 1 IPU then it should raise and exception
        ipu_config = IPUConfig(layers_per_ipu=[4])
        with pytest.raises(
            IncompatibleIPUConfigError,
            match=r"Need ipus_per_replica to be at least 2 to split ipu_config into encoder and decoder configs",
        ):
            e_ipu_config, d_ipu_config = split_encoder_decoder_ipu_config(ipu_config, 2, 2)

        # Handle empty IPU in last stage of encoder
        ipu_config = IPUConfig(layers_per_ipu=[1, 2, 3, 0, 5, 6, 7, 8])
        e_ipu_config, d_ipu_config = split_encoder_decoder_ipu_config(ipu_config, 6, 26)
        self.assertEqual(e_ipu_config.layers_per_ipu, [1, 2, 3, 0])
        self.assertEqual(e_ipu_config.ipus_per_replica, 4)
        self.assertEqual(d_ipu_config.layers_per_ipu, [5, 6, 7, 8])
        self.assertEqual(d_ipu_config.ipus_per_replica, 4)

        # Split where first IPU has 0 layers
        ipu_config = IPUConfig(layers_per_ipu=[0, 6, 0, 6])
        e_ipu_config, d_ipu_config = split_encoder_decoder_ipu_config(ipu_config, 6, 6)
        self.assertEqual(e_ipu_config.layers_per_ipu, [0, 6])
        self.assertEqual(e_ipu_config.ipus_per_replica, 2)
        self.assertEqual(d_ipu_config.layers_per_ipu, [0, 6])
        self.assertEqual(d_ipu_config.ipus_per_replica, 2)

        # Split where there are many zeros
        ipu_config = IPUConfig(layers_per_ipu=[3, 0, 3, 0, 0, 7])
        e_ipu_config, d_ipu_config = split_encoder_decoder_ipu_config(ipu_config, 6, 7)
        self.assertEqual(e_ipu_config.layers_per_ipu, [3, 0, 3, 0])
        self.assertEqual(e_ipu_config.ipus_per_replica, 4)
        self.assertEqual(d_ipu_config.layers_per_ipu, [0, 7])
        self.assertEqual(d_ipu_config.ipus_per_replica, 2)


class IPUConfigExecutionModeTester(unittest.TestCase):
    def test_mode_set(self):
        # Default mode should be training
        config = IPUConfig()
        self.assertEqual(config.mode, "training")

        # Set with convenience methods
        config.train()
        self.assertEqual(config.mode, "training")
        config.eval()
        self.assertEqual(config.mode, "inference")

        # Set with invalid mode value
        with pytest.raises(ValueError, match="`IPUConfig` mode can only take"):
            config.mode = "invalid"

    @parameterized.expand(
        (
            (attr, test_value, mode)
            for mode in IPUConfig.modes
            for attr, test_value in {
                "layers_per_ipu": [1, 2, 3, 4],
                "matmul_proportion": 0.6,
                "ipus_per_replica": 4,
                "serialized_projection_splits_per_ipu": [4],
                "serialized_embedding_splits_per_ipu": [4],
                "projection_serialization_factor": 4,
                "embedding_serialization_factor": 4,
            }.items()
        )
    )
    def test_attr_mode_retrieval(self, attr, value, mode):
        ipu_config = create_mode_ipu_config({attr: value}, mode)
        self.assertEqual(getattr(ipu_config, f"_{attr}"), value)

    @parameterized.expand((mode for mode in IPUConfig.modes))
    def test_layers_per_ipu_wildcard(self, mode):
        ipu_config = create_mode_ipu_config({"ipus_per_replica": 4, "layers_per_ipu": [-1]}, mode=mode)
        layer_ipu = get_layer_ipu(ipu_config, 8)
        self.assertEqual(ipu_config._ipus_per_replica, 4)
        self.assertEqual(layer_ipu, [0, 0, 1, 1, 2, 2, 3, 3])

    @parameterized.expand((mode for mode in IPUConfig.modes))
    def test_split_encoder_decoder_ipu_config(self, mode):
        ipu_config = create_mode_ipu_config(
            {
                "layers_per_ipu": [1, 2, 3, 4],
                "matmul_proportion": [0.1, 0.2, 0.3, 0.4],
                "ipus_per_replica": 4,
                "serialized_projection_splits_per_ipu": [0, 0, 2, 2],
                "serialized_embedding_splits_per_ipu": [2, 2, 0, 0],
            },
            mode,
        )

        e_ipu_config, d_ipu_config = split_encoder_decoder_ipu_config(ipu_config, 3, 7)

        self.assertEqual(e_ipu_config._layers_per_ipu, [1, 2])
        self.assertEqual(e_ipu_config._ipus_per_replica, 2)
        self.assertEqual(e_ipu_config._matmul_proportion, [0.1, 0.2])
        self.assertEqual(e_ipu_config._serialized_projection_splits_per_ipu, None)
        self.assertEqual(e_ipu_config._serialized_embedding_splits_per_ipu, [2, 2])

        self.assertEqual(d_ipu_config._layers_per_ipu, [3, 4])
        self.assertEqual(d_ipu_config._ipus_per_replica, 2)
        self.assertEqual(d_ipu_config._matmul_proportion, [0.3, 0.4])
        self.assertEqual(d_ipu_config._serialized_projection_splits_per_ipu, [2, 2])
        self.assertEqual(d_ipu_config._serialized_embedding_splits_per_ipu, None)


class IPUConfigAttributeValidationTester(unittest.TestCase):
    @parameterized.expand(
        (
            (attr, test_value, mode)
            for mode in IPUConfig.modes
            for attr, test_value in {
                "layers_per_ipu": [-1],
                "matmul_proportion": [0.2, 0.3, 0.0, 0.4],
                "replication_factor": 1,
                "gradient_accumulation_steps": 1,
                "ipus_per_replica": 1,
                "embedding_serialization_factor": 1,
                "device_iterations": 1,
                "projection_serialization_factor": 1,
            }.items()
        )
    )
    def test_contents_geq_value_validator(self, attr, value, mode):
        ipu_config = IPUConfig()
        ipu_config.mode = mode
        attr = ipu_config._get_managed_attr_mode_name(attr)
        setattr(ipu_config, attr, value)

        with pytest.raises(ValueError, match=f"`IPUConfig` attribute `{attr}` must .* >="):
            if isinstance(value, list):
                argmin = min(enumerate(value), key=lambda x: x[1])[0]
                value[argmin] -= 1
            else:
                value -= 1
            setattr(ipu_config, attr, value)

    def test_output_mode_validator(self):
        with pytest.raises(ValueError):
            IPUConfig(output_mode="invalid")
        # should not raise
        allowed_output_modes = ("all", "sum", "final", "default")
        for output_mode in allowed_output_modes:
            with self.subTest(output_mode=output_mode):
                IPUConfig(output_mode=output_mode)

    @parameterized.expand(
        (serialized_layer, mode) for mode in IPUConfig.modes for serialized_layer in ("projection", "embedding")
    )
    def test_serialized_splits_per_ipu_validator(self, layer, mode):
        # Must be of type List[int>=0]
        ipu_config = IPUConfig()
        ipu_config.mode = mode
        serialized_mode_layer = ipu_config._get_managed_attr_mode_name(f"serialized_{layer}_splits_per_ipu")
        with pytest.raises(ValueError, match=f"`IPUConfig` attribute `{serialized_mode_layer}` must .* >="):
            setattr(ipu_config, serialized_mode_layer, [0, 2, 2, -1])

        # Must have atleast 1 split if the pipeline is provided
        with pytest.raises(
            ValueError,
            match=re.escape(f"`IPUConfig` attribute `{serialized_mode_layer}=[0, 0]` must have atleast 1 split"),
        ):
            setattr(ipu_config, serialized_mode_layer, [0, 0])

        # Splits should be on consecutive IPUs
        with pytest.raises(
            ValueError,
            match=re.escape(f"`IPUConfig` attribute `{serialized_mode_layer}=[0, 3, 0, 2]` must have its splits on"),
        ):
            setattr(ipu_config, serialized_mode_layer, [0, 3, 0, 2])

    @parameterized.expand((mode for mode in IPUConfig.modes))
    def test_validate_ipu_config(self, mode):
        with self.subTest(
            "if *matmul_proportion is a List[float], it must use the same number of IPUs as *ipus_per_replica"
        ):
            ipus_per_replica = 4
            matmul_proportion = [0.2] * (ipus_per_replica + 1)
            with pytest.raises(
                IncompatibleIPUConfigError,
                match=re.escape(f"matmul_proportion={matmul_proportion} should use the same number"),
            ):
                create_mode_ipu_config(
                    {"ipus_per_replica": ipus_per_replica, "matmul_proportion": matmul_proportion}, mode
                )
                IPUConfig(ipus_per_replica=ipus_per_replica, matmul_proportion=matmul_proportion)

        with self.subTest(
            "If there are no wildcards in *layers_per_ipu, the pipeline length should equal *ipus_per_replica"
        ):
            layers_per_ipu = [2] * (ipus_per_replica + 1)
            with pytest.raises(
                IncompatibleIPUConfigError,
                match=re.escape(f"layers_per_ipu={layers_per_ipu} should use the same number"),
            ):
                create_mode_ipu_config({"ipus_per_replica": ipus_per_replica, "layers_per_ipu": layers_per_ipu}, mode)

        # The user cannot provide both {projection/embedding}_serialization_factor and
        # serialized_{projection/embedding}_splits_per_ipu
        for layer in ("projection", "embedding"):
            with self.subTest(
                f"The user cannot provide both {layer}_serialization_factor" f" and serialized_{layer}_splits_per_ipu."
            ):
                with pytest.raises(ValueError, match=f"Only one of .*{layer}.*"):
                    create_mode_ipu_config(
                        {f"{layer}_serialization_factor": 2, f"serialized_{layer}_splits_per_ipu": [1, 1]}, mode
                    )

            with self.subTest(
                f"The pipeline length of serialized_{layer}_splits_per_ipu must equal"
                " the number of IPUs specified by IPUs per replica."
            ):
                with pytest.raises(
                    ValueError, match=f".*{layer}.*=\\[0, 0, 1, 1\\] should use the same number of IPUs as"
                ):
                    create_mode_ipu_config(
                        {"ipus_per_replica": 8, f"serialized_{layer}_splits_per_ipu": [0, 0, 1, 1]}, mode
                    )
