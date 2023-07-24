# coding=utf-8
#  Copyright 2021 The HuggingFace Team. All rights reserved.
#  Copyright (c) 2022 Graphcore Ltd. All rights reserved.
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
import json
import warnings
from collections import defaultdict
from functools import partial
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional, Sequence, Union, get_type_hints

import popart
import poptorch
import torch
import typeguard
from poptorch import Options, OutputMode

from optimum.configuration_utils import BaseConfig
from optimum.utils import logging


# For container types check all items for type correctness
# rather than just the first element
typeguard._config.global_config.collection_check_strategy = typeguard.config.collection_check_strategy.ALL_ITEMS
logger = logging.get_logger(__name__)

IPU_CONFIG_NAME = "ipu_config.json"


class IncompatibleIPUConfigError(ValueError):
    "Exception raised when IPU config values are invalid"
    "or are not compatible with a model"
    pass


class IPUConfig(BaseConfig):
    """
    Class for configuring PopArt and PyTorch for the IPU. Handles the conversion to `poptorch` options as well as configuration of the
    IPU-Pod type specialization.

    Args:
        seed (`int`, *optional*):
            Sets the seed for the random number generator on the IPU.
        auto_loss_scaling (`bool`, *optional*, defaults to `False`):
            If `True`, enables automatic loss scaling on the IPU.
            When using float16/half-precision values for activations, gradients, and weights, the loss value needs to be scaled by
            a constant factor to avoid underflows or overflows. This adjustment is known as loss scaling. This setting
            automatically sets a global loss scaling factor during training.
            **Note: This is an experimental feature and may not behave as expected.**
        executable_cache_dir (`str`, *optional*, defaults to `""`):
            Enables caching the compile executables to a directory.

        > Parameters for controlling the batch size

        replication_factor (`int`, *optional*, defaults to 1):
            The number of replicas for data-parallelism during training. It depends on the size of the pipeline as well
            as the number of IPUs available. For example: on a Pod16, with a 4-IPU pipeline, the replication_factor must
            be between 1 and 4.
        inference_replication_factor (`int`, *optional*, defaults to 1):
            The number of replicas for data-parallelism during inference. It depends on the size of the pipeline as well
            as the number of IPUs available. For example: on a Pod16, with a 4-IPU pipeline, the replication_factor must
            be between 1 and 4.
        gradient_accumulation_steps (`int`, *optional*, defaults to 1):
            Number of micro-batches to accumulate for the gradient calculation.
            Accumulates the gradient `gradient_accumulation` times before updating the model using the gradient.

        > Parameters related to parallelism

        layers_per_ipu (`List[int]`):
            Specifies the number of layers that will be put on each IPU for pipelined execution during training.
            For instance: `[2, 3, 4, 2]` specifies a 4-IPU pipeline, where the first two layers will be put on IPU0,
            the following three on IPU1, the next four on IPU2 and the last two on IPU3.
            If the default of [-1] is used, the layers will be split evenly over `ipus_per_replica` IPUs.
            The wildcard value '-1' can also be used in combination with integers.
            For instance: `[1, 2, -1, -1]` specifies a 4-IPU pipeline, where the first layer is put on IPU0,
            the next two layers on IPU1, and the remaining layers split evenly between IPU2 and IPU3.
        inference_layers_per_ipu (`List[int]`):
            Same as `layers_per_ipu` for inference only.
        ipus_per_replica (`int`, *optional*, defaults to `len(layers_per_ipu)`):
            Specifies the number of IPUs to use during training. This must be consistent with
            the number of IPUs used in `layers_per_ipu`.
        inference_ipus_per_replica (`int`, *optional*, defaults to `len(inference_layers_per_ipu) if ipus_per_replica==len(layers_per_ipu) else ipus_per_replica):
            Same as `ipus_per_replica` but for inference only.
        parallelize_kwargs (`Dict[str, Any]`, *optional*, defaults to None):
            Dictionary holding kwargs used for training model calls to `parallelize`.
        inference_parallelize_kwargs (`Dict[str, Any]`, *optional*, defaults to None):
            Dictionary holding kwargs used for inference model calls to `parallelize`.

        > Parameters for memory management

        optimizer_state_offchip (`bool`, *optional*, defaults to `True`):
            If `True`, uses the off-chip memory to store the optimizer state. If `False`, uses the on-chip memory.
        replicated_tensor_sharding (`bool`, *optional*, defaults to `False`):
            Shards the optimizer between replicas with zero-redundancy.
        matmul_proportion (`List[float]` or `float`, *optional*, defaults to 0.2):
            Sets the amount of temporary memory made available during training on per-IPU basis.
            Use this setting to control the amount of temporary memory available to operations such as:
              - convolution
              - matrix multiplication
              - embedding lookups
              - indexing operations
        inference_matmul_proportion (`List[float]` or `float`):
            Same as `matmul_proportion` for inference only.
        enable_half_partials (`bool`, *optional*, defaults to `True`):
            If `True`, sets the data type of partial results for matrix multiplication and convolution operators to float16.
        embedding_serialization_factor (`int`, *optional*, defaults to 1 if `serialized_embedding_splits_per_ipu` is `None`):
            The factor to use to serialize embeddings. Nothing happens if `embedding_serialization_factor = 1`. For
            `embedding_serialization_factor > 1`, the `torch.nn.Embedding` layer is replaced with a
            `optimum.graphcore.modeling_utils.SerializedEmbedding` layer.
            Note: only one of `embedding_serialization_factor` or `serialized_embedding_splits_per_ipu` should be provided.
        inference_embedding_serialization_factor (`int`, *optional*, defaults to 1 if `inference_serialized_embedding_splits_per_ipu` is `None`):
            Same as `embedding_serialization_factor` but for inference only.
        serialized_embedding_splits_per_ipu (`List[int]`, *optional*, defaults to None):
            Specifies the number of splits of the embedding layer that will be put on each IPU for pipelined execution.
            The format has to be the same as that for `layers_per_ipu` however wildcards are not supported.
            For instance: `[3, 1, 0, 0]` specifies how to place an embedding layer serialized into
            4 sub-embedding layers across a 4-IPU pipeline. IPU-1 has 3 splits and IPU-2 has 1 split.
            The remaining IPUs have no sub-embedding layers. If an argument to this parameter is provided,
            it must:
            - be of the form `List[int>=0]` with atleast 1 split.
            - have the same pipeline length as `ipus_per_replica`
            - have splits that are consecutive with no zeros between splits e.g. `[3, 0, 2, 0]` is invalid
            - for generation, splits must lie entirely on the encoder or decoder portion of the pipeline.
            For example the 4-IPU pipeline `[3, 1, 0, 0]` for an encoder-decoder model can be split into
            `[3, 1]` and `[0, 0]`, however `[0, 1, 2, 0]` split into `[0, 1]` and `[2, 0]` is invalid.
            Note: only one of `embedding_serialization_factor` or `serialized_embedding_splits_per_ipu` should be set.
        inference_serialized_embedding_splits_per_ipu (`List[int]`, *optional*, defaults to None):
            Same as `serialized_embedding_splits_per_ipu` but for inference only.
        projection_serialization_factor (`int`, *optional*, defaults to 1 if `serialized_projection_splits_per_ipu` is `None`):
            The factor to use to either serialize the matmuls that are performed in the linear projection layer, or,
            serialize the projection layer into a set of individual linear layers that can be optionally placed on different IPUs.
            Nothing happens if `projection_serialization_factor = 1`. If `projection_serialization_factor > 1`,
            the `torch.nn.Linear` layer is replaced by a `optimum.graphcore.modeling_utils.SplitProjection` layer
            if `serialized_projection_splits_per_ipu` is provided and the linear layer's weights are not tied to another layer.
            Otherwise it is replaced by a `optimum.graphcore.modeling_utils.SerializedLinear` layer.
            Note: only one of `projection_serialization_factor` or `serialized_projection_splits_per_ipu` should be set.
        inference_projection_serialization_factor (`int`, *optional*, defaults to 1 if `inference_serialized_projection_splits_per_ipu` is `None`):
            Same as `projection_serialization_factor` but for inference only.
        serialized_projection_splits_per_ipu (`List[int]`, *optional*, defaults to None):
            Analogous to `serialized_embedding_splits_per_ipu`.
            Note: only one of `projection_serialization_factor` or `serialized_projection_splits_per_ipu` should be set.
        inference_serialized_projection_splits_per_ipu (`List[int]`, *optional*, defaults to None):
            Same as `serialized_projection_splits_per_ipu` but for inference only.
        recompute_checkpoint_every_layer (`bool`, *optional*, defaults to `False`):
            If `True`, uses gradient checkpointing at the end of every layer. It can help to reduce the memory impact.
        explicit_ir_inference (`bool`, *optional*, defaults to `False`):
            If `True`, uses experimental explicit-IR feature of PopART for inference models. This feature is only supported
            for inference models. For some cases explicit-IR can provide a better memory liveness schedule, reducing the peak
            memory during runtime.

        > Parameters related to host/device synchronization

        device_iterations (`int`, *optional*, defaults to 1):
            Number of iterations the device should run over the data before returning to the user during training. This
            is equivalent to running the IPU in a loop over the specified number of iterations, with a new batch of
            data each time. However, increasing the number of device iterations is more efficient because the loop runs on the IPU
            directly.
        inference_device_iterations (`int`, *optional*, defaults to 1):
            Same as `device_iterations` for inference.
        output_mode (`str`, *optional*, defaults to `"final"`):
            Specifies which data to return from a model.
            Allowed values:
              - `all`: returns a result for each batch.
              - `sum`: returns the sum of all batches.
              - `final`: returns the last batch.
              - `default`: `all` for inference, `final` for training.

    """

    CONFIG_NAME = "ipu_config.json"
    FULL_CONFIGURATION_FILE = "ipu_config.json"

    class ManagedAttribute:
        def __init__(self, attr) -> None:
            self.attr = attr

        def __set__(self, obj, value):
            if isinstance(obj, IPUConfig):
                mode_attr = f"inference_{self.attr}" if obj.mode == "inference" else self.attr
                logger.debug(f"ManagedAttribute {self.attr} writing to {mode_attr}")
                return setattr(obj, mode_attr, value)

        def __get__(self, obj, objtype=None):
            if isinstance(obj, IPUConfig):
                mode_attr = f"inference_{self.attr}" if obj.mode == "inference" else self.attr
                logger.debug(f"ManagedAttribute {self.attr} reading from {mode_attr}")
                return getattr(obj, mode_attr)

    # Create descriptor based managed attributes which will either return the
    # `attribute` or `inference_attribute` versions of the attribute depending on the value of
    # `self.mode` ("training" by default). For example `_layers_per_ipu`
    # switches between `layers_per_ipu` and `inference_layers_per_ipu`
    modes = ("training", "inference")
    _layers_per_ipu = ManagedAttribute("layers_per_ipu")
    _ipus_per_replica = ManagedAttribute("ipus_per_replica")
    _matmul_proportion = ManagedAttribute("matmul_proportion")
    _embedding_serialization_factor = ManagedAttribute("embedding_serialization_factor")
    _serialized_embedding_splits_per_ipu = ManagedAttribute("serialized_embedding_splits_per_ipu")
    _projection_serialization_factor = ManagedAttribute("projection_serialization_factor")
    _serialized_projection_splits_per_ipu = ManagedAttribute("serialized_projection_splits_per_ipu")
    _parallelize_kwargs = ManagedAttribute("parallelize_kwargs")

    # Create a mapping of attributes to their list of validation functions
    attribute_validators = defaultdict(list)

    def _contents_geq_value_validator(
        name: str, value: Union[float, int, Sequence], floor_value: Union[float, int]
    ) -> None:
        """
        Validates the values of Sequence and scalar types to be greater than `floor_value`
        For Sequence[Union[int, float]], ensure that all elements are >= floor_value
        For Union[float, int], ensure the scalar is >= floor_value
        """

        # Do nothing for optional types
        if value is None:
            return
        elif isinstance(value, Sequence):
            if not all(elem >= floor_value for elem in value):
                raise ValueError(
                    f"`IPUConfig` attribute `{name}` must have all elements >= {floor_value}. You provided {value=}"
                )
        elif isinstance(value, (int, float)):
            if not value >= floor_value:
                raise ValueError(f"`IPUConfig` attribute `{name}` must be >= {floor_value}. You provided {value=}")
        else:
            raise ValueError(
                f"`contents_geq_value_validator` validates inputs of type:"
                f" Union[float, int, Sequence[Union[int, float]]]. You provided"
                f" attribute `{name}`, {value=}, {type(value)}"
            )

    for attr, floor_value in (
        ("layers_per_ipu", -1),
        ("inference_layers_per_ipu", -1),
        ("matmul_proportion", 0),
        ("inference_matmul_proportion", 0),
        ("replication_factor", 1),
        ("inference_replication_factor", 1),
        ("gradient_accumulation_steps", 1),
        ("ipus_per_replica", 1),
        ("inference_ipus_per_replica", 1),
        ("embedding_serialization_factor", 1),
        ("inference_embedding_serialization_factor", 1),
        ("projection_serialization_factor", 1),
        ("inference_projection_serialization_factor", 1),
        ("device_iterations", 1),
        ("inference_device_iterations", 1),
    ):
        attribute_validators[attr].append(partial(_contents_geq_value_validator, floor_value=floor_value))

    def _output_mode_validator(name: str, value: str):
        allowed_values = ("all", "sum", "final", "default")
        if value not in allowed_values:
            raise ValueError(
                f"`IPUConfig` attribute `output_mode` can only take values in"
                f" {allowed_values}. You provided: {value=}"
            )

    attribute_validators["output_mode"].append(_output_mode_validator)

    def _serialized_layer_splits_per_ipu_validator(name: str, value: int):
        """
        Validates serialized_{projection/embedding}_splits_per_ipu attributes.
        If `value` is not None. `value` must be of type List[int>=0] with
        atleast 1 split on 1 IPU. Further splits in the pipeline must be
        consecutive.
        """

        if value is None:
            return

        IPUConfig._contents_geq_value_validator(name, value, floor_value=0)

        # There must be atleast 1 split when the pipeline is provided
        if sum(value) < 1:
            raise ValueError(f"`IPUConfig` attribute `{name}={value}` must have atleast 1 split on 1 IPU.")

        # Check that splits are on consecutive IPUs (e.g. [3,0,2,0] is not allowed)
        for i, splits in enumerate(value[:-1]):
            if splits and value[i + 1] == 0 and sum(value[i + 1 :]) != 0:
                raise ValueError(f"`IPUConfig` attribute `{name}={value}` must have its splits on consecutive IPUs.")

    for attr in (
        "serialized_embedding_splits_per_ipu",
        "inference_serialized_embedding_splits_per_ipu",
        "serialized_projection_splits_per_ipu",
        "inference_serialized_projection_splits_per_ipu",
    ):
        attribute_validators[attr].append(_serialized_layer_splits_per_ipu_validator)

    def __init__(
        self,
        replication_factor: int = 1,
        inference_replication_factor: int = 1,
        gradient_accumulation_steps: int = 1,
        layers_per_ipu: List[int] = [-1],
        inference_layers_per_ipu: Optional[List[int]] = None,
        ipus_per_replica: Optional[int] = None,
        inference_ipus_per_replica: Optional[int] = None,
        optimizer_state_offchip: bool = False,
        replicated_tensor_sharding: bool = False,
        matmul_proportion: Union[float, List[float]] = 0.2,
        inference_matmul_proportion: Optional[Union[float, List[float]]] = None,
        enable_half_partials: bool = True,
        embedding_serialization_factor: Optional[int] = None,
        inference_embedding_serialization_factor: Optional[int] = None,
        serialized_embedding_splits_per_ipu: Optional[List[int]] = None,
        inference_serialized_embedding_splits_per_ipu: Optional[List[int]] = None,
        projection_serialization_factor: Optional[int] = None,
        inference_projection_serialization_factor: Optional[int] = None,
        serialized_projection_splits_per_ipu: Optional[List[int]] = None,
        inference_serialized_projection_splits_per_ipu: Optional[List[int]] = None,
        recompute_checkpoint_every_layer: bool = False,
        device_iterations: int = 1,
        inference_device_iterations: int = 1,
        output_mode: str = "final",
        seed: Optional[int] = None,
        auto_loss_scaling: bool = False,
        executable_cache_dir: str = "",
        explicit_ir_inference: bool = False,
        parallelize_kwargs: Optional[Dict[str, Any]] = None,
        inference_parallelize_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.seed = seed

        # Default mode to `training`
        self.train()

        self.layers_per_ipu = layers_per_ipu
        self.inference_layers_per_ipu = inference_layers_per_ipu if inference_layers_per_ipu else self.layers_per_ipu

        self.ipus_per_replica = ipus_per_replica if ipus_per_replica else len(self.layers_per_ipu)
        # If ipus_per_replica is default, recalculate ipus_per_replica from inference_layers_per_ipu instead
        fallback_ipus_per_replica = self.ipus_per_replica
        if fallback_ipus_per_replica == len(self.layers_per_ipu) or self.inference_layers_per_ipu != [-1]:
            fallback_ipus_per_replica = len(self.inference_layers_per_ipu)

        self.inference_ipus_per_replica = (
            inference_ipus_per_replica if inference_ipus_per_replica else fallback_ipus_per_replica
        )

        self.matmul_proportion = matmul_proportion
        # If matmul_proportion is a list and its length is not equal to inference_ipus_per_replica, use the
        # default float value for matmul_proportion instead
        fallback_matmul_proportion = self.matmul_proportion
        if isinstance(self.matmul_proportion, list) and len(self.matmul_proportion) != self.inference_ipus_per_replica:
            fallback_matmul_proportion = 0.2
        self.inference_matmul_proportion = (
            inference_matmul_proportion if inference_matmul_proportion else fallback_matmul_proportion
        )

        def check_and_set_replication_factor(attr_name, attr):
            if isinstance(attr, int):
                setattr(self, attr_name, attr)
            else:
                raise ValueError(f"{attr_name} must be of type `int`. You provided: {attr_name}={attr}, {type(attr)}.")

        check_and_set_replication_factor("replication_factor", replication_factor)
        check_and_set_replication_factor("inference_replication_factor", inference_replication_factor)

        # Non-transformer layers initialisation
        self.embedding_serialization_factor = embedding_serialization_factor
        self.inference_embedding_serialization_factor = inference_embedding_serialization_factor
        self.serialized_embedding_splits_per_ipu = serialized_embedding_splits_per_ipu
        self.inference_serialized_embedding_splits_per_ipu = inference_serialized_embedding_splits_per_ipu

        self.projection_serialization_factor = projection_serialization_factor
        self.inference_projection_serialization_factor = inference_projection_serialization_factor
        self.serialized_projection_splits_per_ipu = serialized_projection_splits_per_ipu
        self.inference_serialized_projection_splits_per_ipu = inference_serialized_projection_splits_per_ipu

        if kwargs.pop("sharded_execution_for_inference", None):
            warnings.warn(
                'The "sharded_execution_for_inference" parameter is deprecated, sharded execution is always used during inference'
            )

        if kwargs.pop("enable_half_first_order_momentum", None):
            warnings.warn('The "enable_half_first_order_momentum" parameter is deprecated')

        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.device_iterations = device_iterations
        self.inference_device_iterations = inference_device_iterations
        self.optimizer_state_offchip = optimizer_state_offchip
        self.replicated_tensor_sharding = replicated_tensor_sharding
        self.auto_loss_scaling = auto_loss_scaling
        self.enable_half_partials = enable_half_partials
        self.executable_cache_dir = executable_cache_dir
        self.explicit_ir_inference = explicit_ir_inference
        self.embedding_serialization_factor = embedding_serialization_factor
        self.recompute_checkpoint_every_layer = recompute_checkpoint_every_layer
        self.output_mode = output_mode

        self.parallelize_kwargs = parallelize_kwargs or {}
        self.inference_parallelize_kwargs = inference_parallelize_kwargs or {}

        # TODO: remove this if unnecessary.
        self.execute_encoder_on_cpu_for_generation = kwargs.pop("execute_encoder_on_cpu_for_generation", False)

        # Raise error if user has provided unknown & unused kwarg
        if unknown_kwargs := (set(kwargs) - set(BaseConfig().to_dict())):
            raise IncompatibleIPUConfigError(
                "IPUConfig received unknown arguments:\n" + "\n".join([f"  {k}={kwargs[k]}" for k in unknown_kwargs])
            )

        self._validate_ipu_config()

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value):
        if value not in self.modes:
            raise ValueError(
                f"`IPUConfig` mode can only take values in {self.modes}."
                f" You provided: {value=}. Use the `train` and `eval` methods"
                " instead to avoid error."
            )
        self._mode = value

    def train(self):
        self._mode = "training"
        return self

    def eval(self):
        self._mode = "inference"
        return self

    def _get_managed_attr_mode_name(self, attr: str) -> str:
        """
        Returns the attribute name that a ManagedAttribute descriptor is
        currently referring to
        """
        # Shallow check to ensure that the input attribute is actually
        # a managed attribute
        if hasattr(self, attr) and hasattr(self, f"inference_{attr}"):
            return attr if self.mode == "training" else f"inference_{attr}"
        # return attr if its not a managed attribute
        return attr

    def _get_attribute_type(self, name: str) -> Any:
        """
        Returns the input `name` attribute type hints. Returns `Any` type by default.
        The return type for an attribute is only specific if it is a parameter in the
        signature of IPUConfig.__init__
        """
        try:
            type_hints = self._attribute_type_hints
        except AttributeError:
            type_hints = get_type_hints(IPUConfig.__init__)
            super().__setattr__("_attribute_type_hints", type_hints)
        return type_hints.get(name, Any)

    def __setattr__(self, name: str, value: Any):
        """Override __setattr__ to include value type checking
        and validation
        """
        attr_type = self._get_attribute_type(name)
        try:
            typeguard.check_type(value, attr_type)
        except typeguard.TypeCheckError as e:
            raise TypeError(
                f"Setting `IPUConfig` attribute: {name}, type: {attr_type}"
                f" with {value=}, type: {type(value)} is invalid."
            ) from e

        # Run attribute value validators
        if name in self.attribute_validators:
            for vfunc in self.attribute_validators[name]:
                vfunc(name, value)

        return super().__setattr__(name, value)

    def _validate_ipu_config(self):
        """
        Tests coherence of `IPUConfig` attributes for all modes
        in self.modes. For example if `matmul_proportion=[0.2, 0.2]`,
        `ipus_per_replica` must have value 2.

        Raises:
            IncompatibleIPUConfigError: Raised if any `IPUConfig` attributes are not coherent.
        """
        if self.replicated_tensor_sharding and self.replication_factor == 1:
            logger.warning("`replicated_tensor_sharding` is not used when `replication_factor=1`")

        old_mode = self.mode
        for mode in self.modes:
            self.mode = mode

            ipus_per_replica_mode_str = self._get_managed_attr_mode_name("ipus_per_replica")

            # len(matmul_proportion) must equal ipus_per_replica
            if isinstance(self._matmul_proportion, list) and len(self._matmul_proportion) != self._ipus_per_replica:
                matmul_proportion_mode_str = self._get_managed_attr_mode_name("matmul_proportion")
                raise IncompatibleIPUConfigError(
                    f"{matmul_proportion_mode_str}={self._matmul_proportion} should use the"
                    f" same number of IPUs as {ipus_per_replica_mode_str}={self._ipus_per_replica}."
                )

            # layers_per_ipu must have the same length as ipus per replica.
            # If wildcards are present in layers_per_ipu, let the call to `model.parallelize`
            # handle the validation
            if -1 not in self._layers_per_ipu and len(self._layers_per_ipu) != self._ipus_per_replica:
                layers_per_ipu_mode_str = self._get_managed_attr_mode_name("layers_per_ipu")
                raise IncompatibleIPUConfigError(
                    f"{layers_per_ipu_mode_str}={self._layers_per_ipu} should use the"
                    f" same number of IPUs as {ipus_per_replica_mode_str}={self._ipus_per_replica}."
                )

            # Validate non-transformer layer placement configuration
            for layer in ("embedding", "projection"):
                mode_layer_splits_per_ipu_str = self._get_managed_attr_mode_name(f"serialized_{layer}_splits_per_ipu")
                mode_layer_splits_per_ipu = getattr(self, mode_layer_splits_per_ipu_str)
                mode_layer_serialisation_factor_str = self._get_managed_attr_mode_name(f"{layer}_serialization_factor")
                mode_layer_serialization_factor = getattr(self, mode_layer_serialisation_factor_str)

                # If the user has not provided either the layer_serialization_factor or
                # layer_splits_per_ipu, default the layer_serialization_factor to 1
                if not (mode_layer_splits_per_ipu or mode_layer_serialization_factor):
                    setattr(self, mode_layer_serialisation_factor_str, 1)

                # If the user provides both options, tell them only one is allowed and what each option is for
                if mode_layer_splits_per_ipu and mode_layer_serialization_factor:
                    raise ValueError(
                        f"Only one of `{mode_layer_serialisation_factor_str}` and `{mode_layer_splits_per_ipu_str}` should"
                        f" be used at once. `{mode_layer_serialisation_factor_str}` should be used when you want your"
                        f" {layer} layer serialised on the same IPU (which IPU depends on the model)."
                        f" `{mode_layer_splits_per_ipu_str}` should be used when you want your {layer} layer to be split"
                        " across multiple IPUs of your choice (or to choose which single IPU the layer is serialised on)."
                    )

                # Serialized layer splits per ipu pipeline must have the same pipeline length
                # as the number of ipus per replica
                if mode_layer_splits_per_ipu and len(mode_layer_splits_per_ipu) != self._ipus_per_replica:
                    raise ValueError(
                        f"{mode_layer_splits_per_ipu_str}={mode_layer_splits_per_ipu}"
                        f" should use the same number of IPUs as {ipus_per_replica_mode_str}={self._ipus_per_replica}."
                    )

        self.mode = old_mode
        return self

    def _to_options(self, for_inference: bool = False, compile_only: bool = False) -> poptorch.Options:
        if not compile_only and poptorch.ipuHardwareVersion() not in (2, 21):
            raise RuntimeError("This requires an IPU Mk2 system to run.")

        if self.execute_encoder_on_cpu_for_generation:
            raise NotImplementedError("execute_encoder_on_cpu_for_generation is not supported yet.")

        old_mode = self.mode
        self.eval() if for_inference else self.train()

        opts = Options()
        opts.autoRoundNumIPUs(True)
        opts.replicationFactor(self.inference_replication_factor if for_inference else self.replication_factor)
        opts.deviceIterations(self.inference_device_iterations if for_inference else self.device_iterations)

        if not for_inference:
            # Set gradient accumulation factor
            opts.Training.gradientAccumulation(self.gradient_accumulation_steps)
            opts.Training.accumulationAndReplicationReductionType(poptorch.ReductionType.Mean)

        # Enable automatic loss scaling
        # Note that this is an experimental feature. Note also that it expects
        # accumulationAndReplicationReductionType to be set to Mean as above,
        # and for accumulation by the optimizer to be done in half precision
        # using accum_type=torch.float16 during optimizer instantiation.
        if self.auto_loss_scaling and not for_inference:
            opts.Training.setAutomaticLossScaling(True)

        # Return all results from IPU to host
        output_mode_mapping = {
            "all": OutputMode.All,
            "sum": OutputMode.Sum,
            "final": OutputMode.Final,
            "default": OutputMode.Default,
        }
        training_output_mode = output_mode_mapping.get(self.output_mode, None)
        if training_output_mode is None:
            supported_output_modes = ", ".join(output_mode_mapping.keys())
            raise KeyError(
                f"{self.output_mode} is not a valid poptorch.OutputMode, supported output modes: {supported_output_modes}"
            )
        opts.outputMode(OutputMode.All if for_inference else training_output_mode)

        if self.seed:
            opts.randomSeed(self.seed)

        # Enable replicated tensor sharding of optimizer state
        # with optimizer state residing either on-chip or in DRAM.
        # RTS is only enabled if replication factor is also greater than 1
        opts.TensorLocations.setOptimizerLocation(
            poptorch.TensorLocationSettings()
            # Optimizer state lives on- or off-chip
            .useOnChipStorage(not self.optimizer_state_offchip)
            # Shard optimizer state between replicas with zero-redundancy
            .useReplicatedTensorSharding(self.replicated_tensor_sharding and opts.replication_factor > 1)
        )

        if for_inference:
            opts.setExecutionStrategy(poptorch.ShardedExecution(poptorch.AutoStage.AutoIncrement))
        else:
            # Use Pipelined Execution
            opts.setExecutionStrategy(poptorch.PipelinedExecution(poptorch.AutoStage.AutoIncrement))

        # Compile offline (no IPUs required)
        if compile_only:
            opts.useOfflineIpuTarget()

        matmul_proportion = copy.deepcopy(self._matmul_proportion)
        if isinstance(matmul_proportion, float):
            matmul_proportion = [matmul_proportion] * self._ipus_per_replica
        mem_prop = {f"IPU{i}": matmul_proportion[i] for i in range(self._ipus_per_replica)}
        opts.setAvailableMemoryProportion(mem_prop)

        # Enable caching the compiled executable to disk
        if self.executable_cache_dir and self.executable_cache_dir != "disabled":
            opts.enableExecutableCaching(self.executable_cache_dir)

        opts._Popart.set("saveInitializersToFile", NamedTemporaryFile().name)

        # Enable stochastic rounding (recommended for training with FP16)
        opts.Precision.enableStochasticRounding(not for_inference)

        # Half precision partials for matmuls and convolutions
        if self.enable_half_partials:
            opts.Precision.setPartialsType(torch.float16)

        # PopART performance options #
        # Only stream needed tensors back to host
        opts._Popart.set("disableGradAccumulationTensorStreams", True)
        # Parallelize optimizer step update across IPUs
        opts._Popart.set(
            "accumulateOuterFragmentSettings.schedule",
            int(popart.AccumulateOuterFragmentSchedule.OverlapMemoryOptimized),
        )
        opts._Popart.set("accumulateOuterFragmentSettings.excludedVirtualGraphs", ["0"])
        # Enable patterns for better throughput and memory reduction
        opts._Popart.set("outlineThreshold", 10.0)
        opts._Popart.set("subgraphCopyingStrategy", int(popart.SubgraphCopyingStrategy.JustInTime))
        opts._Popart.set("scheduleNonWeightUpdateGradientConsumersEarly", True)
        opts._Popart.setPatterns(
            {"TiedGather": True, "TiedGatherAccumulate": True, "UpdateInplacePrioritiesForIpu": True}
        )

        # Options for profiling with Popvision
        engine_options = {
            "opt.useAutoloader": "true",
            "target.syncReplicasIndependently": "true",
        }

        if for_inference and self.explicit_ir_inference:
            opts._popart.set("enableExplicitIR", True)

        opts._Popart.set("engineOptions", engine_options)

        self.mode = old_mode
        return opts

    def to_options(self, for_inference: bool = False, compile_only: bool = False) -> poptorch.Options:
        """
        Creates a `poptorch.Options` instance from the `IPUConfig` instance.

        Args:
            for_inference (`bool`, defaults to `False`):
                If `True`, the resulting `poptorch.Options` will be adapted for inference. If `False`, the resulting `poptorch.Options` will be adapted for training.
            compile_only (`bool`, defaults to `False`):
                If True, compilation will be performed offline, no IPUs required.

        Returns:
            `poptorch.Options`: The options representing the `IPUConfig` instance.
        """
        return self._to_options(for_inference=for_inference, compile_only=compile_only)

    # Adapted from BaseConfig.to_dict
    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        output = super().to_dict()

        # Remove type hints as they are not serializable
        output.pop("_attribute_type_hints", None)

        # Remove mode as it's not relevant for a dict
        output.pop("_mode", None)

        return output

    def batch_size_factor(self, for_inference: bool = False) -> int:
        """
        Computes the factor to apply to the micro batch size to calculate the combined batch size.

        Args:
            for_inference (`bool`, defaults to `False`):
                Whether the factor is being use to compute the batch size for inference or not.

        Returns:
            `int`: The batch size factor.
        """
        replication_factor = self.inference_replication_factor if for_inference else self.replication_factor
        gradient_accumulation_steps = 1 if for_inference else self.gradient_accumulation_steps
        device_iterations = self.inference_device_iterations if for_inference else self.device_iterations
        return replication_factor * gradient_accumulation_steps * device_iterations

    def update_from_string(self, update_str: str):
        """
        Updates attributes of the `IPUConfig` class with attributes from `update_str`.

        The expected format is ints, floats and strings as is, and for booleans use `true` or `false`, and for lists
        use `[a b c d]`. For example: `"n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index,
        matmul_proportion=[0.08 0.2 0.25 0.25]"`.

        The keys to change must already exist in the config object.

        Args:
            update_str (`str`): String with attributes that should be updated for this class.

        """

        d = dict(x.split("=") for x in update_str.split(","))
        for k, v in d.items():
            if not hasattr(self, k):
                raise ValueError(f"Key {k} isn't in the original config dict")

            old_v = getattr(self, k)
            if isinstance(old_v, bool):
                if v.lower() in ["true", "1", "y", "yes"]:
                    v = True
                elif v.lower() in ["false", "0", "n", "no"]:
                    v = False
                else:
                    raise ValueError(f"Can't derive true or false from {v} (key {k})")
            elif isinstance(old_v, int):
                v = int(v)
            elif isinstance(old_v, float):
                v = float(v)
            elif isinstance(old_v, list):
                v = json.loads(v.replace(" ", ","))
            elif not isinstance(old_v, str):
                raise ValueError(
                    f"You can only update int, float, bool, list or string values in the config, got {v} for key {k}"
                )

            setattr(self, k, v)
