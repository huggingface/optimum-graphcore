# coding=utf-8
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
import json
import warnings
from functools import partial
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional, Sequence, Set, Type, Union, get_type_hints

import torch

import popart
import poptorch
import typeguard
from optimum.configuration_utils import BaseConfig
from optimum.utils import logging
from poptorch import Options, OutputMode


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
            Specifies the number of layers that will be put on each IPU for pipelined execution.
            For instance: `[2, 3, 4, 2]` specifies a 4-IPU pipeline, where the first two layers will be put on IPU0,
            the following three on IPU1, the next four on IPU2 and the last two on IPU3.
            If the default of [-1] is used, the layers will be split evenly over `ipus_per_replica` IPUs.
            The wildcard value '-1' can also be used in combination with integers.
            For instance: `[1, 2, -1, -1]` specifies a 4-IPU pipeline, where the first layer is put on IPU0,
            the next two layers on IPU1, and the remaining layers split evenly between IPU2 and IPU3.
        training_layers_per_ipu (`List[int]`):
            Same as `layers_per_ipu` for training only.
        inference_layers_per_ipu (`List[int]`):
            Same as `layers_per_ipu` for inference only.
        ipus_per_replica (`int`, *optional*, defaults to `len(layers_per_ipu)`):
            Specifies the number of IPUs to use during training. This must be consistent with
            the number of IPUs used in `layers_per_ipu`.
        inference_ipus_per_replica (`int`, *optional*, defaults to `len(inference_layers_per_ipu) if ipus_per_replica==len(layers_per_ipu) else ipus_per_replica):
            Same as `ipus_per_replica` but for inference only.

        > Parameters for memory management

        optimizer_state_offchip (`bool`, *optional*, defaults to `True`):
            If `True`, uses the off-chip memory to store the optimizer state. If `False`, uses the on-chip memory.
        replicated_tensor_sharding (`bool`, *optional*, defaults to `False`):
            Shards the optimizer between replicas with zero-redundancy.
        matmul_proportion (`List[float]` or `float`, *optional*, defaults to 0.2):
            Sets the amount of temporary memory made available on per-IPU basis.
            Use this setting to control the amount of temporary memory available to operations such as:
              - convolution
              - matrix multiplication
              - embedding lookups
              - indexing operations
        training_matmul_proportion (`List[int]`):
            Same as `matmul_proportion` for training only.
        inference_matmul_proportion (`List[int]`):
            Same as `matmul_proportion` for inference only.
        enable_half_partials (`bool`, *optional*, defaults to `True`):
            If `True`, sets the data type of partial results for matrix multiplication and convolution operators to float16.
        embedding_serialization_factor (`int`, *optional*, defaults to 1):
            The factor to use to serialize embeddings. Nothing happens if `embedding_serialization_factor = 1`. For
            `embedding_serialization_factor > 1`, the `torch.nn.Embedding` layer is replaced with a
            `optimum.graphcore.modeling_utils.SerializedEmbedding` layer.
        recompute_checkpoint_every_layer (`bool`, *optional*, defaults to `False`):
            If `True`, uses gradient checkpointing at the end of every layer. It can help to reduce the memory impact.

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
                logger.debug(f"ManagedAttribute {self.attr} writing to {obj.mode}_{self.attr}")
                return setattr(obj, f"{obj.mode}_{self.attr}", value)

        def __get__(self, obj, objtype=None):
            if isinstance(obj, IPUConfig):
                logger.debug(f"ManagedAttribute {self.attr} reading from {obj.mode}_{self.attr}")
                return getattr(obj, f"{obj.mode}_{self.attr}")

    # Create descriptor based managed attributes which will either return the
    # `training_` or `inference_` versions of the attribute depending on the value of
    # `self.mode` ("training" by default)
    modes = ("training", "inference")
    layers_per_ipu = ManagedAttribute("layers_per_ipu")
    ipus_per_replica = ManagedAttribute("ipus_per_replica")
    matmul_proportion = ManagedAttribute("matmul_proportion")
    replication_factor = ManagedAttribute("replication_factor")

    # Create a mapping of attribute value validating functions to a set of attributes
    # to be validated by that function
    attribute_validators = dict()

    def contents_geq_value_validator(
        name: str, value: Union[float, int, Sequence], floor_value: Union[float, int]
    ) -> None:
        """
        Validates the values of Sequence and scalar types to be greater than `floor_value`
        For Sequence[Union[int, float]], ensure that all elements are >= floor_value
        For Union[float, int], ensure the scalar is >= floor_value
        """

        if isinstance(value, Sequence):
            if not all([elem >= floor_value for elem in value]):
                raise ValueError(
                    f"`IPUConfig` attribute `{name}` must have all elements >= {floor_value}. You provided {value=}"
                )
        elif isinstance(value, (int, float)):
            if not value >= floor_value:
                raise ValueError(f"`IPUConfig` attribute `{name}` must be >= {floor_value}. You provided {value=}")
        else:
            raise ValueError(
                f"`contents_geq_value_validator` validates inputs of type:"
                f" Union[float, int, Sequence[Union[int, float]]]. You provided: {value=},{type(value)}"
            )

    attribute_validators[partial(contents_geq_value_validator, floor_value=1)] = {
        "training_replication_factor",
        "inference_replication_factor",
        "gradient_accumulation_steps",
        "training_ipus_per_replica",
        "inference_ipus_per_replica",
        "embedding_serialization_factor",
        "device_iterations",
        "inference_device_iterations",
    }

    attribute_validators[partial(contents_geq_value_validator, floor_value=0)] = {
        "training_matmul_proportion",
        "inference_matmul_proportion",
    }

    attribute_validators[partial(contents_geq_value_validator, floor_value=-1)] = {
        "training_layers_per_ipu",
        "inference_layers_per_ipu",
    }

    def output_mode_validator(name: str, value: str):
        allowed_values = ("all", "sum", "final", "default")
        if value not in allowed_values:
            raise ValueError(
                f"`IPUConfig` attribute `output_mode` can only take values in"
                f" {allowed_values}. You provided: {value=}"
            )

    attribute_validators[output_mode_validator] = {"output_mode"}

    def __init__(
        self,
        replication_factor: int = 1,
        training_replication_factor: Optional[int] = None,
        inference_replication_factor: int = 1,
        gradient_accumulation_steps: int = 1,
        layers_per_ipu: List[int] = [-1],
        training_layers_per_ipu: Optional[List[int]] = None,
        inference_layers_per_ipu: Optional[List[int]] = None,
        ipus_per_replica: Optional[int] = None,
        training_ipus_per_replica: Optional[int] = None,
        inference_ipus_per_replica: Optional[int] = None,
        optimizer_state_offchip: bool = False,
        replicated_tensor_sharding: bool = False,
        matmul_proportion: Union[float, List[float]] = 0.2,
        training_matmul_proportion: Optional[Union[float, List[float]]] = None,
        inference_matmul_proportion: Optional[Union[float, List[float]]] = None,
        enable_half_partials: bool = True,
        embedding_serialization_factor: int = 1,
        recompute_checkpoint_every_layer: bool = False,
        device_iterations: int = 1,
        inference_device_iterations: int = 1,
        output_mode: str = "final",
        seed: Optional[int] = None,
        auto_loss_scaling: bool = False,
        executable_cache_dir: str = "",
        **kwargs,
    ):
        self.seed = seed

        # Default mode to `training`
        self.train()

        if ipus_per_replica is None:
            ipus_per_replica = len(layers_per_ipu)

        self.training_layers_per_ipu = training_layers_per_ipu if training_layers_per_ipu else layers_per_ipu
        self.inference_layers_per_ipu = inference_layers_per_ipu if inference_layers_per_ipu else layers_per_ipu

        def init_ipus_per_replica(mode_ipus_per_replica, mode_layers_per_ipu):
            fallback_value = ipus_per_replica
            # if ipus_per_replica is default, recalculate ipus_per_replica from {mode}_layers_per_ipu instead
            if ipus_per_replica == len(layers_per_ipu):
                fallback_value = len(mode_layers_per_ipu)
            return mode_ipus_per_replica if mode_ipus_per_replica else fallback_value

        self.training_ipus_per_replica = init_ipus_per_replica(training_ipus_per_replica, self.training_layers_per_ipu)
        self.inference_ipus_per_replica = init_ipus_per_replica(
            inference_ipus_per_replica, self.inference_layers_per_ipu
        )

        def init_mode_matmul_proportion(mode_matmul_proportion, mode_ipus_per_replica):
            fallback_value = matmul_proportion
            # if matmul_proportion is a list and its length is not equal to {mode}_ipus_per_replica, use the
            # default float value for matmul_proportion instead
            if isinstance(matmul_proportion, list) and len(matmul_proportion) != mode_ipus_per_replica:
                fallback_value = 0.2
            return mode_matmul_proportion if mode_matmul_proportion else fallback_value

        self.training_matmul_proportion = init_mode_matmul_proportion(
            training_matmul_proportion, self.training_ipus_per_replica
        )
        self.inference_matmul_proportion = init_mode_matmul_proportion(
            inference_matmul_proportion, self.inference_ipus_per_replica
        )

        def check_and_set_replication_factor(attr_name, attr, default=False):
            if isinstance(attr, int):
                setattr(self, attr_name, attr)
            elif isinstance(attr, dict):
                base_message = (
                    f"Dictionary values for `{attr_name}`"
                    " have been deprecated. Provide values of type `int` instead."
                )

                if "default" not in attr:
                    raise ValueError(
                        base_message + f" Attempted to use the `default`"
                        f" replication factor in `{attr_name}={attr}"
                        " however no such key exists."
                    )

                try:
                    setattr(self, attr_name, attr.get("default"))
                except TypeError as e:
                    raise TypeError(
                        base_message + f" `Attempted to set"
                        f" `{attr_name}` using the `default` key of `{attr_name}"
                        " but a TypeError was raised. Check the error traceback for more information."
                    ) from e

                warnings.warn(base_message, FutureWarning, stacklevel=2)
            else:
                raise ValueError(f"{attr_name} must be of type `int`. You provided: {attr_name}={attr}, {type(attr)}.")

        check_and_set_replication_factor(
            "training_replication_factor",
            training_replication_factor if training_replication_factor else replication_factor,
        )
        check_and_set_replication_factor("inference_replication_factor", inference_replication_factor)

        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.device_iterations = device_iterations
        self.inference_device_iterations = inference_device_iterations
        self.optimizer_state_offchip = optimizer_state_offchip
        self.replicated_tensor_sharding = replicated_tensor_sharding
        self.auto_loss_scaling = auto_loss_scaling

        if "sharded_execution_for_inference" in kwargs:
            warnings.warn(
                'The "sharded_execution_for_inference" parameter is deprecated, sharded execution is always used during inference'
            )

        if "enable_half_first_order_momentum" in kwargs:
            warnings.warn('The "enable_half_first_order_momentum" parameter is deprecated')

        self.enable_half_partials = enable_half_partials
        self.executable_cache_dir = executable_cache_dir
        self.embedding_serialization_factor = embedding_serialization_factor
        self.recompute_checkpoint_every_layer = recompute_checkpoint_every_layer
        self.output_mode = output_mode
        # TODO: remove this if unnecessary.
        self.execute_encoder_on_cpu_for_generation = kwargs.pop("execute_encoder_on_cpu_for_generation", False)

        self.validate_ipu_config()

    @property
    def mode(self):
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
        if hasattr(self, attr) and hasattr(self, f"inference_{attr}") and hasattr(self, f"training_{attr}"):
            return f"training_{attr}" if self.mode == "training" else f"inference_{attr}"
        else:
            warnings.warn(f"{attr} is not a `ManagedAttribute`. Returning modeless name: {attr}")
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

        # Run additional attribute value validators
        for func, attrs in self.attribute_validators.items():
            if name in attrs:
                func(name, value)

        return super().__setattr__(name, value)

    def validate_ipu_config(self):
        """
        Tests coherence of `IPUConfig` attributes for all modes
        in self.modes. For example if `matmul_proportion=[0.2, 0.2]`,
        `ipus_per_replica` must have value 2.

        Raises:
            IncompatibleIPUConfigError: Raised if any `IPUConfig` attributes
            are not coherent.
        """
        if self.replicated_tensor_sharding and self.replication_factor == 1:
            logger.warning("Setting replicated_tensor_sharding to False when replication_factor=1")
            self.replicated_tensor_sharding = False

        old_mode = self.mode
        for mode in self.modes:
            self.mode = mode

            ipus_per_replica_mode_str = self._get_managed_attr_mode_name("ipus_per_replica")

            # len(matmul_proportion) must equal ipus_per_replica
            if isinstance(self.matmul_proportion, list) and len(self.matmul_proportion) != self.ipus_per_replica:
                matmul_proportion_mode_str = self._get_managed_attr_mode_name("matmul_proportion")
                raise IncompatibleIPUConfigError(
                    f"{matmul_proportion_mode_str}={self.matmul_proportion} should use the"
                    f" same number of IPUs as {ipus_per_replica_mode_str}={self.ipus_per_replica}"
                )

            # layers_per_ipu must have the same length as ipus per replica.
            # If wildcards are present in layers_per_ipu, let the call to `model.parallelize`
            # handle the validation
            if -1 not in self.layers_per_ipu and len(self.layers_per_ipu) != self.ipus_per_replica:
                layers_per_ipu_mode_str = self._get_managed_attr_mode_name("layers_per_ipu")
                raise IncompatibleIPUConfigError(
                    f"{layers_per_ipu_mode_str}={self.layers_per_ipu} should use the"
                    f" same number of IPUs as {ipus_per_replica_mode_str}={self.ipus_per_replica}"
                )

        self.mode = old_mode

    def _to_options(self, for_inference: bool = False, compile_only: bool = False) -> poptorch.Options:
        if not compile_only and poptorch.ipuHardwareVersion() != 2:
            raise RuntimeError("This requires an IPU Mk2 system to run.")

        if self.execute_encoder_on_cpu_for_generation:
            raise NotImplementedError("execute_encoder_on_cpu_for_generation is not supported yet.")

        old_mode = self.mode
        self.eval() if for_inference else self.train()

        opts = Options()
        opts.autoRoundNumIPUs(True)
        opts.replicationFactor(self.replication_factor)
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
        # with optimizer state residing either on-chip or in DRAM
        opts.TensorLocations.setOptimizerLocation(
            poptorch.TensorLocationSettings()
            # Optimizer state lives on- or off-chip
            .useOnChipStorage(not self.optimizer_state_offchip)
            # Shard optimizer state between replicas with zero-redundancy
            .useReplicatedTensorSharding(self.replicated_tensor_sharding)
        )

        if for_inference:
            opts.setExecutionStrategy(poptorch.ShardedExecution(poptorch.AutoStage.AutoIncrement))
        else:
            # Use Pipelined Execution
            opts.setExecutionStrategy(poptorch.PipelinedExecution(poptorch.AutoStage.AutoIncrement))

        # Compile offline (no IPUs required)
        if compile_only:
            opts.useOfflineIpuTarget()

        matmul_proportion = copy.deepcopy(self.matmul_proportion)
        if isinstance(self.matmul_proportion, float):
            matmul_proportion = [self.matmul_proportion]
        if len(matmul_proportion) == 1:
            matmul_proportion = matmul_proportion * self.ipus_per_replica
        mem_prop = {f"IPU{i}": matmul_proportion[i] for i in range(self.ipus_per_replica)}
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
        replication_factor = self.inference_replication_factor if for_inference else self.training_replication_factor
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
