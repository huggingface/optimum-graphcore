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
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union, get_type_hints

import torch

import popart
import poptorch
import typeguard
from optimum.configuration_utils import BaseConfig
from optimum.utils import logging
from poptorch import Options, OutputMode
from typing_extensions import TypeAlias


logger = logging.get_logger(__name__)

IPU_CONFIG_NAME = "ipu_config.json"
ALLOWED_POD_TYPES = ["pod4", "pod8", "pod16", "pod32", "pod64"]


class IPUConfigAttributeTypes:
    T = TypeVar("T")
    Type: TypeAlias = Union[T, Dict[str, T]]
    OptionalType = Union[Optional[T], Dict[str, Optional[T]]]


@typeguard.typechecked
class IPUConfig(BaseConfig):
    """
    Class for PopArt and PopTorch configuration. Handles the conversion to poptorch options as well as configuration
    pod type specialization.

    Args:
        seed (`int`, *optional*):
            Sets the seed for the random number generator on the IPU.
        auto_loss_scaling (`bool`, *optional*, defaults to `False`):
            Whether automatic loss scaling is enabled on the IPU.
            When using float16/half values for activations, gradients, and weights, the loss value needs to be scaled by
            a constant factor to avoid underflow/overflow. This adjustment is known as loss scaling. This setting
            automatically sets a global loss scaling factor during training.
            **Note: This is an experimental feature and may not behave as expected.**
        executable_cache_dir (`str`, *optional*, defaults to `""`):
            Enables caching the compile executables to a directory.

        > Parameters for controlling the batch size

        training_replication_factor (`int`, *optional*, defaults to 1):
            The number of replicas for data-parallelism during training. It depends on the size of the pipeline as well
            as the number of IPUs available. For example: on a Pod16, with a 4-IPU pipeline, replication_factor must
            be betwen 1 and 4.
        inference_replication_factor (`int`, *optional*, defaults to 1):
            Same as `training_replication_factor` for inference.
        gradient_accumulation_steps (`int`, *optional*, defaults to 1):
            Number of micro-batches to accumulate for the gradient calculation.
            Accumulates the gradient gradient_accumulation times before updating the model using the gradient.

        > Parameters related to parallelism

        training_layers_per_ipu (`List[int]`):
            Specifies the number of layers that will be put on each IPU for pipelined execution during training.
            For instance: `[2, 3, 4, 2]` specifies a 4-IPU pipeline, where the first two layers will be put on IPU0,
            the following three on IPU1, the next four on IPU2 and the last two on IPU3.
            If the default of [-1] is used, the layers will be split evenly over `ipus_per_replica` IPUs.
            The wildcard value '-1' can also be used in combination with integers.
            For instance: `[1, 2, -1, -1]` specifies a 4-IPU pipeline, where the first layer is put on IPU0,
            the next two layers on IPU1, and the remaining layers split evenly between IPU2 and IPU3.
        inference_layers_per_ipu (`List[int]`, *optional*, defaults to `training_layers_per_ipu`):
            Same as `training_layers_per_ipu` for inference only.

        > Parameters for memory management

        optimizer_state_offchip (`bool`, *optional*, defaults to `True`):
            Whether to use the off chip memory to store the optimizer state or to use the on chip memory.
        replicated_tensor_sharding (`bool`, *optional*, defaults to `False`):
            Shards the optimizer between replicas with zero-redundancy.
        training_matmul_proportion (`List[float]` or `float`, *optional*, defaults to 0.2):
            Sets the amount of temporary memory made available on per-IPU basis during training.
            Use this setting to control the amount of temporary memory available to operations such as:
              - convolution
              - matrix multiplication
              - embedding lookups
              - indexing operations
        inference_matmul_proportion (`List[int]`, defaults to `training_matmul_proportion` if `inference_ipus_per_replica==len(training_matmul_proportion)`, otherwise defaults to `0.2` ):
            Same as `training_matmul_proportion` for inference only.
        enable_half_partials (`bool`, *optional*, defaults to `True`):
            Whether the data type of partial results for matrix multiplication and convolution operators should be
            float16 or not.
        embedding_serialization_factor (`int`, *optional*, defaults to 1):
            The factor to use to serialze embeddings. Nothing happens if `embedding_serialization_factor = 1`, and for
            `embedding_serialization_factor > 1`, the `torch.nn.Embedding` layer is replaced by a
            `optimum.graphcore.modeling_utils.SerializedEmbedding` layer.
        recompute_checkpoint_every_layer (`bool`, *optional*, defaults to `False`):
            Whether to use gradient checkpointing at the end of every layer. It can help in reducing the memory impact.

        > Parameters related to host / device synchronization

        training_device_iterations (`int`, *optional*, defaults to 1):
            Number of iterations the device should run over the data before returning to the user during training. This
            is equivalent to running the IPU in a loop over that the specified number of iterations, with a new batch of
            data each time. However, increasing deviceIterations is more efficient because the loop runs on the IPU
            directly.
        inference_device_iterations (`int`, *optional*, defaults to 1):
            Same as `training_device_iterations` for inference only.
        output_mode (`str`, *optional*, defaults to `"final"`):
            Specifies which data to return from a model.
            Allowed values:
              - `all`: returns a result for each batch.
              - `sum`: returns the sum of all batches.
              - `final`: returns the last batch.
              - `default`: `all` for inference, `final` for training.

    """

    CONFIG_NAME: str = "ipu_config.json"
    FULL_CONFIGURATION_FILE: str = "ipu_config.json"

    class ManagedAttribute:
        def __set_name__(self, owner, name) -> None:
            self.attr = name

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
    modes: Tuple[str] = ("training", "inference")

    layers_per_ipu = ManagedAttribute()
    ipus_per_replica = ManagedAttribute()
    matmul_proportion = ManagedAttribute()
    replication_factor = ManagedAttribute()
    device_iterations = ManagedAttribute()

    def __init__(
        self,
        seed: IPUConfigAttributeTypes.OptionalType[int] = None,
        auto_loss_scaling: IPUConfigAttributeTypes.Type[bool] = False,
        executable_cache_dir: IPUConfigAttributeTypes.Type[str] = "",
        training_replication_factor: IPUConfigAttributeTypes.Type[int] = 1,
        inference_replication_factor: IPUConfigAttributeTypes.Type[int] = 1,
        gradient_accumulation_steps: IPUConfigAttributeTypes.Type[int] = 1,
        training_layers_per_ipu: IPUConfigAttributeTypes.Type[List[int]] = [-1],
        inference_layers_per_ipu: IPUConfigAttributeTypes.OptionalType[List[int]] = None,
        training_ipus_per_replica: IPUConfigAttributeTypes.OptionalType[int] = None,
        inference_ipus_per_replica: IPUConfigAttributeTypes.OptionalType[int] = None,
        optimizer_state_offchip: IPUConfigAttributeTypes.Type[bool] = True,
        replicated_tensor_sharding: IPUConfigAttributeTypes.Type[bool] = False,
        training_matmul_proportion: IPUConfigAttributeTypes.Type[Union[float, List[float]]] = 0.2,
        inference_matmul_proportion: IPUConfigAttributeTypes.OptionalType[Union[float, List[float]]] = None,
        enable_half_partials: IPUConfigAttributeTypes.Type[bool] = True,
        embedding_serialization_factor: IPUConfigAttributeTypes.Type[int] = 1,
        recompute_checkpoint_every_layer: IPUConfigAttributeTypes.Type[bool] = True,
        training_device_iterations: IPUConfigAttributeTypes.Type[int] = 1,
        inference_device_iterations: IPUConfigAttributeTypes.Type[int] = 1,
        output_mode: IPUConfigAttributeTypes.Type[str] = "final",
        execute_encoder_on_cpu_for_generation: IPUConfigAttributeTypes.Type[bool] = False,
        **kwargs,
    ):
        self.seed = seed
        # Default mode to `training`
        self.train()

        def raise_deprecation_warning(attribute):
            if attribute in kwargs:
                warnings.warn(
                    f"`IPUConfig` parameter `{attribute}` will be removed in the next release. Use `training_{attribute}` instead.",
                    FutureWarning,
                    stacklevel=2,
                )

        raise_deprecation_warning("layers_per_ipu")
        self.training_layers_per_ipu = kwargs.pop("layers_per_ipu", training_layers_per_ipu)
        self.inference_layers_per_ipu = (
            inference_layers_per_ipu if inference_layers_per_ipu else self.training_layers_per_ipu
        )

        # initialise `ipus_per_replica` using `layers_per_ipu` if it is None
        raise_deprecation_warning("ipus_per_replica")
        self.training_ipus_per_replica = kwargs.pop("ipus_per_replica", training_ipus_per_replica)
        if self.training_ipus_per_replica is None:
            self.training_ipus_per_replica = len(self.training_layers_per_ipu)

        # If ipus_per_replica is default, recalculate inference_ipus_per_replica from inference_layers_per_ipu instead
        self.inference_ipus_per_replica = inference_ipus_per_replica
        if inference_ipus_per_replica is None:
            self.inference_ipus_per_replica = (
                len(self.inference_layers_per_ipu)
                if self.training_ipus_per_replica == self.training_layers_per_ipu
                else self.training_ipus_per_replica
            )

        # If matmul_proportion is a list and its length is not equal to {mode}_ipus_per_replica, use the
        # default float value for matmul_proportion instead
        raise_deprecation_warning("matmul_proportion")
        self.training_matmul_proportion = kwargs.pop("matmul_proportion", training_matmul_proportion)
        default_matmul_proportion = self.training_matmul_proportion
        if (
            isinstance(self.training_matmul_proportion, list)
            and len(self.training_matmul_proportion) != self.inference_ipus_per_replica
        ):
            default_matmul_proportion = 0.2
        self.inference_matmul_proportion = (
            inference_matmul_proportion if inference_matmul_proportion else default_matmul_proportion
        )

        self.embeedding_serialization_factor = embedding_serialization_factor

        raise_deprecation_warning("replication_factor")
        self.training_replication_factor = kwargs.pop("replication_factor", training_replication_factor)
        self.inference_replication_factor = inference_replication_factor
        self.gradient_accumulation_steps = gradient_accumulation_steps
        raise_deprecation_warning("device_iterations")
        self.training_device_iterations = kwargs.pop("device_iterations", training_device_iterations)
        self.inference_device_iterations = inference_device_iterations
        self.optimizer_state_offchip = optimizer_state_offchip
        self.replicated_tensor_sharding = replicated_tensor_sharding
        self.auto_loss_scaling = auto_loss_scaling

        if self.replicated_tensor_sharding and self.replication_factor == 1:
            logger.warning("Setting replicated_tensor_sharding to False when replication_factor=1")
            self.replicated_tensor_sharding = False

        if "sharded_execution_for_inference" in kwargs:
            warnings.warn(
                'The "sharded_execution_for_inference" parameter is deprecated, sharded execution is always used during inference'
            )
        if "enable_half_first_order_momentum" in kwargs:
            warnings.warn('The "enable_half_first_order_momentum" parameter is deprecated')

        self.enable_half_partials = enable_half_partials
        self.executable_cache_dir = executable_cache_dir
        self.recompute_checkpoint_every_layer = recompute_checkpoint_every_layer
        self.output_mode = output_mode
        # TODO: remove this if unnecessary
        self.execute_encoder_on_cpu_for_generation = execute_encoder_on_cpu_for_generation

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        assert (
            value in self.modes
        ), f"{value=} must be one of {self.modes}. Use the `train` and `eval` methods to set the ipu config mode to avoid error."
        self._mode = value

    def train(self):
        self._mode = "training"
        return self

    def eval(self):
        self._mode = "inference"
        return self

    def _get_attribute_type(self, name):
        try:
            attribute_type_hints = self.attribute_type_hints
        except AttributeError:
            attribute_type_hints = get_type_hints(IPUConfig.__init__)
            super().__setattr__("attribute_type_hints", attribute_type_hints)
        return attribute_type_hints.get(name, Any)

    def __setattr__(self, name, value):
        attribute_type = self._get_attribute_type(name)
        try:
            typeguard.check_type(value, attribute_type)
        except typeguard.TypeCheckError as e:
            raise TypeError(
                f"Setting attribute: {name}, type: {attribute_type} with value: {value}, type: {type(value)} is invalid"
            ) from e

        # TODO: add functionality to run additional validator functions for attributes
        super().__setattr__(name, value)

    def _prepare_config_attribute_for_pod_type(
        self, config_attribute_name: str, config_attribute: Union[Any, Dict[str, Any]], pod_type: Optional[str]
    ) -> Any:
        """
        Prepares a config attribute by extracting the proper value for this attribute considering the POD type.

        Args:
            config_attribute_name: The config attribute name (i.e. the name of the config field).
            config_attribute: The config attribute to extract the value from.
            pod_type: The POD type.

        Returns:
            The extracted config attribute value.
        """
        if not isinstance(config_attribute, dict) or not config_attribute.keys() <= (
            set(ALLOWED_POD_TYPES) | {"default"}
        ):
            return config_attribute

        if pod_type is None and "default" not in config_attribute:
            raise RuntimeError(
                f"No POD type was specified and no default value was provided for {config_attribute_name}, cannot infer"
                " which value to use"
            )
        elif pod_type is None:
            return config_attribute["default"]
        elif pod_type not in ALLOWED_POD_TYPES:
            raise ValueError(
                f"{pod_type} is not a correct value for a POD type, supported POD types: {', '.join(ALLOWED_POD_TYPES)}"
            )
        elif pod_type not in config_attribute:
            raise KeyError(
                f"the {config_attribute_name} configuration field does not contain a value for POD type {pod_type}"
            )
        else:
            return config_attribute[pod_type]

    def for_pod_type(self, pod_type: Optional[str] = None) -> "IPUConfig":
        """
        Creates an `IPUConfig` specialized for a POD type.

        Args:
            pod_type (`str`, *optional*):
                The POD type. If left to None, either the default value or the lowest value will be used for each
                configuration field.

        Returns:
            `IPUConfig`: The IPUConfig instance.
        """
        config_dict = self.to_dict()
        config_dict = {k: self._prepare_config_attribute_for_pod_type(k, v, pod_type) for k, v in config_dict.items()}
        return IPUConfig(**config_dict)

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
        opts.deviceIterations(self.device_iterations)

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

        # Enable Replicated Tensor Sharding (RTS) of optimizer state
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

    def to_options(
        self, for_inference: bool = False, compile_only: bool = False, pod_type: Optional[str] = None
    ) -> poptorch.Options:
        """
        Creates a `poptorch.Options` from the `IPUConfig`.

        Args:
            for_inference (`bool`, defaults to `False`):
                If True, the resulting poptorch.Options will be adapted inference, it will be adapted for training
                otherwise.
            compile_only (`bool`, defaults to `False`):
                If True, compilation will be performed offline, no IPUs required.
            pod_type (`str`, *optional*):
                The POD type to specialize the `poptorch.Options` for.

        Returns:
            `poptorch.Options`: The option representing the `IPUConfig`.
        """
        return self.for_pod_type(pod_type)._to_options(for_inference=for_inference, compile_only=compile_only)

    def to_dict(self) -> Dict[str, Any]:
        """
        Same as super().to_dict() but with the additional removal of private attributes

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """

        original_mode = self.mode

        self.train()
        output = super().to_dict()

        # revert private training attributes to match the public names
        new_output = {}
        prefix = "training_"
        for attr_name, attr_value in output.items():
            if attr_name.startswith(prefix):
                new_output[attr_name[len(prefix) :]] = attr_value
            else:
                new_output[attr_name] = attr_value

        self._mode = original_mode

        return new_output

    def batch_size_factor(self, for_inference: bool = False, pod_type: Optional[str] = None) -> int:
        """
        Computes the factor to apply to the micro batch size to get the combined batch size.

        Args:
            for_inference (`bool`, defaults to `False`):
                Whether the factor is being use to compute the batch size for inference or not.
            pod_type (`str`, *optional*):
                The pod type that is being used. This is needed because the batch size factor can be pod type dependent.

        Returns:
            `int`: The batch size factor.
        """
        ipu_config = self.for_pod_type(pod_type)
        replication_factor = (
            ipu_config.inference_replication_factor if for_inference else ipu_config.replication_factor
        )
        gradient_accumulation_steps = 1 if for_inference else ipu_config.gradient_accumulation_steps
        device_iterations = ipu_config.inference_device_iterations if for_inference else ipu_config.device_iterations
        return replication_factor * gradient_accumulation_steps * device_iterations

    def update_from_string(self, update_str: str):
        """
        Updates attributes of this class with attributes from `update_str`.

        The expected format is ints, floats and strings as is, and for booleans use `true` or `false`, and for lists
        use `[a b c d]`. For example: `"n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index,
        matmul_proportion=[0.08 0.2 0.25 0.25]"`.

        The keys to change have to already exist in the config object.

        Args:
            update_str (`str`): String with attributes that should be updated for this class.

        """

        d = dict(x.split("=") for x in update_str.split(","))
        for k, v in d.items():
            if not hasattr(self, k):
                raise ValueError(f"key {k} isn't in the original config dict")

            old_v = getattr(self, k)
            if isinstance(old_v, bool):
                if v.lower() in ["true", "1", "y", "yes"]:
                    v = True
                elif v.lower() in ["false", "0", "n", "no"]:
                    v = False
                else:
                    raise ValueError(f"can't derive true or false from {v} (key {k})")
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
