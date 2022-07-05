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
from typing import Any, Dict, Optional, Union

import torch

import popart
import popdist
import poptorch
from optimum.configuration_utils import BaseConfig
from optimum.utils import logging
from poptorch import Options, OutputMode


logger = logging.get_logger(__name__)

IPU_CONFIG_NAME = "ipu_config.json"
ALLOWED_POD_TYPES = ["pod4", "pod8", "pod16", "pod32", "pod64"]


class IPUConfig(BaseConfig):
    CONFIG_NAME = "ipu_config.json"
    FULL_CONFIGURATION_FILE = "ipu_config.json"

    def __init__(self, **kwargs):
        self.use_popdist = kwargs.pop("use_popdist", False)
        self.seed = kwargs.pop("seed", None)

        self.ipus_per_replica = kwargs.pop("ipus_per_replica", 1)
        self.layers_per_ipu = kwargs.pop("layers_per_ipu", [1])

        self.replication_factor = kwargs.pop("replication_factor", 1)
        self.inference_replication_factor = kwargs.pop("inference_replication_factor", 1)
        self.gradient_accumulation_steps = kwargs.pop("gradient_accumulation_steps", 1)
        self.device_iterations = kwargs.pop("device_iterations", 1)
        self.inference_device_iterations = kwargs.pop("inference_device_iterations", 1)
        self.optimizer_state_offchip = kwargs.pop("optimizer_state_offchip", True)
        self.replicated_tensor_sharding = kwargs.pop("replicated_tensor_sharding", False)
        self.decompose_grad_sum = kwargs.pop("decompose_grad_sum", False)

        if self.replicated_tensor_sharding and self.replication_factor == 1:
            logger.warning("Setting replicated_tensor_sharding to False when replication_factor=1")
            self.replicated_tensor_sharding = False

        self.sharded_execution_for_inference = kwargs.pop("sharded_execution_for_inference", False)

        self.matmul_proportion = kwargs.pop("matmul_proportion", 0.6)

        self.enable_half_first_order_momentum = kwargs.pop("enable_half_first_order_momentum", False)
        self.enable_half_partials = kwargs.pop("enable_half_partials", False)

        self.executable_cache_dir = kwargs.pop("executable_cache_dir", "")
        self.profile_dir = kwargs.pop("profile_dir", "")

        self.embedding_serialization_factor = kwargs.pop("embedding_serialization_factor", 1)

        self.recompute_checkpoint_every_layer = kwargs.pop("recompute_checkpoint_every_layer", False)
        self.output_mode = kwargs.pop("output_mode", "final")

        self.execute_encoder_on_cpu_for_generation = kwargs.pop("execute_encoder_on_cpu_for_generation", False)

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
        Creates an IPUConfig specialized for a POD type.

        Args:
            pod_type: The POD type. If left to None, either the default value or the lowest value will be used for each
                configuration field.

        Returns:
            The IPUConfig instance.
        """
        config_dict = self.to_dict()
        config_dict = {k: self._prepare_config_attribute_for_pod_type(k, v, pod_type) for k, v in config_dict.items()}
        return IPUConfig(**config_dict)

    def _to_options(self, for_inference: bool = False, compile_only: bool = False) -> poptorch.Options:
        if not compile_only and poptorch.ipuHardwareVersion() != 2:
            raise RuntimeError("This requires an IPU Mk2 system to run.")

        if self.execute_encoder_on_cpu_for_generation:
            raise NotImplementedError("execute_encoder_on_cpu_for_generation is not supported yet.")

        opts = Options()

        # Define a policy to make sure LayerNorm related ops are computed in fp32
        fp32 = [torch.mean, torch.rsqrt, torch.pow, torch.add]
        policy = poptorch.autocasting.Policy([], fp32, [], [])
        opts.Precision.autocastPolicy(policy)

        opts.replicationFactor(self.inference_replication_factor if for_inference else self.replication_factor)

        opts.autoRoundNumIPUs(True)
        opts.deviceIterations(self.inference_device_iterations if for_inference else self.device_iterations)

        if not for_inference:
            # Set gradient accumulation factor
            opts.Training.gradientAccumulation(self.gradient_accumulation_steps)
            opts.Training.accumulationAndReplicationReductionType(poptorch.ReductionType.Mean)

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
        #  with optimizer state residing either on-chip or in DRAM
        opts.TensorLocations.setOptimizerLocation(
            poptorch.TensorLocationSettings()
            # Optimizer state lives on- or off-chip
            .useOnChipStorage(not self.optimizer_state_offchip)
            # Shard optimizer state between replicas with zero-redundancy
            .useReplicatedTensorSharding(self.replicated_tensor_sharding)
        )

        if for_inference and self.sharded_execution_for_inference:
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

        # Enable stochastic rounding (recommended for training with FP16)
        opts.Precision.enableStochasticRounding(not for_inference)

        # Half precision partials for matmuls and convolutions
        if self.enable_half_partials:
            opts.Precision.setPartialsType(torch.float16)

        # PopART performance options #
        # Replaces single sums of partial gradients with a tree of additions
        opts._Popart.set("decomposeGradSum", self.decompose_grad_sum)
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

        if self.profile_dir:
            engine_options = {
                **engine_options,
                **{
                    "debug.allowOutOfMemory": "true",
                    "autoReport.directory": self.profile_dir,
                    "autoReport.all": "true",
                },
            }
        opts._Popart.set("engineOptions", engine_options)

        return opts

    def to_options(
        self, for_inference: bool = False, compile_only: bool = False, pod_type: Optional[str] = None
    ) -> poptorch.Options:
        """
        Creates a poptorch.Options from the IPUConfig.

        Args:
            for_inference: If True, the resulting poptorch.Options will be adapted inference, it will be adapted for
                training otherwise.
            compile_only: If True, compilation will be performed offline, no IPUs required.
            pod_type: The POD type to specialize the poptorch.Options for.

        Returns:
            The poptorch.Options instance.
        """
        return self.for_pod_type(pod_type)._to_options(for_inference=for_inference, compile_only=compile_only)

    def batch_size_factor(self, for_inference: bool = False, pod_type: Optional[str] = None) -> int:
        """
        Computes the factor to apply to the micro batch size to get the combined batch size.

        Args:
            for_inference: Whether the factor is being use to compute the batch size for inference or not.

        Returns:
            The batch size factor.
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
        use `[a b c d]`. For example: "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index,
        matmul_proportion=[0.08 0.2 0.25 0.25]"

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
