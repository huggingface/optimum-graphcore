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
from typing import Any, Dict, Optional, Union

import torch

import popart
import popdist
import poptorch
from optimum.configuration_utils import BaseConfig
from optimum.utils import logging
from poptorch import Options


logger = logging.get_logger(__name__)

IPU_CONFIG_NAME = "ipu_config.json"
ALLOWED_POD_TYPES = ["pod4", "pod16", "pod64", "pod128", "pod256"]


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

        self.matmul_proportion = kwargs.pop("matmul_proportion", 0.6)

        self.enable_half_first_order_momentum = kwargs.pop("enable_half_first_order_momentum", False)
        self.enable_half_partials = kwargs.pop("enable_half_partials", False)
        self.synthetic_data = kwargs.pop("synthetic_data", False)

        self.executable_cache_dir = kwargs.pop("executable_cache_dir", None)
        self.profile_dir = kwargs.pop("profile_dir", None)

        self.embedding_serialization_factor = kwargs.pop("embedding_serialization_factor", 1)

        self.recompute_checkpoint_every_layer = kwargs.pop("recompute_checkpoint_every_layer", False)

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

        # TODO: fix that with popdist.
        # if self.use_popdist:
        #     opts = popdist.poptorch.Options(ipus_per_replica=self.ipus_per_replica)
        # else:

        opts = Options()
        opts.replicationFactor(self.inference_replication_factor if for_inference else self.replication_factor)

        opts.autoRoundNumIPUs(True)
        opts.deviceIterations(self.inference_device_iterations if for_inference else self.device_iterations)

        if not for_inference:
            # Set gradient accumulation factor
            opts.Training.gradientAccumulation(self.gradient_accumulation_steps)
            opts.Training.accumulationAndReplicationReductionType(poptorch.ReductionType.Mean)

        # Return all results from IPU to host
        opts.outputMode(poptorch.OutputMode.All)

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
        if self.executable_cache_dir:
            opts.enableExecutableCaching(self.executable_cache_dir)

        # Enable stochastic rounding (recommended for training with FP16)
        opts.Precision.enableStochasticRounding(not for_inference)

        # Half precision partials for matmuls and convolutions
        if self.enable_half_partials:
            opts.Precision.setPartialsType(torch.float16)

        # Enable synthetic random data generated on device (so with no I/O)
        if self.synthetic_data:
            opts.enableSyntheticData(int(popart.SyntheticDataMode.RandomNormal))

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
