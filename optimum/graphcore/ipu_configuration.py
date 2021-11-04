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

import os
from typing import Any, Dict, Tuple, Union

import popart
import popdist
import poptorch
import torch
import transformers
from poptorch import Options
from transformers import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

CONFIG_NAME = "ipu_config.json"


class IPUConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        self.use_popdist = kwargs.pop("use_popdist", False)
        self.compile_only = kwargs.pop("compile_only", False)
        self.random_seed = kwargs.pop("random_seed", None)

        self.ipus_per_replica = kwargs.pop("ipus_per_replica", 1)
        # TODO: invalid value for layers_per_ipu which must be a list.
        self.layers_per_ipu = kwargs.pop("layers_per_ipu", 1)

        self.replication_factor = kwargs.pop("replication_factor", 1)
        self.gradient_accumulation_steps = kwargs.pop("gradient_accumulation_steps", 1)
        self.device_iterations = kwargs.pop("device_iterations", 1)
        self.optimizer_state_offchip = kwargs.pop("optimizer_state_offchip", True)
        self.replicated_tensor_sharding = kwargs.pop("replicated_tensor_sharding", False)

        self.matmul_proportion = kwargs.pop("matmul_proportion", 0.6)
        if isinstance(self.matmul_proportion, float):
            self.matmul_proportion = [self.matmul_proportion]
        if len(self.matmul_proportion) == 1:
            self.matmul_proportion = self.matmul_proportion * self.ipus_per_replica

        self.enable_half_partials = kwargs.pop("enable_half_partials", False)
        self.synthetic_data = kwargs.pop("synthetic_data", False)

        self.executable_cache_dir = kwargs.pop("executable_cache_dir", None)
        self.profile_dir = kwargs.pop("profile_dir", None)

        self.embedding_serialization_factor = kwargs.pop("embedding_serialization_factor", 1)

        self.recompute_checkpoint_every_layer = kwargs.pop("recompute_checkpoint_every_layer", False)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        orig_transformers_config_name = transformers.file_utils.CONFIG_NAME
        transformers.configuration_utils.CONFIG_NAME = CONFIG_NAME
        super().save_pretrained(save_directory, push_to_hub=push_to_hub, **kwargs)
        transformers.configuration_utils.CONFIG_NAME = orig_transformers_config_name

    @classmethod
    def get_config_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        orig_transformers_config_name = transformers.file_utils.CONFIG_NAME
        transformers.configuration_utils.CONFIG_NAME = CONFIG_NAME
        ipu_config = super().get_config_dict(pretrained_model_name_or_path, **kwargs)
        transformers.configuration_utils.CONFIG_NAME = orig_transformers_config_name
        return ipu_config

    def to_options(self, for_inference=False) -> poptorch.Options:
        if not self.compile_only and poptorch.ipuHardwareVersion() != 2:
            raise RuntimeError("This version of BERT requires an IPU Mk2 system to run.")

        if self.use_popdist:
            opts = popdist.poptorch.Options(ipus_per_replica=self.ipus_per_replica)
        else:
            opts = Options()
            opts.replicationFactor(self.replication_factor)

        opts.autoRoundNumIPUs(True)
        opts.deviceIterations(self.device_iterations)

        if not for_inference:
            # Set gradient accumulation factor
            opts.Training.gradientAccumulation(self.gradient_accumulation_steps)
            opts.Training.accumulationAndReplicationReductionType(poptorch.ReductionType.Mean)

        # Return all results from IPU to host
        opts.anchorMode(poptorch.AnchorMode.All)

        if self.random_seed:
            opts.randomSeed(self.random_seed)

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
        if self.compile_only:
            opts.useOfflineIpuTarget()

        mem_prop = {f"IPU{i}": self.matmul_proportion[i] for i in range(self.ipus_per_replica)}
        opts.setAvailableMemoryProportion(mem_prop)

        # Enable caching the compiled executable to disk
        if self.executable_cache_dir:
            opts.enableExecutableCaching(self.executable_cache_dir)

        # Enable stochastic rounding (recommended for training with FP16)
        opts.Precision.enableStochasticRounding(True)

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
                    "profiler.format": "v3",
                    "autoReport.all": "true",
                },
            }
        opts._Popart.set("engineOptions", engine_options)

        return opts

    @property
    def batch_size_factor(self) -> int:
        return self.replication_factor * self.gradient_accumulation_steps * self.device_iterations
