# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import poptorch
import popart
import popdist
import popdist.poptorch
import numpy as np
import ctypes
import os

from utils import logger


def get_options(config):
    '''
    Set ipu specific options for the model, see documentation:
    https://docs.graphcore.ai/en/latest/
    '''

    if not config.compile_only and poptorch.ipuHardwareVersion() != 2:
        raise RuntimeError("This version of BERT requires an IPU Mk2 system to run.")

    # Custom ops
    if config.custom_ops is True:
        file_dir = os.path.dirname(os.path.realpath(__file__))
        CUSTOM_OP_PATH = os.path.join(file_dir, "custom_ops.so")
        if os.path.exists(CUSTOM_OP_PATH):
            ops_and_patterns = ctypes.cdll.LoadLibrary(CUSTOM_OP_PATH)
            ops_and_patterns.setVocabSize(config.vocab_size)
            ops_and_patterns.setEmbeddingSize(config.hidden_size)
            ops_and_patterns.setHiddenSize(config.hidden_size)
        else:
            logger("Could not find custom_ops.so. Execute `make` before running this script.")
            exit()

    # Numpy options
    np.random.seed(config.random_seed)

    # Poptorch options
    if config.use_popdist:
        opts = popdist.poptorch.Options(ipus_per_replica=config.ipus_per_replica)
    else:
        opts = poptorch.Options()
        opts.replicationFactor(config.replication_factor)

    opts.autoRoundNumIPUs(True)
    opts.deviceIterations(config.batches_per_step)
    opts.Training.gradientAccumulation(config.gradient_accumulation)
    opts.Training.accumulationAndReplicationReductionType(poptorch.ReductionType.Mean)
    if config.auto_loss_scaling is True:
        opts.Training.setAutomaticLossScaling(True)
    opts.anchorMode(poptorch.AnchorMode.Sum)
    opts.TensorLocations.setOptimizerLocation(
        poptorch.TensorLocationSettings()
        .useOnChipStorage(not config.optimizer_state_offchip)
        .useReplicatedTensorSharding(config.replicated_tensor_sharding))
    opts.randomSeed(config.random_seed)
    opts.setExecutionStrategy(
        poptorch.PipelinedExecution(poptorch.AutoStage.AutoIncrement))
    if config.compile_only:
        opts.useOfflineIpuTarget()

    mem_prop = {
        f'IPU{i}': config.matmul_proportion[i]
        for i in range(config.ipus_per_replica)
    }
    opts.setAvailableMemoryProportion(mem_prop)
    if config.executable_cache_dir:
        opts.enableExecutableCaching(config.executable_cache_dir)

    # Precision options
    opts.Precision.enableStochasticRounding(True)
    if config.enable_half_partials:
        opts.Precision.setPartialsType(torch.float16)

    # PopART options
    opts._Popart.set("disableGradAccumulationTensorStreams", True)
    opts._Popart.set("subgraphCopyingStrategy", int(popart.SubgraphCopyingStrategy.JustInTime))
    opts._Popart.set("outlineThreshold", 10.0)
    opts._Popart.set("accumulateOuterFragmentSettings.schedule",
                     int(popart.AccumulateOuterFragmentSchedule.OverlapMemoryOptimized))
    opts._Popart.set("accumulateOuterFragmentSettings.excludedVirtualGraphs", ["0"])

    if config.synthetic_data:
        opts.enableSyntheticData(int(popart.SyntheticDataMode.RandomNormal))

    engine_options = {
        "opt.useAutoloader": "true",
        "target.syncReplicasIndependently": "true",
    }
    if config.profile:
        engine_options = {
            **engine_options,
            **{
                "debug.allowOutOfMemory": "true",
                "autoReport.directory": config.profile_dir,
                "profiler.format": "v3",
                "autoReport.all": "true",
            }
        }
    opts._Popart.set("engineOptions", engine_options)

    return opts
