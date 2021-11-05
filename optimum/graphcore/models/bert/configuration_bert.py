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

import os
import sys
import yaml
import argparse
import ctypes

import torch
import poptorch
import popart
import popdist
import popdist.poptorch
import horovod.torch as hvd
import numpy as np

from ...utils import logger


def get_options(config):
    """
    Set ipu specific options for the model, see documentation:
    https://docs.graphcore.ai/en/latest/
    """

    if not config.compile_only and poptorch.ipuHardwareVersion() != 2:
        raise RuntimeError("This version of BERT requires an IPU Mk2 system to run.")

    # Poptorch options
    if config.use_popdist:
        # Use popdist.poptorch options if running in distributed mode
        opts = popdist.poptorch.Options(ipus_per_replica=config.ipus_per_replica)
    else:
        opts = poptorch.Options()
        # Set the replication factor
        opts.replicationFactor(config.replication_factor)

    opts.autoRoundNumIPUs(True)
    opts.deviceIterations(config.batches_per_step)

    # Set gradient accumulation factor
    opts.Training.gradientAccumulation(config.gradient_accumulation)
    opts.Training.accumulationAndReplicationReductionType(poptorch.ReductionType.Mean)

    # Return all results from IPU to host
    opts.anchorMode(poptorch.AnchorMode.All)

    # Fix the random seeds
    np.random.seed(config.random_seed)
    opts.randomSeed(config.random_seed)

    # Enable Replicated Tensor Sharding (RTS) of optimizer state
    #  with optimizer state residing either on-chip or in DRAM
    opts.TensorLocations.setOptimizerLocation(
        poptorch.TensorLocationSettings()
        # Optimizer state lives on- or off-chip
        .useOnChipStorage(not config.optimizer_state_offchip)
        # Shard optimizer state between replicas with zero-redundancy
        .useReplicatedTensorSharding(config.replicated_tensor_sharding))

    # Use Pipelined Execution
    opts.setExecutionStrategy(
        poptorch.PipelinedExecution(poptorch.AutoStage.AutoIncrement))

    # Compile offline (no IPUs required)
    if config.compile_only:
        opts.useOfflineIpuTarget()

    # Set available Transient Memory For matmuls and convolutions operations
    mem_prop = {
        f'IPU{i}': config.matmul_proportion[i]
        for i in range(config.ipus_per_replica)
    }
    opts.setAvailableMemoryProportion(mem_prop)

    # Enable caching the compiled executable to disk
    if config.executable_cache_dir:
        opts.enableExecutableCaching(config.executable_cache_dir)

    # Enable stochastic rounding (recommended for training with FP16)
    opts.Precision.enableStochasticRounding(True)

    # Half precision partials for matmuls and convolutions
    if config.enable_half_partials:
        opts.Precision.setPartialsType(torch.float16)

    # Enable synthetic random data generated on device (so with no I/O)
    if config.synthetic_data:
        opts.enableSyntheticData(int(popart.SyntheticDataMode.RandomNormal))

    # PopART performance options #
    # Only stream needed tensors back to host
    opts._Popart.set("disableGradAccumulationTensorStreams", True)
    # Parallelize optimizer step update across IPUs
    opts._Popart.set("accumulateOuterFragmentSettings.schedule",
                     int(popart.AccumulateOuterFragmentSchedule.OverlapMemoryOptimized))
    opts._Popart.set("accumulateOuterFragmentSettings.excludedVirtualGraphs", ["0"])
    # Enable patterns for better throughput and memory reduction
    opts._Popart.set("subgraphCopyingStrategy", int(popart.SubgraphCopyingStrategy.JustInTime))
    opts._Popart.set("scheduleNonWeightUpdateGradientConsumersEarly", True)
    opts._Popart.setPatterns({"TiedGather": True, "TiedGatherAccumulate": True, "UpdateInplacePrioritiesForIpu": True})

    # Options for profiling with Popvision
    engine_options = {
        "opt.useAutoloader": "true",
        "target.syncReplicasIndependently": "true",
    }
    if config.profile_dir:
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



config_file = os.path.join(os.path.dirname(__file__), "configs.yml")


def str_to_bool(value):
    if isinstance(value, bool) or value is None:
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise argparse.ArgumentTypeError(f'{value} is not a valid boolean value')


def init_popdist(args):
    hvd.init()
    args.use_popdist = True
    if popdist.getNumTotalReplicas() != args.replication_factor:
        print(f"The number of replicas is overridden by PopRun. "
              f"The new value is {popdist.getNumTotalReplicas()}.")
    args.replication_factor = int(popdist.getNumLocalReplicas())
    args.popdist_rank = popdist.getInstanceIndex()
    args.popdist_size = popdist.getNumInstances()


def parse_bert_args(args=None):
    pparser = argparse.ArgumentParser("BERT Configuration name", add_help=False)
    pparser.add_argument("--config",
                         type=str,
                         help="Configuration Name",
                         default='demo_tiny_128')
    pargs, remaining_args = pparser.parse_known_args(args=args)
    config_name = pargs.config

    parser = argparse.ArgumentParser(
        "Poptorch BERT",
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Execution
    parser.add_argument("--batch-size", type=int, help="Set the micro batch-size")
    parser.add_argument("--training-steps", type=int, help="Number of training steps")
    parser.add_argument("--batches-per-step", type=int, help="Number of batches per training step")
    parser.add_argument("--replication-factor", type=int, help="Number of replicas")
    parser.add_argument("--gradient-accumulation", type=int, help="Number of gradients to accumulate before updating the weights")
    parser.add_argument("--embedding-serialization-factor", type=int, help="Matmul serialization factor the embedding layers")
    parser.add_argument("--recompute-checkpoint-every-layer", type=str_to_bool, nargs="?", const=True, default=False,
                        help="This controls how recomputation is handled in pipelining. "
                        "If True the output of each encoder layer will be stashed keeping the max liveness "
                        "of activations to be at most one layer. "
                        "However, the stash size scales with the number of pipeline stages so this may not always be beneficial. "
                        "The added stash + code could be greater than the reduction in temporary memory.",)
    parser.add_argument("--enable-half-partials", type=str_to_bool, nargs="?", const=True, default=False,
                        help="Enable half partials for matmuls and convolutions globally")
    parser.add_argument("--optimizer-state-offchip", type=str_to_bool, nargs="?", const=True, default=True,
                        help="Set the tensor storage location for optimizer state to be offchip.")
    parser.add_argument("--replicated-tensor-sharding", type=str_to_bool, nargs="?", const=True, default=False,
                        help="Enable replicated tensor sharding of optimizer state")
    parser.add_argument("--ipus-per-replica", type=int, help="Number of IPUs required by each replica")
    parser.add_argument("--layers-per-ipu", type=int, nargs="+",
                        help="Number of encoders placed on each IPU. Can be a single number, for an equal number encoder layers per IPU.\
                              Or it can be a list of numbers, specifying number of encoder layers for each individual IPU.")
    parser.add_argument("--matmul-proportion", type=float, nargs="+", help="Relative IPU memory proportion size allocated for matmul")
    parser.add_argument("--async-dataloader", type=str_to_bool, nargs="?", const=True, default=True,
                        help="Enable asynchronous mode in the DataLoader")
    parser.add_argument("--random-seed", type=int, help="Seed for RNG")
    parser.add_argument("--num-epochs", type=int, help="SQuAD only - number of epochs to train for")

    # Optimizer
    parser.add_argument("--optimizer", type=str, choices=['AdamW', 'LAMB', 'LAMBNoBiasCorrection'], help="optimizer to use for the training")
    parser.add_argument("--learning-rate", type=float, help="Learning rate value for constant schedule, maximum for linear schedule.")
    parser.add_argument("--lr-schedule", type=str, choices=["constant", "linear"],
                        help="Type of learning rate schedule. --learning-rate will be used as the max value")
    parser.add_argument("--lr-warmup", type=float, help="Proportion of lr-schedule spent in warm-up. Number in range [0.0, 1.0]")
    parser.add_argument("--loss-scaling", type=float, help="Loss scaling factor (recommend using powers of 2).\
                             If using automatic loss scaling, this value will be the initial value.")
    parser.add_argument("--weight-decay", type=float, help="Set the weight decay")
    parser.add_argument("--enable-half-first-order-momentum", type=str_to_bool, nargs="?", const=True, default=False,
                        help="Use float16 for the first order momentum in the optimizer.")
    parser.add_argument("--squad-do-training", type=str_to_bool, nargs="?", const=True, default=True,
                        help="Do SQuAD training (run_squad only)")
    parser.add_argument("--squad-do-validation", type=str_to_bool, nargs="?", const=True, default=True,
                        help="Do SQuAD validation (run_squad only)")

    # Model
    parser.add_argument("--sequence-length", type=int, help="The max sequence length")
    parser.add_argument("--mask-tokens", type=int, help="Set the max number of MLM tokens in the input dataset.")
    parser.add_argument("--vocab-size", type=int, help="Set the size of the vocabulary")
    parser.add_argument("--hidden-size", type=int, help="The size of the hidden state of the transformer layers")
    parser.add_argument("--intermediate-size", type=int, help="hidden-size*4")
    parser.add_argument("--num-hidden-layers", type=int, help="The number of transformer layers")
    parser.add_argument("--num-attention-heads", type=int, help="Set the number of heads in self attention")
    parser.add_argument("--layer-norm-eps", type=float, help="The eps value for the layer norms")

    # Hugging Face specific
    parser.add_argument("--attention-probs-dropout-prob", type=float, nargs="?", const=True, help="Attention dropout probability")

    # Dataset
    parser.add_argument("--input-files", type=str, nargs="+", help="Input data files")
    parser.add_argument("--dataset", type=str, choices=['generated', 'pretraining'],
                        help="dataset to use for the training")
    parser.add_argument("--synthetic-data", type=str_to_bool, nargs="?", const=True, default=False,
                        help="No Host/IPU I/O, random data created on device")

    # Misc
    parser.add_argument("--dataloader-workers", type=int, help="The number of dataloader workers")
    parser.add_argument("--profile-dir", type=str, help="Enable profiling and store results in this directory")
    parser.add_argument("--wandb", type=str_to_bool, nargs="?", const=True, default=False,
                        help="Enabling logging to Weights and Biases")
    parser.add_argument("--wandb-param-steps", type=int, default=None,
                        help="Log the model parameter statistics to Weights and Biases after every n training steps")
    parser.add_argument("--disable-progress-bar", type=str_to_bool, nargs="?", const=True, default=False,
                        help="Disable the training progress bar. This is useful if you want to parse the stdout of a run")
    parser.add_argument("--compile-only", type=str_to_bool, nargs="?", const=True, default=False,
                        help="Create an offline IPU target that can only be used for offline compilation.")
    parser.add_argument("--executable-cache-dir", type=str, default="",
                        help="Directory where Poplar executables are cached. If set, recompilation of identical graphs can be avoided. "
                        "Required for both saving and loading executables.")

    # Checkpointing
    parser.add_argument("--checkpoint-output-dir", type=str, default="", help="Directory where checkpoints will be saved to.\
                             This can be either an absolute or relative path.")
    parser.add_argument("--checkpoint-steps", type=int, default=None,
                        help="Option to checkpoint model after every n training steps.")
    parser.add_argument("--resume-training-from-checkpoint", type=str_to_bool, nargs="?", const=True, default=False,
                        help="Restore both the model checkpoint and training state in order to resume a training run.")
    parser.add_argument("--pretrained-checkpoint", type=str, default="", help="Checkpoint to be retrieved for further training. This can\
                              be either an absolute or relative path to the checkpoint directory or the name of a model on HuggingFace model hub.")

    # This is here only for the help message
    parser.add_argument("--config", type=str, help="Configuration name")

    # Load the yaml
    yaml_args = dict()
    if config_name is not None:
        with open(config_file, "r") as f:
            try:
                yaml_args.update(**yaml.safe_load(f)[config_name])
            except yaml.YAMLError as exc:
                print(exc)
                sys.exit(1)

    # Check the yaml args are valid
    known_args = set(vars(parser.parse_args("")))
    unknown_args = set(yaml_args) - known_args

    if unknown_args:
        logger(f" Warning: Unknown arg(s) in config file: {unknown_args}")

    parser.set_defaults(**yaml_args)
    remaining_args = ""
    args = parser.parse_args(remaining_args)

    # Initialise PopDist
    if popdist.isPopdistEnvSet():
        init_popdist(args)
        hvd.broadcast(torch.Tensor([args.random_seed]), root_rank=0)
    else:
        args.use_popdist = False

    # Expand layers_per_ipu input into list representation
    if isinstance(args.layers_per_ipu, int):
        args.layers_per_ipu = [args.layers_per_ipu]

    if len(args.layers_per_ipu) == 1:
        layers_per_ipu_ = args.layers_per_ipu[0]
        args.layers_per_ipu = [layers_per_ipu_] * (args.num_hidden_layers // layers_per_ipu_)

    if sum(args.layers_per_ipu) != args.num_hidden_layers:
        parser.error(f"layers_per_ipu not compatible with number of hidden layers: {args.layers_per_ipu} and {args.num_hidden_layers}")

    # Expand matmul_proportion input into list representation
    if isinstance(args.matmul_proportion, float):
        args.matmul_proportion = [args.matmul_proportion] * args.ipus_per_replica

    if len(args.matmul_proportion) != args.ipus_per_replica:
        if len(args.matmul_proportion) == 1:
            args.matmul_proportion = args.matmul_proportion * args.ipus_per_replica
        else:
            parser.error(f"Length of matmul_proportion doesn't match ipus_per_replica: {args.matmul_proportion} vs {args.ipus_per_replica}")

    if args.checkpoint_steps is not None and args.checkpoint_steps < 1:
        parser.error("checkpoint-steps must be >=1")

    if args.use_popdist:
        args.global_batch_size = args.replication_factor * args.gradient_accumulation * args.batch_size * args.popdist_size
    else:
        args.global_batch_size = args.replication_factor * args.gradient_accumulation * args.batch_size

    args.samples_per_step = args.replication_factor * args.gradient_accumulation * args.batch_size * args.batches_per_step
    args.intermediate_size = args.hidden_size * 4

    return args
