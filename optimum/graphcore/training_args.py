# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import contextlib
import json
import math
import os
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from poptorch import DataLoaderMode
from transformers.debug_utils import DebugOption
from transformers.file_utils import (
    cached_property,
    get_full_repo_name,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_available,
    is_torch_tpu_available,
    torch_required,
)
from transformers.trainer_utils import EvaluationStrategy, HubStrategy, IntervalStrategy, SchedulerType
from transformers.training_args import default_logdir

from .utils import logging


logger = logging.get_logger(__name__)
log_levels = logging.get_log_levels_dict().copy()
trainer_log_levels = dict(**log_levels, passive=-1)


class ParallelMode(Enum):
    IPU = "ipu"


@dataclass
class IPUTrainingArguments:
    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )

    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(default=False, metadata={"help": "Whether to run predictions on the test set."})
    evaluation_strategy: IntervalStrategy = field(
        default="no",
        metadata={"help": "The evaluation strategy to use."},
    )
    prediction_loss_only: bool = field(
        default=False,
        metadata={"help": "When performing evaluation and predictions, only returns the loss."},
    )

    per_device_train_batch_size: int = field(default=2, metadata={"help": "Batch size per IPU for training."})
    per_device_eval_batch_size: int = field(default=8, metadata={"help": "Batch size per IPU for evaluation."})

    # per_gpu_train_batch_size: Optional[int] = field(
    #     default=None,
    #     metadata={
    #         "help": "Deprecated, the use of `--per_device_train_batch_size` is preferred. "
    #         "Batch size per GPU/TPU core/CPU for training."
    #     },
    # )
    # per_gpu_eval_batch_size: Optional[int] = field(
    #     default=None,
    #     metadata={
    #         "help": "Deprecated, the use of `--per_device_eval_batch_size` is preferred. "
    #         "Batch size per GPU/TPU core/CPU for evaluation."
    #     },
    # )

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    eval_accumulation_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Number of predictions steps to accumulate before moving the tensors to the CPU."},
    )

    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})

    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    )
    lr_scheduler_type: SchedulerType = field(
        default="linear",
        metadata={"help": "The scheduler type to use."},
    )
    warmup_ratio: float = field(
        default=0.0, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})

    log_level: Optional[str] = field(
        default="passive",
        metadata={
            "help": "Logger log level to use on the main node. Possible choices are the log levels as strings: 'debug', 'info', 'warning', 'error' and 'critical', plus a 'passive' level which doesn't set anything and lets the application set the level. Defaults to 'passive'.",
            "choices": trainer_log_levels.keys(),
        },
    )
    # log_level_replica: Optional[str] = field(
    #     default="passive",
    #     metadata={
    #         "help": "Logger log level to use on replica nodes. Same choices and defaults as ``log_level``",
    #         "choices": trainer_log_levels.keys(),
    #     },
    # )
    # log_on_each_node: bool = field(
    #     default=True,
    #     metadata={
    #         "help": "When doing a multinode distributed training, whether to log once per node or just once on the main node."
    #     },
    # )
    logging_dir: Optional[str] = field(default=None, metadata={"help": "Tensorboard log dir."})
    logging_strategy: IntervalStrategy = field(
        default="steps",
        metadata={"help": "The logging strategy to use."},
    )
    logging_first_step: bool = field(default=False, metadata={"help": "Log the first global_step"})
    logging_steps: int = field(default=500, metadata={"help": "Log every X updates steps."})
    logging_nan_inf_filter: str = field(default=True, metadata={"help": "Filter nan and inf losses for logging."})
    save_strategy: IntervalStrategy = field(
        default="steps",
        metadata={"help": "The checkpoint save strategy to use."},
    )
    save_steps: int = field(default=500, metadata={"help": "Save checkpoint every X updates steps."})
    save_total_limit: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Limit the total amount of checkpoints. "
                "Deletes the older checkpoints in the output_dir. Default is unlimited checkpoints"
            )
        },
    )
    # save_on_each_node: bool = field(
    #     default=False,
    #     metadata={
    #         "help": "When doing multi-node distributed training, whether to save models and checkpoints on each node, or only on the main one"
    #     },
    # )
    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})

    # fp16: bool = field(
    #     default=False,
    #     metadata={"help": "Whether to use 16-bit (mixed) precision instead of 32-bit"},
    # )
    # fp16_opt_level: str = field(
    #     default="O1",
    #     metadata={
    #         "help": (
    #             "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. "
    #             "See details at https://nvidia.github.io/apex/amp.html"
    #         )
    #     },
    # )
    # fp16_backend: str = field(
    #     default="auto",
    #     metadata={"help": "The backend to be used for mixed precision.", "choices": ["auto", "amp", "apex"]},
    # )
    # fp16_full_eval: bool = field(
    #     default=False,
    #     metadata={"help": "Whether to use full 16-bit precision evaluation instead of 32-bit"},
    # )
    # local_rank: int = field(default=-1, metadata={"help": "For distributed training: local_rank"})
    # xpu_backend: str = field(
    #     default=None,
    #     metadata={"help": "The backend to be used for distributed training on Intel XPU.", "choices": ["mpi", "ccl"]},
    # )
    # tpu_num_cores: Optional[int] = field(
    #     default=None, metadata={"help": "TPU: Number of TPU cores (automatically passed by launcher script)"}
    # )
    # tpu_metrics_debug: bool = field(
    #     default=False,
    #     metadata={
    #         "help": "Deprecated, the use of `--debug tpu_metrics_debug` is preferred. TPU: Whether to print debug metrics"
    #     },
    # )
    debug: str = field(
        default="",
        metadata={
            "help": "Whether or not to enable debug mode. Current options: "
            "`underflow_overflow` (Detect underflow and overflow in activations and weights), "
            "`tpu_metrics_debug` (print debug metrics on TPU)."
        },
    )

    dataloader_drop_last: bool = field(
        default=False, metadata={"help": "Drop the last incomplete batch if it is not divisible by the batch size."}
    )
    eval_steps: int = field(default=None, metadata={"help": "Run an evaluation every X steps."})
    dataloader_num_workers: int = field(
        default=0,
        metadata={
            "help": "Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded in the main process."
        },
    )

    past_index: int = field(
        default=-1,
        metadata={"help": "If >=0, uses the corresponding part of the output as the past state for next step."},
    )

    run_name: Optional[str] = field(
        default=None, metadata={"help": "An optional descriptor for the run. Notably used for wandb logging."}
    )
    disable_tqdm: Optional[bool] = field(
        default=None, metadata={"help": "Whether or not to disable the tqdm progress bars."}
    )

    remove_unused_columns: Optional[bool] = field(
        default=True, metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
    )
    label_names: Optional[List[str]] = field(
        default=None, metadata={"help": "The list of keys in your dictionary of inputs that correspond to the labels."}
    )

    load_best_model_at_end: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to load the best model found during training at the end of training."},
    )
    metric_for_best_model: Optional[str] = field(
        default=None, metadata={"help": "The metric to use to compare two different models."}
    )
    greater_is_better: Optional[bool] = field(
        default=None, metadata={"help": "Whether the `metric_for_best_model` should be maximized or not."}
    )
    ignore_data_skip: bool = field(
        default=False,
        metadata={
            "help": "When resuming training, whether or not to skip the first epochs and batches to get to the same training data."
        },
    )
    # sharded_ddp: str = field(
    #     default="",
    #     metadata={
    #         "help": "Whether or not to use sharded DDP training (in distributed training only). The base option "
    #         "should be `simple`, `zero_dp_2` or `zero_dp_3` and you can add CPU-offload to `zero_dp_2` or `zero_dp_3` "
    #         "like this: zero_dp_2 offload` or `zero_dp_3 offload`. You can add auto-wrap to `zero_dp_2` or "
    #         "with the same syntax: zero_dp_2 auto_wrap` or `zero_dp_3 auto_wrap`.",
    #     },
    # )
    # deepspeed: Optional[str] = field(
    #     default=None,
    #     metadata={
    #         "help": "Enable deepspeed and pass the path to deepspeed json config file (e.g. ds_config.json) or an already loaded json file as a dict"
    #     },
    # )
    label_smoothing_factor: float = field(
        default=0.0, metadata={"help": "The label smoothing epsilon to apply (zero means no label smoothing)."}
    )
    # adafactor: bool = field(default=False, metadata={"help": "Whether or not to replace AdamW by Adafactor."})
    group_by_length: bool = field(
        default=False,
        metadata={"help": "Whether or not to group samples of roughly the same length together when batching."},
    )
    length_column_name: Optional[str] = field(
        default="length",
        metadata={"help": "Column name with precomputed lengths to use when grouping by length."},
    )
    report_to: Optional[List[str]] = field(
        default=None, metadata={"help": "The list of integrations to report the results and logs to."}
    )
    # ddp_find_unused_parameters: Optional[bool] = field(
    #     default=None,
    #     metadata={
    #         "help": "When using distributed training, the value of the flag `find_unused_parameters` passed to "
    #         "`DistributedDataParallel`."
    #     },
    # )
    dataloader_pin_memory: bool = field(
        default=True, metadata={"help": "Whether or not to pin memory for DataLoader."}
    )
    skip_memory_metrics: bool = field(
        default=True, metadata={"help": "Whether or not to skip adding of memory profiler reports to metrics."}
    )
    # use_legacy_prediction_loop: bool = field(
    #     default=False, metadata={"help": "Whether or not to use the legacy prediction_loop in the Trainer."}
    # )
    push_to_hub: bool = field(
        default=False, metadata={"help": "Whether or not to upload the trained model to the model hub after training."}
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "The path to a folder with a valid checkpoint for your model."},
    )
    hub_model_id: str = field(
        default=None, metadata={"help": "The name of the repository to keep in sync with the local `output_dir`."}
    )
    hub_strategy: HubStrategy = field(
        default="every_save",
        metadata={"help": "The hub strategy to use when `--push_to_hub` is activated."},
    )
    hub_token: str = field(default=None, metadata={"help": "The token to use to push to the Model Hub."})
    gradient_checkpointing: bool = field(
        default=False,
        metadata={
            "help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."
        },
    )
    # Deprecated arguments
    push_to_hub_model_id: str = field(
        default=None, metadata={"help": "The name of the repository to which push the `Trainer`."}
    )
    push_to_hub_organization: str = field(
        default=None, metadata={"help": "The name of the organization in with to which push the `Trainer`."}
    )
    push_to_hub_token: str = field(default=None, metadata={"help": "The token to use to push to the Model Hub."})
    # _n_gpu: int = field(init=False, repr=False, default=-1)
    # mp_parameters: str = field(
    #     default="",
    #     metadata={"help": "Used by the SageMaker launcher to send mp-specific args. Ignored in Trainer"},
    # )
    # IPU Specific arguments
    ipu_config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained IPU config name or path if not the same as model_name"}
    )
    fp32: bool = field(
        default=False,
        metadata={"help": "Whether to use 32-bit (full) precision instead of 16-bit"},
    )
    lamb: bool = field(default=False, metadata={"help": "Whether or not to replace AdamW by LAMB."})
    lamb_no_bias_correction: bool = field(
        default=False, metadata={"help": "Whether or not to replace AdamW by LAMB without bias correction."}
    )
    loss_scaling: Optional[float] = field(
        default=None,
        metadata={
            "help": "Loss scaling factor (recommend using powers of 2)"
            "If using automatic loss scaling, this value will be the initial value."
        },
    )
    # TODO: add choices.
    # dataloader_mode: Literal["sync", "async", "async_rebatched"] = field(
    dataloader_mode: str = field(default="sync", metadata={"help": "The way data should be accessed."})
    compile_only: bool = field(default=False, metadata={"help": ""})

    def __post_init__(self):
        # Handle --use_env option in torch.distributed.launch (local_rank not passed as an arg then).
        # This needs to happen before any call to self.device or self.n_gpu.
        # env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
        # if env_local_rank != -1 and env_local_rank != self.local_rank:
        #     self.local_rank = env_local_rank

        # convert to int
        self.log_level = trainer_log_levels[self.log_level]
        # self.log_level_replica = trainer_log_levels[self.log_level_replica]

        # expand paths, if not os.makedirs("~/bar") will make directory
        # in the current directory instead of the actual home
        #  see https://github.com/huggingface/transformers/issues/10628
        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)
        if self.logging_dir is None and self.output_dir is not None:
            self.logging_dir = os.path.join(self.output_dir, default_logdir())
        if self.logging_dir is not None:
            self.logging_dir = os.path.expanduser(self.logging_dir)

        if self.disable_tqdm is None:
            self.disable_tqdm = logger.getEffectiveLevel() > logging.WARN

        if isinstance(self.evaluation_strategy, EvaluationStrategy):
            warnings.warn(
                "using `EvaluationStrategy` for `evaluation_strategy` is deprecated and will be removed in version 5 of 🤗 Transformers. Use `IntervalStrategy` instead",
                FutureWarning,
            )
            # Go back to the underlying string or we won't be able to instantiate `IntervalStrategy` on it.
            self.evaluation_strategy = self.evaluation_strategy.value

        self.evaluation_strategy = IntervalStrategy(self.evaluation_strategy)
        self.logging_strategy = IntervalStrategy(self.logging_strategy)
        self.save_strategy = IntervalStrategy(self.save_strategy)
        self.hub_strategy = HubStrategy(self.hub_strategy)

        self.lr_scheduler_type = SchedulerType(self.lr_scheduler_type)
        if self.do_eval is False and self.evaluation_strategy != IntervalStrategy.NO:
            self.do_eval = True

        # eval_steps has to be defined and non-zero, fallbacks to logging_steps if the latter is non-zero
        if self.evaluation_strategy == IntervalStrategy.STEPS and (self.eval_steps is None or self.eval_steps == 0):
            if self.logging_steps > 0:
                logger.info(f"using `logging_steps` to initialize `eval_steps` to {self.logging_steps}")
                self.eval_steps = self.logging_steps
            else:
                raise ValueError(
                    f"evaluation strategy {self.evaluation_strategy} requires either non-zero --eval_steps or --logging_steps"
                )

        # logging_steps must be non-zero for logging_strategy that is other than 'no'
        if self.logging_strategy == IntervalStrategy.STEPS and self.logging_steps == 0:
            raise ValueError(f"logging strategy {self.logging_strategy} requires non-zero --logging_steps")

        # Sanity checks for load_best_model_at_end: we require save and eval strategies to be compatible.
        if self.load_best_model_at_end:
            if self.evaluation_strategy != self.save_strategy:
                raise ValueError(
                    "--load_best_model_at_end requires the save and eval strategy to match, but found\n- Evaluation "
                    f"strategy: {self.evaluation_strategy}\n- Save strategy: {self.save_strategy}"
                )
            if self.evaluation_strategy == IntervalStrategy.STEPS and self.save_steps % self.eval_steps != 0:
                raise ValueError(
                    "--load_best_model_at_end requires the saving steps to be a round multiple of the evaluation "
                    f"steps, but found {self.save_steps}, which is not a round multiple of {self.eval_steps}."
                )

        if self.load_best_model_at_end and self.metric_for_best_model is None:
            self.metric_for_best_model = "loss"
        if self.greater_is_better is None and self.metric_for_best_model is not None:
            self.greater_is_better = self.metric_for_best_model not in ["loss", "eval_loss"]
        if self.run_name is None:
            self.run_name = self.output_dir

        # if is_torch_available() and self.device.type != "cuda" and (self.fp16 or self.fp16_full_eval):
        #     raise ValueError(
        #         "Mixed precision training with AMP or APEX (`--fp16`) and FP16 evaluation can only be used on CUDA devices."
        #     )
        if self.report_to is None:
            logger.info(
                "The default value for the training argument `--report_to` will change in v5 (from all installed "
                "integrations to none). In v5, you will need to use `--report_to all` to get the same behavior as "
                "now. You should start updating your code and make this info disappear :-)."
            )
            self.report_to = "all"
        if self.report_to == "all" or self.report_to == ["all"]:
            # Import at runtime to avoid a circular import.
            from transformers.integrations import get_available_reporting_integrations

            self.report_to = get_available_reporting_integrations()
        elif self.report_to == "none" or self.report_to == ["none"]:
            self.report_to = []
        elif not isinstance(self.report_to, list):
            self.report_to = [self.report_to]

        if self.warmup_ratio < 0 or self.warmup_ratio > 1:
            raise ValueError("warmup_ratio must lie in range [0,1]")
        elif self.warmup_ratio > 0 and self.warmup_steps > 0:
            logger.info(
                "Both warmup_ratio and warmup_steps given, warmup_steps will override any effect of warmup_ratio during training"
            )

        # if isinstance(self.sharded_ddp, bool):
        #     self.sharded_ddp = "simple" if self.sharded_ddp else ""
        # if isinstance(self.sharded_ddp, str):
        #     self.sharded_ddp = [ShardedDDPOption(s) for s in self.sharded_ddp.split()]
        # if self.sharded_ddp == [ShardedDDPOption.OFFLOAD]:
        #     raise ValueError(
        #         "`--sharded_ddp offload` can't work on its own. It needs to be added to `--sharded_ddp zero_dp_2` or "
        #         '`--sharded_ddp zero_dp_3`. For example, `--sharded_ddp "zero_dp_2 offload"`.'
        #     )
        # elif len(self.sharded_ddp) > 1 and ShardedDDPOption.SIMPLE in self.sharded_ddp:
        #     raise ValueError("`--sharded_ddp simple` is not compatible with any other option.")
        # elif ShardedDDPOption.ZERO_DP_2 in self.sharded_ddp and ShardedDDPOption.ZERO_DP_3 in self.sharded_ddp:
        #     raise ValueError("`--sharded_ddp zero_dp_2` is not compatible with `--sharded_ddp zero_dp_3`.")

        # if self.tpu_metrics_debug:
        #     warnings.warn(
        #         "using `--tpu_metrics_debug` is deprecated and will be removed in version 5 of 🤗 Transformers. Use `--debug tpu_metrics_debug` instead",
        #         FutureWarning,
        #     )
        #     self.debug += " tpu_metrics_debug"
        #     self.tpu_metrics_debug = False
        if isinstance(self.debug, str):
            self.debug = [DebugOption(s) for s in self.debug.split()]

        # if self.deepspeed:
        #     # - must be run very last in arg parsing, since it will use a lot of these settings.
        #     # - must be run before the model is created.
        #     from transformers.deepspeed import HfTrainerDeepSpeedConfig

        #     # will be used later by the Trainer
        #     # note: leave self.deepspeed unmodified in case a user relies on it not to be modified)
        #     self.hf_deepspeed_config = HfTrainerDeepSpeedConfig(self.deepspeed)
        #     self.hf_deepspeed_config.trainer_config_process(self)

        if self.push_to_hub_token is not None:
            warnings.warn(
                "`--push_to_hub_token` is deprecated and will be removed in version 5 of 🤗 Transformers. Use "
                "`--hub_token` instead.",
                FutureWarning,
            )
            self.hub_token = self.push_to_hub_token

        if self.push_to_hub_model_id is not None:
            self.hub_model_id = get_full_repo_name(
                self.push_to_hub_model_id, organization=self.push_to_hub_organization, token=self.hub_token
            )
            if self.push_to_hub_organization is not None:
                warnings.warn(
                    "`--push_to_hub_model_id` and `--push_to_hub_organization` are deprecated and will be removed in "
                    "version 5 of 🤗 Transformers. Use `--hub_model_id` instead and pass the full repo name to this "
                    f"argument (in this case {self.hub_model_id}).",
                    FutureWarning,
                )
            else:
                warnings.warn(
                    "`--push_to_hub_model_id` is deprecated and will be removed in version 5 of 🤗 Transformers. Use "
                    "`--hub_model_id` instead and pass the full repo name to this argument (in this case "
                    f"{self.hub_model_id}).",
                    FutureWarning,
                )
        elif self.push_to_hub_organization is not None:
            self.hub_model_id = f"{self.push_to_hub_organization}/{Path(self.output_dir).name}"
            warnings.warn(
                "`--push_to_hub_organization` is deprecated and will be removed in version 5 of 🤗 Transformers. Use "
                "`--hub_model_id` instead and pass the full repo name to this argument (in this case "
                f"{self.hub_model_id}).",
                FutureWarning,
            )

        # IPU specific
        dataloader_mode_mapping = {"sync": 0, "async": 1, "async_rebatched": 2}
        self.dataloader_mode = DataLoaderMode(dataloader_mode_mapping[self.dataloader_mode])

    # @property
    # def train_batch_size(self) -> int:
    #     """
    #     The actual batch size for training (may differ from :obj:`per_gpu_train_batch_size` in distributed training).
    #     """
    #     if self.per_gpu_train_batch_size:
    #         logger.warning(
    #             "Using deprecated `--per_gpu_train_batch_size` argument which will be removed in a future "
    #             "version. Using `--per_device_train_batch_size` is preferred."
    #         )
    #     per_device_batch_size = self.per_gpu_train_batch_size or self.per_device_train_batch_size
    #     train_batch_size = per_device_batch_size * max(1, self.n_gpu)
    #     return train_batch_size

    # @property
    # def eval_batch_size(self) -> int:
    #     """
    #     The actual batch size for evaluation (may differ from :obj:`per_gpu_eval_batch_size` in distributed training).
    #     """
    #     if self.per_gpu_eval_batch_size:
    #         logger.warning(
    #             "Using deprecated `--per_gpu_eval_batch_size` argument which will be removed in a future "
    #             "version. Using `--per_device_eval_batch_size` is preferred."
    #         )
    #     per_device_batch_size = self.per_gpu_eval_batch_size or self.per_device_eval_batch_size
    #     eval_batch_size = per_device_batch_size * max(1, self.n_gpu)
    #     return eval_batch_size

    @cached_property
    @torch_required
    def _setup_devices(self) -> "torch.device":
        logger.info("PyTorch: setting up devices")
        # if self.no_cuda:
        #     device = torch.device("cpu")
        #     self._n_gpu = 0
        #     if self.local_rank != -1:
        #         # Initializes distributed backend for cpu
        #         if self.xpu_backend not in ("mpi", "ccl"):
        #             raise ValueError(
        #                 "CPU distributed training backend is not properly set. "
        #                 "Please set '--xpu_backend' to either 'mpi' or 'ccl'."
        #             )
        #         torch.distributed.init_process_group(backend=self.xpu_backend)
        # elif is_torch_tpu_available():
        #     device = xm.xla_device()
        #     self._n_gpu = 0
        # elif is_sagemaker_mp_enabled():
        #     local_rank = smp.local_rank()
        #     device = torch.device("cuda", local_rank)
        #     self._n_gpu = 1
        # elif is_sagemaker_dp_enabled():
        #     sm_dist.init_process_group()
        #     self.local_rank = sm_dist.get_local_rank()
        #     device = torch.device("cuda", self.local_rank)
        #     self._n_gpu = 1
        # elif self.deepspeed:
        #     # deepspeed inits torch.distributed internally
        #     from .deepspeed import is_deepspeed_available

        #     if not is_deepspeed_available():
        #         raise ImportError("--deepspeed requires deepspeed: `pip install deepspeed`.")
        #     import deepspeed

        #     deepspeed.init_distributed()

        #     # workaround for setups like notebooks where the launcher can't be used,
        #     # but deepspeed requires a dist env.
        #     # env LOCAL_RANK could be set manually by the user, or via init_distributed if mpi4py is installed
        #     self.local_rank = int(os.environ.get("LOCAL_RANK", "-1"))

        #     device = torch.device("cuda", self.local_rank)
        #     self._n_gpu = 1
        # elif self.local_rank == -1:
        #     # if n_gpu is > 1 we'll use nn.DataParallel.
        #     # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
        #     # Explicitly set CUDA to the first (index 0) CUDA device, otherwise `set_device` will
        #     # trigger an error that a device index is missing. Index 0 takes into account the
        #     # GPUs available in the environment, so `CUDA_VISIBLE_DEVICES=1,2` with `cuda:0`
        #     # will use the first GPU in that env, i.e. GPU#1
        #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #     # Sometimes the line in the postinit has not been run before we end up here, so just checking we're not at
        #     # the default value.
        #     self._n_gpu = torch.cuda.device_count()
        # else:
        # Here, we'll use torch.distributed.
        # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
        # torch.distributed.init_process_group(backend="nccl")
        # device = torch.device("cuda", self.local_rank)
        # self._n_gpu = 1

        # if device.type == "cuda":
        #     torch.cuda.set_device(device)

        device = torch.device("cpu")

        return device

    @property
    @torch_required
    def device(self) -> "torch.device":
        """
        The device used by this process.
        """
        return self._setup_devices

    # @property
    # @torch_required
    # def n_gpu(self):
    #     """
    #     The number of GPUs used by this process.

    #     Note:
    #         This will only be greater than one when you have multiple GPUs available but are not using distributed
    #         training. For distributed training, it will always be 1.
    #     """
    #     # Make sure `self._n_gpu` is properly setup.
    #     _ = self._setup_devices
    #     return self._n_gpu

    @property
    @torch_required
    def parallel_mode(self):
        return ParallelMode.IPU
        # if is_torch_tpu_available():
        #     return ParallelMode.TPU
        # elif is_sagemaker_mp_enabled():
        #     return ParallelMode.SAGEMAKER_MODEL_PARALLEL
        # elif is_sagemaker_dp_enabled():
        #     return ParallelMode.SAGEMAKER_DATA_PARALLEL
        # elif self.local_rank != -1:
        #     return ParallelMode.DISTRIBUTED
        # elif self.n_gpu > 1:
        #     return ParallelMode.NOT_DISTRIBUTED
        # else:
        #     return ParallelMode.NOT_PARALLEL

    @property
    @torch_required
    def world_size(self):
        """
        The number of processes used in parallel.
        """
        # if is_torch_tpu_available():
        #     return xm.xrt_world_size()
        # elif is_sagemaker_mp_enabled():
        #     return smp.dp_size()
        # elif is_sagemaker_dp_enabled():
        #     return sm_dist.get_world_size()
        # elif self.local_rank != -1:
        #     return torch.distributed.get_world_size()
        return 1

    @property
    @torch_required
    def process_index(self):
        """
        The index of the current process used.
        """
        # if is_torch_tpu_available():
        #     return xm.get_ordinal()
        # elif is_sagemaker_mp_enabled():
        #     return smp.dp_rank()
        # elif is_sagemaker_dp_enabled():
        #     return sm_dist.get_rank()
        # elif self.local_rank != -1:
        #     return torch.distributed.get_rank()
        return 0

    @property
    @torch_required
    def local_process_index(self):
        """
        The index of the local process used.
        """
        # if is_torch_tpu_available():
        #     return xm.get_local_ordinal()
        # elif is_sagemaker_mp_enabled():
        #     return smp.local_rank()
        # elif is_sagemaker_dp_enabled():
        #     return sm_dist.get_rank()
        # elif self.local_rank != -1:
        #     return self.local_rank
        return 0

    @property
    def should_log(self):
        """
        Whether or not the current process should produce log.
        """
        return self.process_index
        # if self.log_on_each_node:
        #     return self.local_process_index == 0
        # else:
        #     # if is_sagemaker_mp_enabled():
        #     #     return smp.rank() == 0
        #     # else:
        #     return self.process_index == 0

    @property
    def should_save(self):
        """
        Whether or not the current process should write to disk, e.g., to save models and checkpoints.
        """
        return self.process_index == 0
        # if self.save_on_each_node:
        #     return self.local_process_index == 0
        # else:
        #     if is_sagemaker_mp_enabled():
        #         return smp.rank() == 0
        #     else:
        #         return self.process_index == 0

    def get_process_log_level(self):
        """
        Returns the log level to be used depending on whether this process is the main process of node 0, main process
        of node non-0, or a non-main process.

        For the main process the log level defaults to ``logging.INFO`` unless overridden by ``log_level`` argument.

        For the replica processes the log level defaults to ``logging.WARNING`` unless overridden by
        ``log_level_replica`` argument.

        The choice between the main and replica process settings is made according to the return value of
        ``should_log``.
        """
        log_level_main_node = logging.INFO if self.log_level == -1 else self.log_level
        return log_level_main_node
        # log_level_replica_node = logging.WARNING if self.log_level_replica == -1 else self.log_level_replica
        # return log_level_main_node if self.should_log else log_level_replica_node

    @property
    def place_model_on_device(self):
        """
        Can be subclassed and overridden for some specific integrations.
        """
        return not is_sagemaker_mp_enabled()

    @property
    def _no_sync_in_gradient_accumulation(self):
        """
        Whether or not to use no_sync for the gradients when doing gradient accumulation.
        """
        return not (self.deepspeed or is_sagemaker_dp_enabled() or is_sagemaker_mp_enabled())

    @contextlib.contextmanager
    def main_process_first(self, local=True, desc="work"):
        """
            A context manager for torch distributed environment where on needs to do something on the main process,
            while blocking replicas, and when it's finished releasing the replicas.

            One such use is for ``datasets``'s ``map`` feature which to be efficient should be run once on the main
            process, which upon completion saves a cached version of results and which then automatically gets loaded
            by the replicas.

        Args:
            local (:obj:`bool`, `optional`, defaults to :obj:`True`):
                if :obj:`True` first means process of rank 0 of each node if :obj:`False` first means process of rank 0
                of node rank 0 In multi-node environment with a shared filesystem you most likely will want to use
                ``local=False`` so that only the main process of the first node will do the processing. If however, the
                filesystem is not shared, then the main process of each node will need to do the processing, which is
                the default behavior.
            desc (:obj:`str`, `optional`, defaults to ``"work"``):
                a work description to be used in debug logs

        """
        if is_torch_available() and self.world_size > 1:
            if local:
                is_main_process = self.local_process_index == 0
                main_process_desc = "main local process"
            else:
                is_main_process = self.process_index == 0
                main_process_desc = "main process"

            try:
                if not is_main_process:
                    # tell all replicas to wait
                    logger.debug(f"{self.process_index}: waiting for the {main_process_desc} to perform {desc}")
                    if is_torch_tpu_available():
                        xm.rendezvous(desc)
                    elif is_sagemaker_dp_enabled():
                        sm_dist.barrier()
                    else:
                        torch.distributed.barrier()
                yield
            finally:
                if is_main_process:
                    # the wait is over
                    logger.debug(f"{self.process_index}: {main_process_desc} completed {desc}, releasing all replicas")
                    if is_torch_tpu_available():
                        xm.rendezvous(desc)
                    elif is_sagemaker_dp_enabled():
                        sm_dist.barrier()
                    else:
                        torch.distributed.barrier()
        else:
            yield

    def get_warmup_steps(self, num_training_steps: int):
        """
        Get number of steps used for a linear warmup.
        """
        warmup_steps = (
            self.warmup_steps if self.warmup_steps > 0 else math.ceil(num_training_steps * self.warmup_ratio)
        )
        return warmup_steps

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(self.to_dict(), indent=2)

    def to_sanitized_dict(self) -> Dict[str, Any]:
        """
        Sanitized serialization to use with TensorBoard’s hparams
        """
        d = self.to_dict()
        d = {**d, **{"train_batch_size": self.train_batch_size, "eval_batch_size": self.eval_batch_size}}

        valid_types = [bool, int, float, str]
        if is_torch_available():
            valid_types.append(torch.Tensor)

        return {k: v if type(v) in valid_types else str(v) for k, v in d.items()}

    @property
    def train_batch_size(self) -> int:
        """
        The actual batch size for training (may differ from :obj:`per_gpu_train_batch_size` in distributed training).
        """
        train_batch_size = self.per_device_train_batch_size
        return train_batch_size

    @property
    def eval_batch_size(self) -> int:
        """
        The actual batch size for evaluation (may differ from :obj:`per_gpu_eval_batch_size` in distributed training).
        """
        eval_batch_size = self.per_device_eval_batch_size
        return eval_batch_size
