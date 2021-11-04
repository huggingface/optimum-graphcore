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

from dataclasses import dataclass, field
from typing import Optional

from poptorch import DataLoaderMode
from transformers.training_args import TrainingArguments
from transformers.utils import logging

logger = logging.get_logger(__name__)


@dataclass
class IPUTrainingArguments(TrainingArguments):
    ipu_config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained IPU config name or path if not the same as model_name"}
    )
    fp32: bool = field(
        default=False,
        metadata={"help": "Whether to use 32-bit (full) precision instead of 16-bit"},
    )
    enable_half_first_order_momentum: bool = field(
        default=False,
        metadata={"help": "Sets first order momentum type to float16 instead of float32"},
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
        super().__post_init__()

        dataloader_mode_mapping = {"sync": 0, "async": 1, "async_rebatched": 2}
        self.dataloader_mode = DataLoaderMode(dataloader_mode_mapping[self.dataloader_mode])

    @property
    def train_batch_size(self) -> int:
        """
        The actual batch size for training (may differ from :obj:`per_gpu_train_batch_size` in distributed training).
        """
        if self.per_gpu_train_batch_size:
            logger.warning(
                "Using deprecated `--per_gpu_train_batch_size` argument which will be removed in a future "
                "version. Using `--per_device_train_batch_size` is preferred."
            )
        per_device_batch_size = self.per_gpu_train_batch_size or self.per_device_train_batch_size
        train_batch_size = (
            per_device_batch_size * self.replication_factor * self.gradient_accumulation_steps * self.device_iterations
        )
        return train_batch_size

    @property
    def eval_batch_size(self) -> int:
        """
        The actual batch size for evaluation (may differ from :obj:`per_gpu_eval_batch_size` in distributed training).
        """
        if self.per_gpu_eval_batch_size:
            logger.warning(
                "Using deprecated `--per_gpu_eval_batch_size` argument which will be removed in a future "
                "version. Using `--per_device_eval_batch_size` is preferred."
            )
        per_device_batch_size = self.per_gpu_eval_batch_size or self.per_device_eval_batch_size
        eval_batch_size = per_device_batch_size * self.replication_factor * 1 * self.device_iterations
        return eval_batch_size
