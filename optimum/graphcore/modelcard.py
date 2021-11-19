# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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


import os
from dataclasses import dataclass
from pathlib import Path

from transformers.modelcard import (
    _TRAINING_ARGS_KEYS,
    TASK_MAPPING,
    TrainingSummary,
    _get_mapping_values,
    is_hf_dataset,
    parse_log_history,
)


@dataclass
class IPUTrainingSummary(TrainingSummary):
    @classmethod
    def from_trainer(
        cls,
        trainer,
        language=None,
        license=None,
        tags=None,
        model_name=None,
        finetuned_from=None,
        tasks=None,
        dataset_tags=None,
        dataset=None,
        dataset_args=None,
    ):
        # Infer default from dataset
        one_dataset = trainer.train_dataset if trainer.train_dataset is not None else trainer.eval_dataset
        if is_hf_dataset(one_dataset) and (dataset_tags is None or dataset_args is None):
            default_tag = one_dataset.builder_name
            # Those are not real datasets from the Hub so we exclude them.
            if default_tag not in ["csv", "json", "pandas", "parquet", "text"]:
                if dataset_tags is None:
                    dataset_tags = [default_tag]
                if dataset_args is None:
                    dataset_args = [one_dataset.config_name]

        if dataset is None and dataset_tags is not None:
            dataset = dataset_tags

        # Infer default finetuned_from
        if (
            finetuned_from is None
            and hasattr(trainer.model.config, "_name_or_path")
            and not os.path.isdir(trainer.model.config._name_or_path)
        ):
            finetuned_from = trainer.model.config._name_or_path

        # Infer default task tag:
        if tasks is None:
            model_class_name = trainer.model.__class__.__name__
            for task, mapping in TASK_MAPPING.items():
                if model_class_name in _get_mapping_values(mapping):
                    tasks = task

        if model_name is None:
            model_name = Path(trainer.args.output_dir).name

        # Add `generated_from_trainer` to the tags
        if tags is None:
            tags = ["generated_from_trainer"]
        elif isinstance(tags, str) and tags != "generated_from_trainer":
            tags = [tags, "generated_from_trainer"]
        elif "generated_from_trainer" not in tags:
            tags.append("generated_from_trainer")

        _, eval_lines, eval_results = parse_log_history(trainer.state.log_history)
        hyperparameters = extract_hyperparameters_from_trainer(trainer)

        return cls(
            language=language,
            license=license,
            tags=tags,
            model_name=model_name,
            finetuned_from=finetuned_from,
            tasks=tasks,
            dataset_tags=dataset_tags,
            dataset=dataset,
            dataset_args=dataset_args,
            eval_results=eval_results,
            eval_lines=eval_lines,
            hyperparameters=hyperparameters,
        )


def extract_hyperparameters_from_trainer(trainer):
    hyperparameters = {k: getattr(trainer.args, k) for k in _TRAINING_ARGS_KEYS}

    hyperparameters["distributed_type"] = "IPU"

    if trainer.args.gradient_accumulation_steps > 1:
        hyperparameters["gradient_accumulation_steps"] = trainer.args.gradient_accumulation_steps

    total_train_batch_size = trainer.args.train_batch_size * trainer.ipu_config.batch_size_factor()
    if total_train_batch_size != hyperparameters["train_batch_size"]:
        hyperparameters["total_train_batch_size"] = total_train_batch_size
    total_eval_batch_size = trainer.args.eval_batch_size * trainer.ipu_config.batch_size_factor(for_inference=True)
    if total_eval_batch_size != hyperparameters["eval_batch_size"]:
        hyperparameters["total_eval_batch_size"] = total_eval_batch_size

    if trainer.args.lamb:
        hyperparameters["optimizer"] = "LAMB"
    else:
        hyperparameters[
            "optimizer"
        ] = f"Adam with betas=({trainer.args.adam_beta1},{trainer.args.adam_beta2}) and epsilon={trainer.args.adam_epsilon}"

    hyperparameters["lr_scheduler_type"] = trainer.args.lr_scheduler_type.value
    if trainer.args.warmup_ratio != 0.0:
        hyperparameters["lr_scheduler_warmup_ratio"] = trainer.args.warmup_ratio
    if trainer.args.warmup_steps != 0.0:
        hyperparameters["lr_scheduler_warmup_steps"] = trainer.args.warmup_steps
    if trainer.args.max_steps != -1:
        hyperparameters["training_steps"] = trainer.args.max_steps
    else:
        hyperparameters["num_epochs"] = trainer.args.num_train_epochs

    if trainer.args.fp32:
        hyperparameters["training precision"] = "Full Precision"
    else:
        hyperparameters["training precision"] = "Mixed Precision"
    # if trainer.args.fp16:
    #     if trainer.use_amp:
    #         hyperparameters["mixed_precision_training"] = "Native AMP"
    #     elif trainer.use_apex:
    #         hyperparameters["mixed_precision_training"] = f"Apex, opt level {trainer.args.fp16_opt_level}"

    if trainer.args.label_smoothing_factor != 0.0:
        hyperparameters["label_smoothing_factor"] = trainer.args.label_smoothing_factor

    return hyperparameters
