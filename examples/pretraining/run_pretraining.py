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

import sys
import time
import wandb
import warnings
from tqdm import tqdm
from pathlib import Path
import logging

import torch
import transformers
from poptorch import trainingModel
from optimum.graphcore.data.pretraining_data import get_dataloader, get_generated_datum
from optimum.graphcore import PipelinedBertForPretraining
from optimum.graphcore.models.bert import get_options, parse_bert_args
from optimum.graphcore.utils.optimization import get_lr_scheduler, get_optimizer
from optimum.grapchore.utils.checkpointing import save_checkpoint, checkpoints_exist
from optimum.graphcore.utils.utils import get_sdk_version, cycle, logger, sync_metrics


if __name__ == "__main__":

    # Ignore known warnings
    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
    logging.getLogger("poptorch::python").setLevel(logging.ERROR)

    # Build config from args
    config = transformers.BertConfig(**(vars(parse_bert_args())))

    # Warnings for configs where embeddings may not fit
    if config.embedding_serialization_factor == 1:
        if config.replication_factor == 1:
            logger("[warning] With replication_factor == 1 you may need to set "
                   "embedding_serialization_factor > 1 for the model to fit")
        elif not config.replicated_tensor_sharding:
            logger("[warning] With replicated_tensor_sharding=False you may need to set "
                   "embedding_serialization_factor > 1 for the model to fit")
    # prevent overwriting of existing checkpoints
    if checkpoints_exist(config.checkpoint_output_dir):
        raise RuntimeError("Found previously saved checkpoint(s) at checkpoint-dir. "
                           "Overwriting checkpoints is not supported. "
                           "Please specify a different checkpoint-dir to "
                           "save checkpoints from this run.")

    # Execution parameters
    opts = get_options(config)

    # Dataloader
    logger("------------------- Data Loading Started ------------------")
    start_loading = time.perf_counter()
    loader = get_dataloader(config, opts)
    steps_per_epoch = len(loader)
    if steps_per_epoch < 1:
        raise RuntimeError("Not enough data in input_files for current configuration, "
                           "try reducing deviceIterations or gradientAccumulation.")
    duration_loader = time.perf_counter() - start_loading
    logger(f"Data loaded in {duration_loader} secs")
    logger("-----------------------------------------------------------")

    # Restore model from checkpoint
    steps_finished = 0
    if config.pretrained_checkpoint:
        # Load from checkpoint
        model = PipelinedBertForPretraining.from_pretrained(config.pretrained_checkpoint, config=config).parallelize().half().train()
        optimizer = get_optimizer(config, model)
        scheduler = get_lr_scheduler(optimizer, config.lr_schedule,
                                     config.lr_warmup, config.training_steps)

        if config.resume_training_from_checkpoint:
            training_state = torch.load(Path(config.pretrained_checkpoint) / "training_state.pt")
            scheduler.last_epoch = steps_finished = training_state["step"]
            checkpoint_metrics = training_state["metrics"]
    else:
        # Train model from scratch
        model = PipelinedBertForPretraining(config).parallelize().half().train()
        optimizer = get_optimizer(config, model)
        scheduler = get_lr_scheduler(optimizer, config.lr_schedule,
                                     config.lr_warmup, config.training_steps)

    # Checkpoint model at start of run
    save_checkpoint(config, model, steps_finished)

    poptorch_model = trainingModel(model, opts, optimizer=optimizer)

    # Compile model
    logger("---------- Compilation/Loading from Cache Started ---------")
    start_compile = time.perf_counter()
    datum = get_generated_datum(config)
    poptorch_model.compile(*datum)
    duration_compilation = time.perf_counter() - start_compile
    logger(f"Compiled/Loaded model in {duration_compilation} secs")
    logger("-----------------------------------------------------------")

    # Save model and end here if compile only mode is enabled
    if config.compile_only:
        logger("Model successfully compiled. Exiting now as '--compile-only' argument was passed.")
        sys.exit(0)

    # W&B
    if config.wandb and (not config.use_popdist or config.popdist_rank == 0):
        wandb.init(project="torch-bert")
        wandb_config = vars(config)
        wandb_config['sdk_version'] = get_sdk_version()
        wandb.config.update(wandb_config)

    # Training loop
    logger("--------------------- Training Started --------------------")
    start_train = time.perf_counter()
    loader = cycle(loader)
    train_iterator = tqdm(range(steps_finished, config.training_steps),
                          desc="Training", disable=config.disable_progress_bar or (config.use_popdist and not(config.popdist_rank == 0)))
    for step in train_iterator:
        start_step = time.perf_counter()
        outputs = poptorch_model(*next(loader))
        scheduler.step()
        poptorch_model.setOptimizer(optimizer)
        step_length = sync_metrics(time.perf_counter() - start_step)
        outputs_sync = sync_metrics(outputs)

        if not config.use_popdist or config.popdist_rank == 0:
            train_iterator.set_description(
                f"Step: {step} / {config.training_steps-1} - "
                f"LR: {scheduler.get_last_lr()[0]:.2e} - "
                f"Loss: {outputs_sync[0]:3.3f} - "
                f"Loss/MLM: {outputs_sync[1]:3.3f} - "
                f"Loss/NSP: {outputs_sync[2]:3.3f} - "
                f"Acc/MLM: {outputs_sync[3]:3.3f} - "
                f"Acc/NSP: {outputs_sync[4]:3.3f}")
            num_instances = config.popdist_size if config.use_popdist else 1
            step_throughput = config.samples_per_step * num_instances / step_length
            train_iterator.set_postfix_str(f"{step_throughput:.1f} sequences/s")

            if config.disable_progress_bar:
                logger(f"{train_iterator.desc} {train_iterator.postfix}")

            if config.wandb:
                wandb.log({"Loss": outputs_sync[0],
                           "Loss/MLM": outputs_sync[1],
                           "Loss/NSP": outputs_sync[2],
                           "Acc/MLM": outputs_sync[3],
                           "Acc/NSP": outputs_sync[4],
                           "LR": scheduler.get_last_lr()[0],
                           "Step": step,
                           "Throughput": step_throughput})

                if config.wandb_param_steps and (step % config.wandb_param_steps) == 0:
                    for name, parameter in poptorch_model.named_parameters():
                        wandb.run.history.torch.log_tensor_stats(parameter.data, name)

            if config.checkpoint_steps and (step % config.checkpoint_steps) == 0:
                save_checkpoint(config, model, step,
                                metrics={"Loss": outputs_sync[0],
                                         "Acc/MLM": outputs_sync[3],
                                         "Acc/NSP": outputs_sync[4]})

        if step + 1 == config.training_steps:
            break  # Training finished mid-epoch

    stop_train = time.perf_counter()
    # Checkpoint at end of run
    if not config.use_popdist or config.popdist_rank == 0:
        save_checkpoint(config, model, step,
                        metrics={"Loss": outputs[0].mean().item(),
                                 "Acc/MLM": outputs[3].mean().item(),
                                 "Acc/NSP": outputs[4].mean().item()})
    logger("-----------------------------------------------------------")

    logger("-------------------- Training Metrics ---------------------")
    logger(f"global_batch_size: {config.global_batch_size}")
    logger(f"batches_per_step: {config.batches_per_step}")
    logger(f"training_steps: {config.training_steps}")
    duration_run = stop_train - start_train
    num_samples = config.samples_per_step * config.training_steps
    logger(f"Training time: {duration_run:.3f} secs")
    logger("-----------------------------------------------------------")
