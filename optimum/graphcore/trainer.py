#  copyright 2021 the huggingface team. all rights reserved.
#
#  licensed under the apache license, version 2.0 (the "license");
#  you may not use this file except in compliance with the license.
#  you may obtain a copy of the license at
#
#      http://www.apache.org/licenses/license-2.0
#
#  unless required by applicable law or agreed to in writing, software
#  distributed under the license is distributed on an "as is" basis,
#  without warranties or conditions of any kind, either express or implied.
#  see the license for the specific language governing permissions and
#  limitations under the license.

import collections
import copy
import math
import os
import sys
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import optuna
import poptorch
import torch
import torch.nn as nn
from optimum.version import __version__
from poptorch import PoplarExecutor
from poptorch.optim import LAMB, AdamW
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import (
    DataCollator,
    EvalPrediction,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    is_datasets_available,
)
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.file_utils import CONFIG_NAME, WEIGHTS_NAME
from transformers.modeling_utils import unwrap_model
from transformers.optimization import get_scheduler
from transformers.trainer import TRAINER_STATE_NAME, TRAINING_ARGS_NAME
from transformers.trainer_callback import TrainerState
from transformers.trainer_pt_utils import (
    IterableDatasetShard,
    find_batch_size,
    get_parameter_names,
    nested_concat,
    nested_numpify,
    nested_truncate,
)
from transformers.trainer_utils import (
    EvalLoopOutput,
    TrainOutput,
    denumpify_detensorize,
    get_last_checkpoint,
    set_seed,
    speed_metrics,
)
from transformers.utils import logging

from .ipu_configuration import IPUConfig
from .modeling_utils import to_pipelined
from .trainer_utils import _WorkerInit, to_poptorch_dataloader

if is_datasets_available():
    import datasets


logger = logging.get_logger(__name__)


class IPUTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        ipu_config: IPUConfig = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
    ):
        if optimizers != (None, None):
            raise NotImplementedError("providing optimizers to IPUTrainer is not supported yet.")

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        self.ipu_config = copy.deepcopy(ipu_config)
        self.ipu_config.random_seed = self.args.seed

        if self.args.output_dir:
            path = os.path.join(self.args.output_dir, "executable_cache")
            logger.info(f"Executable caching enabled, cache directory: {path}")
            if self.ipu_config.executable_cache_dir is not None:
                logger.warning(f"IPUConfig executable_cache_dir was overriden to be: {path}")
            self.ipu_config.executable_cache_dir = path

        # for k, v in self.args.__dict__.items():
        #     if v is None:
        #         continue
        #     if hasattr(self.ipu_config, k) and getattr(self.ipu_config, k) != v:
        #         logger.warning(f"IPUConfig {k} attribute was overriden from TrainingArguments to {v}")
        #         setattr(self.ipu_config, k, v)

        # TODO: find a better way to track combinbed batch size instead of setting attributes to self.args
        self.args.__dict__.update(ipu_config.__dict__)
        self.opts = ipu_config.to_options()
        self.eval_opts = ipu_config.to_options(for_inference=True)

        self.original_model = to_pipelined(self.model, ipu_config)
        self.model = copy.deepcopy(self.original_model).parallelize()
        if not self.args.fp32:
            self.model = self.model.half()
        self.model_wrapped = self.model

        self.training_model = None
        self.inference_model = None

    def _compile_model(
        self,
        model: poptorch.PoplarExecutor,
        sample_batch: Union[Dict[str, torch.Tensor], Tuple[torch.Tensor]],
        log: bool = False,
    ):
        # Skipping compilation if the model was already compiled.
        if model.isCompiled():
            return
        if log:
            logger.info("Compiling Model...")
        sample_batch = self._prepare_inputs(sample_batch)
        start_compile = time.perf_counter()
        if isinstance(sample_batch, tuple):
            model.compile(*sample_batch)
        else:
            model.compile(**sample_batch)
        duration_compilation = time.perf_counter() - start_compile
        if log:
            logger.info(f"Compiled/Loaded model in {duration_compilation} secs")

    def get_train_dataloader(self) -> poptorch.DataLoader:
        """
        Returns the training :class:`~poptorch.DataLoader`.

        Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler (adapted
        to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")

        poptorch_specific_kwargs = {
            "drop_last": True,  # Not dropping last will end up causing NaN during training if the combined batch size does not divide the number of steps
            "auto_distributed_partitioning": not isinstance(train_dataset, torch.utils.data.IterableDataset),
            "mode": self.args.dataloader_mode,
            "worker_init_fn": _WorkerInit(123),
        }

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            # TODO: add support, should be easy.
            raise NotImplementedError("Training with IterableDataset not supported yet.")

        #     if self.args.world_size > 1:
        #         train_dataset = IterableDatasetShard(
        #             train_dataset,
        #             batch_size=self.args.train_batch_size,
        #             drop_last=self.args.dataloader_drop_last,
        #             num_processes=self.args.world_size,
        #             process_index=self.args.process_index,
        #         )

        #     return poptorch.DataLoader(
        #         self.opts,
        #         train_dataset,
        #         batch_size=self.args.train_batch_size,
        #         collate_fn=self.data_collator,
        #         num_workers=self.args.dataloader_num_workers,
        #         pin_memory=self.args.dataloader_pin_memory,
        #         **poptorch_specific_kwargs,
        #     )

        train_sampler = self._get_train_sampler()

        return poptorch.DataLoader(
            self.opts,
            train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            # drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            **poptorch_specific_kwargs,
        )

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> poptorch.DataLoader:
        """
        Returns the evaluation :class:`~poptorch.DataLoader`.

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (:obj:`torch.utils.data.Dataset`, `optional`):
                If provided, will override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`, columns not
                accepted by the ``model.forward()`` method are automatically removed. It must implement :obj:`__len__`.
        """
        poptorch_specific_kwargs = {
            "auto_distributed_partitioning": not isinstance(eval_dataset, torch.utils.data.IterableDataset),
            "mode": self.args.dataloader_mode,
            "worker_init_fn": _WorkerInit(123),
        }

        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")

        if isinstance(eval_dataset, torch.utils.data.IterableDataset):
            # TODO: add support, should be easy.
            raise NotImplementedError("Evluation with IterableDataset not supported yet.")
            # if self.args.world_size > 1:
            #     eval_dataset = IterableDatasetShard(
            #         eval_dataset,
            #         batch_size=self.args.eval_batch_size,
            #         drop_last=self.args.dataloader_drop_last,
            #         num_processes=self.args.world_size,
            #         process_index=self.args.process_index,
            #     )
            # return DataLoader(
            #     eval_dataset,
            #     batch_size=self.args.eval_batch_size,
            #     collate_fn=self.data_collator,
            #     num_workers=self.args.dataloader_num_workers,
            #     pin_memory=self.args.dataloader_pin_memory,
            # )

        eval_sampler = self._get_eval_sampler(eval_dataset)

        return poptorch.DataLoader(
            self.eval_opts,
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            **poptorch_specific_kwargs,
        )

    def get_test_dataloader(self, test_dataset: Dataset) -> poptorch.DataLoader:
        """
        Returns the test :class:`~poptorch.DataLoader`.

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            test_dataset (:obj:`torch.utils.data.Dataset`, `optional`):
                The test dataset to use. If it is an :obj:`datasets.Dataset`, columns not accepted by the
                ``model.forward()`` method are automatically removed. It must implement :obj:`__len__`.
        """
        poptorch_specific_kwargs = {
            "auto_distributed_partitioning": not isinstance(test_dataset, torch.utils.data.IterableDataset),
            "mode": self.args.dataloader_mode,
            "worker_init_fn": _WorkerInit(123),
        }

        if is_datasets_available() and isinstance(test_dataset, datasets.Dataset):
            test_dataset = self._remove_unused_columns(test_dataset, description="test")

        if isinstance(test_dataset, torch.utils.data.IterableDataset):
            # TODO: add support, should be easy.
            raise NotImplementedError("Testing with IterableDataset not supported yet.")
            # if self.args.world_size > 1:
            #     test_dataset = IterableDatasetShard(
            #         test_dataset,
            #         batch_size=self.args.eval_batch_size,
            #         drop_last=self.args.dataloader_drop_last,
            #         num_processes=self.args.world_size,
            #         process_index=self.args.process_index,
            #     )
            # return DataLoader(
            #     test_dataset,
            #     batch_size=self.args.eval_batch_size,
            #     collate_fn=self.data_collator,
            #     num_workers=self.args.dataloader_num_workers,
            #     pin_memory=self.args.dataloader_pin_memory,
            # )

        test_sampler = self._get_eval_sampler(test_dataset)

        # We use the same batch_size as for eval.
        return poptorch.DataLoader(
            self.eval_opts,
            test_dataset,
            sampler=test_sampler,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            pin_memory=self.args.dataloader_pin_memory,
            **poptorch_specific_kwargs,
        )

    # @to_poptorch_dataloader(for_training=True)
    # def get_train_dataloader(self) -> poptorch.DataLoader:
    #     """
    #     Returns the training :class:`~poptorch.DataLoader`.

    #     Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler (adapted
    #     to distributed training if necessary) otherwise.

    #     Subclass and override this method if you want to inject some custom behavior.
    #     """
    #     return super().get_train_dataloader()

    # @to_poptorch_dataloader(for_training=False)
    # def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> poptorch.DataLoader:
    #     """
    #     Returns the evaluation :class:`~poptorch.DataLoader`.

    #     Subclass and override this method if you want to inject some custom behavior.

    #     Args:
    #         eval_dataset (:obj:`torch.utils.data.Dataset`, `optional`):
    #             If provided, will override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`, columns not
    #             accepted by the ``model.forward()`` method are automatically removed. It must implement :obj:`__len__`.
    #     """
    #     return super().get_eval_dataloader()

    # @to_poptorch_dataloader(for_training=False)
    # def get_test_dataloader(self, test_dataset: Dataset) -> poptorch.DataLoader:
    #     """
    #     Returns the test :class:`~poptorch.DataLoader`.

    #     Subclass and override this method if you want to inject some custom behavior.

    #     Args:
    #         test_dataset (:obj:`torch.utils.data.Dataset`, `optional`):
    #             The test dataset to use. If it is an :obj:`datasets.Dataset`, columns not accepted by the
    #             ``model.forward()`` method are automatically removed. It must implement :obj:`__len__`.
    #     """
    #     return super().get_test_dataloader()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            # TODO: make sure the same thing is done as in GraphCore example.
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                    "weight_decay": 0.0,
                },
            ]
            if self.args.lamb or self.args.lamb_no_bias_correction:
                optimizer_cls = LAMB
                optimizer_kwargs = {
                    "max_weight_norm": None,
                    "bias_correction": self.lamb_no_bias_correction,
                }
            else:
                optimizer_cls = AdamW
                optimizer_kwargs = {
                    "betas": (self.args.adam_beta1, self.args.adam_beta2),
                    "eps": self.args.adam_epsilon,
                    "bias_correction": False,
                }

            # TODO: update the training args
            first_order_type = torch.float16 if self.args.enable_half_first_order_momentum else torch.float32
            optimizer_kwargs["lr"] = self.args.learning_rate
            optimizer_kwargs["weight_decay"] = 0
            optimizer_kwargs["loss_scaling"] = self.args.loss_scaling
            optimizer_kwargs["accum_type"] = torch.float16
            optimizer_kwargs["first_order_momentum_accum_type"] = first_order_type
            optimizer_kwargs["second_order_momentum_accum_type"] = torch.float32

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if self.args.lamb or self.args.lamb_no_bias_correction:
                self.optimizer.variable_attrs.markAsConstant("max_weight_norm")

            self.optimizer.variable_attrs.markAsConstant("weight_decay")

            # TODO: enable this feature.
            # if self.args.use_popdist:
            #     # TODO make sure the proper model is provided.
            #     hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)

        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """

        num_warmup_steps = self.args.get_warmup_steps(num_training_steps)
        optimizer = self.optimizer if optimizer is None else optimizer
        if self.args.lr_scheduler_type == "linear":
            num_warmup_steps *= num_training_steps
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )
            optimizer._step_count = 1

        return self.lr_scheduler

    def _wrap_model(self, model: Union[PreTrainedModel, PoplarExecutor], training=True):
        wrapped = None
        if isinstance(model, poptorch.PoplarExecutor):
            wrapped = model
        elif training:
            if self.training_model is None:
                self.training_model = poptorch.trainingModel(
                    model.train(), options=self.opts, optimizer=self.optimizer
                )
            wrapped = self.training_model
        else:
            if self.inference_model is None:
                self.inference_model = poptorch.inferenceModel(model.eval(), options=self.eval_opts)
            wrapped = self.inference_model

        # Attaching to device when the model that is being access was already compiled but detached from previous loop.
        if wrapped.isCompiled() and not wrapped.isAttachedToDevice():
            wrapped.attachToDevice()

        return wrapped

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Main training entry point.

        Args:
            resume_from_checkpoint (:obj:`str` or :obj:`bool`, `optional`):
                If a :obj:`str`, local path to a saved checkpoint as saved by a previous instance of
                :class:`~transformers.Trainer`. If a :obj:`bool` and equals `True`, load the last checkpoint in
                `args.output_dir` as saved by a previous instance of :class:`~transformers.Trainer`. If present,
                training will resume from the model/optimizer/scheduler states loaded here.
            trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            ignore_keys_for_eval (:obj:`List[str]`, `optional`)
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions for evaluation during the training.
            kwargs:
                Additional keyword arguments used to hide deprecated arguments
        """
        resume_from_checkpoint = None if not resume_from_checkpoint else resume_from_checkpoint

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        args = self.args

        self.is_in_train = True

        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )
        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")

        # This might change the seed so needs to run first.
        # self._hp_search_setup(trial)
        # Seed must be set before instantiating the model when using model_init.
        # set_seed(args.seed)
        # self.model = self.call_model_init(trial)
        # model_reloaded = True
        # # Reinitializes optimizer and scheduler
        # self.optimizer, self.lr_scheduler = None, None

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")

        if resume_from_checkpoint is not None:
            if not os.path.isfile(os.path.join(resume_from_checkpoint, WEIGHTS_NAME)):
                raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

            logger.info(f"Loading model from {resume_from_checkpoint}).")

            if os.path.isfile(os.path.join(resume_from_checkpoint, CONFIG_NAME)):
                # TODO: how do we reload IPU specific configs.
                config = PretrainedConfig.from_json_file(os.path.join(resume_from_checkpoint, CONFIG_NAME))
                checkpoint_version = config.transformers_version
                if checkpoint_version is not None and checkpoint_version != __version__:
                    logger.warn(
                        f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
                        f"Transformers but your current version is {__version__}. This is not recommended and could "
                        "yield to errors or unwanted behaviors."
                    )

            if args.deepspeed:
                # will be resumed in deepspeed_init
                pass
            else:
                # We load the model state dict on the CPU to avoid an OOM error.
                state_dict = torch.load(os.path.join(resume_from_checkpoint, WEIGHTS_NAME), map_location="cpu")
                # If the model is on the GPU, it still works!
                self._load_state_dict_in_model(state_dict)

                # release memory
                del state_dict

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        # if model_reloaded:
        #     if self.place_model_on_device:
        #         self._move_model_to_device(self.model, args.device)
        #     self.model_wrapped = self.model

        # Keeping track whether we can can len() on the dataset or not
        train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size
        if train_dataset_is_sized:
            # No need to divide by the number of gradient accumulation steps as poptorch already accounts for that.
            # num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
            num_update_steps_per_epoch = len(train_dataloader)
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )

                # May be slightly incorrect if the last batch in the training datalaoder has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = len(self.train_dataset) * args.num_train_epochs
        else:
            # see __init__. max_steps is set when the dataset has no __len__
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_train_samples = args.max_steps * total_train_batch_size

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # TODO: handle optimizer and scheduler creation
        # if delay_optimizer_creation:
        #     self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        self._compile_model(model, next(iter(train_dataloader)), log=True)

        # Train!
        num_examples = (
            self.num_examples(train_dataloader) if train_dataset_is_sized else total_train_batch_size * args.max_steps
        )

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Device Iterations = {self.ipu_config.device_iterations}")
        logger.info(f"  Replication Factor = {self.ipu_config.replication_factor}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                # No need to multiply by the number of gradient accumulation steps as poptorch already accounts for that.
                # steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
        # if trial is not None:
        #     assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
        #     self.state.trial_params = hp_params(assignments)
        # else:
        self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                # We just need to begin an iteration to create the randomization of the sampler.
                for _ in train_dataloader:
                    break

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, poptorch.DataLoader) and isinstance(
                train_dataloader.sampler, DistributedSampler
            ):
                train_dataloader.sampler.set_epoch(epoch)
            elif isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator) if train_dataset_is_sized else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                # TODO: gradient accumulation happens inside PopTorch, how to handle this then?
                # if step % args.gradient_accumulation_steps == 0:
                self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                # if (
                #     ((step + 1) % args.gradient_accumulation_steps != 0)
                #     and args.local_rank != -1
                #     and args._no_sync_in_gradient_accumulation
                # ):
                #     # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                #     with model.no_sync():
                #         tr_loss_step = self.training_step(model, inputs)
                # else:
                tr_loss_step = self.training_step(model, inputs)

                # if (
                #     args.logging_nan_inf_filter
                #     and not is_torch_tpu_available()
                #     and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                # ):
                #     # if loss is nan or inf simply add the average of previous logged losses
                #     tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                # else:
                tr_loss += tr_loss_step

                # TODO: see how to enable this (if necessary), slows down training a lot.
                # self.current_flos += float(self.floating_point_ops(inputs))

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                # if self.deepspeed:
                #     self.deepspeed.step()

                # if (step + 1) % args.gradient_accumulation_steps == 0 or (
                #     # last step in epoch but step is always smaller than gradient_accumulation_steps
                #     steps_in_epoch <= args.gradient_accumulation_steps
                #     and (step + 1) == steps_in_epoch
                # ):
                # Gradient clipping
                # if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                #    # deepspeed does its own clipping
                #    if self.use_amp:
                #        # AMP: gradients need unscaling
                #        self.scaler.unscale_(self.optimizer)

                # if hasattr(self.optimizer, "clip_grad_norm"):
                #     # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                #     self.optimizer.clip_grad_norm(args.max_grad_norm)
                # elif hasattr(model, "clip_grad_norm_"):
                #     # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                #     model.clip_grad_norm_(args.max_grad_norm)
                # else:
                #     # Revert to normal clipping otherwise, handling Apex or full precision
                #     nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                # Optimizer step
                optimizer_was_run = True
                # elif self.use_amp:
                #     scale_before = self.scaler.get_scale()
                #     self.scaler.step(self.optimizer)
                #     self.scaler.update()
                #     scale_after = self.scaler.get_scale()
                #     optimizer_was_run = scale_before <= scale_after
                # else:
                # self.optimizer.step()

                if optimizer_was_run and not self.deepspeed:
                    self.lr_scheduler.step()

                # model.zero_grad()
                self.state.global_step += 1
                self.state.epoch = epoch + (step + 1) / steps_in_epoch
                self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                # else:
                #     self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            # if is_torch_tpu_available():
            #     xm.rendezvous("load_best_model_at_end")
            # elif args.local_rank != -1:
            #     dist.barrier()

            logger.info(
                f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
            )

            best_model_path = os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME)
            if os.path.exists(best_model_path):
                # We load the model state dict on the CPU to avoid an OOM error.
                state_dict = torch.load(best_model_path, map_location="cpu")
                # If the model is on the GPU, it still works!
                self._load_state_dict_in_model(state_dict)
            else:
                logger.warn(
                    f"Could not locate the best model at {best_model_path}, if you are running a distributed training "
                    "on multiple nodes, you should activate `--save_on_each_node`."
                )

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Detaching model from device to let the inference model attach itself
        model.detachFromDevice()

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def _load_state_dict_in_model(self, state_dict):
        load_result = self.original_model.load_state_dict(state_dict, strict=False)
        model = copy.deepcopy(self.original_model).parallelize()
        self.model.load_state_dict(model.state_dict())

        if self.training_model and self.training_model.isAttachedToDevice():
            self.training_model.copyWeghtsToDevice()

        if self.inference_model and self.inference_model.isAttachedToDevice():
            self.inference_model.copyWeghtsToDevice()

        if len(load_result.missing_keys) != 0:
            if self.model._keys_to_ignore_on_save is not None and set(load_result.missing_keys) == set(
                self.model._keys_to_ignore_on_save
            ):
                self.model.tie_weights()
            else:
                logger.warn(f"There were missing keys in the checkpoint model loaded: {load_result.missing_keys}.")
        if len(load_result.unexpected_keys) != 0:
            logger.warn(f"There were unexpected keys in the checkpoint model loaded: {load_result.unexpected_keys}.")

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            model.detachFromDevice()
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            model.attachToDevice()
            self._report_to_hp_search(trial, epoch, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def training_step(
        self, model: poptorch.PoplarExecutor, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
        # model.train()
        inputs = self._prepare_inputs(inputs)
        loss = self.compute_loss(model, inputs)
        loss = loss.mean()
        return loss

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.

        Works both with or without labels.
        """
        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        model = self._wrap_model(self.model, training=False)
        self._compile_model(model, next(iter(dataloader)), log=True)

        # if full fp16 is wanted on eval and this ``evaluation`` or ``predict`` isn't called while
        # ``train`` is running, halve it first and then put on device
        if not self.is_in_train and self.args.fp16_full_eval:
            model = model.half().to(self.args.device)

        batch_size = dataloader.batch_size

        logger.info(f"***** Running {description} *****")
        if isinstance(dataloader.dataset, collections.abc.Sized):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        # if is_torch_tpu_available():
        #     dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)

        if self.args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if self.args.eval_accumulation_steps is not None and (step + 1) % self.args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if not isinstance(eval_dataset, IterableDataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        # Detaching model from device to let the training model attach itself
        if self.args.do_train:
            model.detachFromDevice()

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def save_model(self, output_dir: Optional[str] = None):
        """
        Will save the model, so you can reload it using :obj:`from_pretrained()`.

        Will only save from the main process.
        """

        if output_dir is None:
            output_dir = self.args.output_dir

        if self.args.should_save:
            self._save(output_dir)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        # Updating self.model weights with the weights stored on device.
        if self.model_wrapped.isAttachedToDevice():
            self.model_wrapped.copyWeightsToHost()

        # TODO: should we keep the unwrap checks?
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            if isinstance(unwrap_model(self.model), PreTrainedModel):
                if state_dict is None:
                    state_dict = self.model.state_dict()
                unwrap_model(self.model).save_pretrained(output_dir, state_dict=state_dict)
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if state_dict is None:
                    state_dict = self.model.state_dict()
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            # TODO: make this more efficient.
            deparallelized = copy.deepcopy(self.model).deparallelize()
            deparallelized.save_pretrained(output_dir, state_dict=state_dict)
            # Freeing up memory.
            del deparallelized
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        self.ipu_config.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
