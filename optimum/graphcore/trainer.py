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
import inspect
import math
import os
import random
import re
import shutil
import sys
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from tqdm.auto import tqdm


# Integrations must be imported before ML frameworks:
from transformers.integrations import (  # isort: split
    default_hp_search_backend,
    get_reporting_integration_callbacks,
    hp_params,
    is_fairscale_available,
    is_optuna_available,
    is_ray_tune_available,
    is_sigopt_available,
    run_hp_search_optuna,
    run_hp_search_ray,
    run_hp_search_sigopt,
)

import numpy as np
import torch
from packaging import version
from torch import nn, optim
from torch.utils.data import (
    DataLoader,
    Dataset,
    IterableDataset,
    RandomSampler,
    SequentialSampler,
    SubsetRandomSampler,
)
from torch.utils.data.distributed import DistributedSampler

import poptorch
from huggingface_hub import Repository
from optimum.graphcore.version import __version__
from optimum.utils import logging
from poptorch import DataLoaderMode, PoplarExecutor
from poptorch.optim import LAMB, AdamW
from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.file_utils import CONFIG_NAME, WEIGHTS_NAME, get_full_repo_name, is_datasets_available
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES
from transformers.optimization import get_scheduler
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import OPTIMIZER_NAME, SCALER_NAME, SCHEDULER_NAME, TRAINER_STATE_NAME, TRAINING_ARGS_NAME
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    ShardSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
    reissue_pt_warnings,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    ShardedDDPOption,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    denumpify_detensorize,
    get_last_checkpoint,
    number_of_arguments,
    set_seed,
    speed_metrics,
)
from transformers.training_args import ParallelMode, TrainingArguments

from .ipu_configuration import IPU_CONFIG_NAME, IPUConfig
from .modelcard import IPUTrainingSummary
from .modeling_utils import to_pipelined
from .trainer_utils import _WorkerInit
from .training_args import IPUTrainingArguments


if is_datasets_available():
    import datasets


logger = logging.get_logger(__name__)

_is_torch_generator_available = False

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback


class IPUTrainer:

    from transformers.trainer_pt_utils import log_metrics, metrics_format, save_metrics, save_state

    from .trainer_pt_utils import _get_learning_rate

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
        force_to_pipelined: bool = False,
    ):
        if optimizers != (None, None):
            raise NotImplementedError("providing optimizers to IPUTrainer is not supported yet.")

        if args is None:
            output_dir = "tmp_trainer"
            logger.info(f"No `TrainingArguments` passed, using `output_dir={output_dir}`.")
            args = IPUTrainingArguments(output_dir=output_dir)

        self.args = args
        # Seed must be set before instantiating the model when using model
        set_seed(self.args.seed)
        self.hp_name = None
        self.is_in_train = False

        # memory metrics - must set up as early as possible
        self._memory_tracker = TrainerMemoryTracker(self.args.skip_memory_metrics)
        self._memory_tracker.start()

        # set the correct log level depending on the node
        log_level = args.get_process_log_level()
        logging.set_verbosity(log_level)

        # force device and distributed setup init explicitly
        args._setup_devices

        if model is None:
            if model_init is not None:
                self.model_init = model_init
                model = self.call_model_init()
            else:
                raise RuntimeError("`Trainer` requires either a `model` or `model_init` argument")
        else:
            if model_init is not None:
                warnings.warn(
                    "`Trainer` requires either a `model` or `model_init` argument, but not both. "
                    "`model_init` will overwrite your model when calling the `train` method. This will become a fatal error in the next release.",
                    FutureWarning,
                )
            self.model_init = model_init

        if hasattr(model, "is_parallelizable") and model.is_parallelizable and model.model_parallel:
            self.is_model_parallel = True
        else:
            self.is_model_parallel = False

        default_collator = default_data_collator if tokenizer is None else DataCollatorWithPadding(tokenizer)
        self.data_collator = data_collator if data_collator is not None else default_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        # if self.place_model_on_device:
        #     self._move_model_to_device(model, args.device)

        # later use `self.model is self.model_wrapped` to check if it's wrapped or not
        self.ipu_config = copy.deepcopy(ipu_config).for_pod_type(self.args.pod_type)
        if args.ipu_config_overrides:
            logger.info(f"Overriding IPU config: {args.ipu_config_overrides}")
            self.ipu_config.update_from_string(args.ipu_config_overrides)
        self.ipu_config.seed = self.args.seed
        self.opts = self.ipu_config.to_options()
        self.eval_opts = self.ipu_config.to_options(for_inference=True)

        self.model = to_pipelined(model, self.ipu_config, force=force_to_pipelined)
        self.model.parallelize()

        self.original_model = model

        if not self.args.fp32:
            self.model = self.model.half()

        self.model_wrapped = self.model
        self.training_model = None
        self.inference_model = None

        self.compute_metrics = compute_metrics
        self.optimizer, self.lr_scheduler = optimizers
        if model_init is not None and (self.optimizer is not None or self.lr_scheduler is not None):
            raise RuntimeError(
                "Passing a `model_init` is incompatible with providing the `optimizers` argument. "
                "You should subclass `Trainer` and override the `create_optimizer_and_scheduler` method."
            )
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            callbacks, self.model, self.tokenizer, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)

        # Will be set to True by `self._setup_loggers()` on first call to `self.log()`.
        self._loggers_initialized = False

        # Create clone of distant repo and output directory if needed
        if self.args.push_to_hub:
            self.init_git_repo()

        # if self.args.should_save:
        # os.makedirs(self.args.output_dir, exist_ok=True)
        # path = os.path.join(self.args.output_dir, "executable_cache")
        # logger.info(f"Executable caching enabled, cache directory: {path}")
        # if self.ipu_config.executable_cache_dir is not None:
        #     logger.warning(f"IPUConfig executable_cache_dir was overriden to be: {path}")
        # self.ipu_config.executable_cache_dir = path

        if not callable(self.data_collator) and callable(getattr(self.data_collator, "collate_batch", None)):
            raise ValueError("The `data_collator` should be a simple callable (function, class with `__call__`).")

        if args.max_steps > 0:
            logger.info("max_steps is given, it will override any value given in num_train_epochs")

        if train_dataset is not None and not isinstance(train_dataset, collections.abc.Sized) and args.max_steps <= 0:
            raise ValueError("train_dataset does not implement __len__, max_steps has to be specified")

        self._signature_columns = None

        # Label smoothing
        if self.args.label_smoothing_factor != 0:
            self.label_smoother = LabelSmoother(epsilon=self.args.label_smoothing_factor)
        else:
            self.label_smoother = None

        self.state = TrainerState()
        self.control = TrainerControl()
        # Internal variable to count flos in each process, will be accumulated in `self.state.total_flos` then
        # returned to 0 every time flos need to be logged
        self.current_flos = 0
        self.hp_search_backend = None
        self.use_tune_checkpoints = False
        default_label_names = (
            ["start_positions", "end_positions"]
            if type(self.model).__name__ in MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES.values()
            else ["labels"]
        )
        self.label_names = default_label_names if self.args.label_names is None else self.args.label_names
        self.control = self.callback_handler.on_init_end(self.args, self.state, self.control)

        # very last
        self._memory_tracker.stop_and_update_metrics()

    def _pytorch_optimizer_to_poptorch(self, optimizer: optim.Optimizer):
        # TODO: implement this function
        pytorch_to_poptorch_mapping = {
            optim.SGD: poptorch.optim.SGD,
            optim.Adam: poptorch.optim.Adam,
            optim.AdamW: poptorch.optim.AdamW,
            optim.RMSpop: poptorch.optim.RMSprop,
        }
        pass

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

    def add_callback(self, callback):
        """
        Add a callback to the current list of :class:`~transformer.TrainerCallback`.

        Args:
           callback (:obj:`type` or :class:`~transformer.TrainerCallback`):
               A :class:`~transformer.TrainerCallback` class or an instance of a :class:`~transformer.TrainerCallback`.
               In the first case, will instantiate a member of that class.
        """
        self.callback_handler.add_callback(callback)

    def pop_callback(self, callback):
        """
        Remove a callback from the current list of :class:`~transformer.TrainerCallback` and returns it.

        If the callback is not found, returns :obj:`None` (and no error is raised).

        Args:
           callback (:obj:`type` or :class:`~transformer.TrainerCallback`):
               A :class:`~transformer.TrainerCallback` class or an instance of a :class:`~transformer.TrainerCallback`.
               In the first case, will pop the first member of that class found in the list of callbacks.

        Returns:
            :class:`~transformer.TrainerCallback`: The callback removed, if found.
        """
        return self.callback_handler.pop_callback(callback)

    def remove_callback(self, callback):
        """
        Remove a callback from the current list of :class:`~transformer.TrainerCallback`.

        Args:
           callback (:obj:`type` or :class:`~transformer.TrainerCallback`):
               A :class:`~transformer.TrainerCallback` class or an instance of a :class:`~transformer.TrainerCallback`.
               In the first case, will remove the first member of that class found in the list of callbacks.
        """
        self.callback_handler.remove_callback(callback)

    # TODO: combine this with _compile_model?
    def _move_model_to_device(self, model, device):
        model = model.to(device)
        # Moving a model to an XLA device disconnects the tied weights, so we have to retie them.
        if self.args.parallel_mode == ParallelMode.TPU and hasattr(model, "tie_weights"):
            model.tie_weights()

    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        if not self.args.remove_unused_columns:
            return dataset
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += ["label", "label_ids"]
        columns = [k for k in self._signature_columns if k in dataset.column_names]
        ignored_columns = list(set(dataset.column_names) - set(self._signature_columns))
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set "
            logger.info(
                f"The following columns {dset_description} don't have a corresponding argument in "
                f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
            )

        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if not isinstance(self.train_dataset, collections.abc.Sized):
            return None
        generator = None
        if _is_torch_generator_available:
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))

        combined_batch_size = self.args.per_device_train_batch_size * self.ipu_config.batch_size_factor()

        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None

            # TODO: does this work with an IPU?
            return LengthGroupedSampler(
                combined_batch_size,
                dataset=self.train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
                generator=generator,
            )

        else:
            if self.args.complete_last_batch:
                num_examples = len(self.train_dataset)
                num_missing_examples = num_examples % combined_batch_size
                if num_missing_examples > 0:
                    indices = torch.cat(
                        [torch.arange(num_examples), torch.randint(0, num_examples, size=(num_missing_examples,))]
                    )
                return SubsetRandomSampler(indices, generator)
            return RandomSampler(self.train_dataset)

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

        # TODO: make a better check, any decorator would return True while we really want to test if it was decorated
        # by pad_on_batch_axis.
        should_drop_last = not hasattr(self.data_collator, "__wrapped__") and not self.args.complete_last_batch
        poptorch_specific_kwargs = {
            "drop_last": should_drop_last,  # Not dropping last will end up causing NaN during training if the combined batch size does not divide the number of steps
            "auto_distributed_partitioning": not isinstance(train_dataset, torch.utils.data.IterableDataset),
            "mode": self.args.dataloader_mode,
            "worker_init_fn": _WorkerInit(123),
        }

        # TODO: support for iterable dataset
        # if isinstance(train_dataset, torch.utils.data.IterableDataset):
        #    self.options.Distributed.setNum,
        #    return poptorch.DataLoader(
        #        self.opts,
        #        train_dataset,
        #        batch_size=self.args.train_batch_size,
        #        collate_fn=self.data_collator,
        #        num_workers=self.args.dataloader_num_workers,
        #        pin_memory=self.args.dataloader_pin_memory,
        #        **poptorch_specific_kwargs,
        #    )

        train_sampler = self._get_train_sampler()
        combined_batch_size = self.args.per_device_train_batch_size * self.ipu_config.batch_size_factor()
        rebatched_worker_size = (
            2 * (combined_batch_size // self.args.dataloader_num_workers)
            if self.args.dataloader_num_workers
            else combined_batch_size
        )

        return poptorch.DataLoader(
            self.opts,
            train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            rebatched_worker_size=rebatched_worker_size,
            **poptorch_specific_kwargs,
        )

    def _get_eval_sampler(self, eval_dataset: Dataset) -> Optional[torch.utils.data.Sampler]:
        if self.args.world_size <= 1:
            return SequentialSampler(eval_dataset)

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
            "mode": DataLoaderMode.Sync,
            "worker_init_fn": _WorkerInit(123),
        }

        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")

        # self.data_collator can be a data collator that was wrapped by pad_on_batch_axis, retrieving the original
        # data collator as the wrapped version is only wanted for training.
        data_collator = getattr(self.data_collator, "__wrapped__", self.data_collator)

        # if isinstance(eval_dataset, torch.utils.data.IterableDataset):
        #     # TODO: add support, should be easy.
        #     # raise NotImplementedError("Evluation with IterableDataset not supported yet.")
        #     # if self.args.world_size > 1:
        #     #     eval_dataset = IterableDatasetShard(
        #     #         eval_dataset,
        #     #         batch_size=self.args.eval_batch_size,
        #     #         drop_last=self.args.dataloader_drop_last,
        #     #         num_processes=self.args.world_size,
        #     #         process_index=self.args.process_index,
        #     #     )
        #     return poptorch.DataLoader(
        #         self.eval_opts,
        #         eval_dataset,
        #         batch_size=self.args.per_device_eval_batch_size,
        #         collate_fn=data_collator,
        #         num_workers=self.args.dataloader_num_workers,
        #         pin_memory=self.args.dataloader_pin_memory,
        #     )

        eval_sampler = self._get_eval_sampler(eval_dataset)

        return poptorch.DataLoader(
            self.eval_opts,
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=data_collator,
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
            "mode": DataLoaderMode.Sync,
            "worker_init_fn": _WorkerInit(123),
        }

        if is_datasets_available() and isinstance(test_dataset, datasets.Dataset):
            test_dataset = self._remove_unused_columns(test_dataset, description="test")

        # self.data_collator can be a data collator that was wrapped by pad_on_batch_axis, retrieving the original
        # data collator as the wrapped version is only wanted for training.
        data_collator = getattr(self.data_collator, "__wrapped__", self.data_collator)

        if isinstance(test_dataset, torch.utils.data.IterableDataset):
            # if self.args.world_size > 1:
            #     test_dataset = IterableDatasetShard(
            #         test_dataset,
            #         batch_size=self.args.eval_batch_size,
            #         drop_last=self.args.dataloader_drop_last,
            #         num_processes=self.args.world_size,
            #         process_index=self.args.process_index,
            #     )
            return poptorch.DataLoader(
                self.eval_opts,
                test_dataset,
                batch_size=self.args.per_device_eval_batch_size,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        # Slow things down.
        # test_sampler = self._get_eval_sampler(test_dataset)

        # We use the same batch_size as for eval.
        return poptorch.DataLoader(
            self.eval_opts,
            test_dataset,
            # sampler=test_sampler,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            pin_memory=self.args.dataloader_pin_memory,
            **poptorch_specific_kwargs,
        )

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method (or :obj:`create_optimizer`
        and/or :obj:`create_scheduler`) in a subclass.
        """
        self.create_optimizer()
        self.create_scheduler(num_training_steps=num_training_steps, optimizer=self.optimizer)

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
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
                    "bias_correction": not self.args.lamb_no_bias_correction,
                    "eps": 1e-6,  # TODO: use self.args.adam_epsilon?
                }
            else:
                optimizer_cls = AdamW
                optimizer_kwargs = {
                    # Makes test_trainer.py fail but should be added
                    # "max_grad_norm": self.args.max_grad_norm,
                    "betas": (self.args.adam_beta1, self.args.adam_beta2),
                    "eps": self.args.adam_epsilon,
                    "bias_correction": False,
                }

            # TODO: update the training args
            first_order_type = torch.float16 if self.ipu_config.enable_half_first_order_momentum else torch.float32
            optimizer_kwargs["lr"] = self.args.learning_rate
            optimizer_kwargs["loss_scaling"] = self.args.loss_scaling
            optimizer_kwargs[
                "accum_type"
            ] = torch.float16  # TODO: should take into account if the model is in full or half precision.
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
        optimizer = self.optimizer if optimizer is None else optimizer
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )
            optimizer._step_count = 1

        return self.lr_scheduler

    def num_examples(self, dataloader: DataLoader) -> int:
        """
        Helper to get number of samples in a :class:`~torch.utils.data.DataLoader` by accessing its dataset.

        Will raise an exception if the underlying dataset does not implement method :obj:`__len__`
        """
        return len(dataloader.dataset)

    # TODO: keep this for later.
    # def _hp_search_setup(self, trial: Union["optuna.Trial", Dict[str, Any]]):
    #     """HP search setup code"""
    #     self._trial = trial

    #     if self.hp_search_backend is None or trial is None:
    #         return
    #     if self.hp_search_backend == HPSearchBackend.OPTUNA:
    #         params = self.hp_space(trial)
    #     elif self.hp_search_backend == HPSearchBackend.RAY:
    #         params = trial
    #         params.pop("wandb", None)
    #     elif self.hp_search_backend == HPSearchBackend.SIGOPT:
    #         params = {k: int(v) if isinstance(v, str) else v for k, v in trial.assignments.items()}

    #     for key, value in params.items():
    #         if not hasattr(self.args, key):
    #             logger.warn(
    #                 f"Trying to set {key} in the hyperparameter search but there is no corresponding field in `TrainingArguments`."
    #             )
    #             continue
    #         old_attr = getattr(self.args, key, None)
    #         # Casting value to the proper type
    #         if old_attr is not None:
    #             value = type(old_attr)(value)
    #         setattr(self.args, key, value)
    #     if self.hp_search_backend == HPSearchBackend.OPTUNA:
    #         logger.info("Trial:", trial.params)
    #     if self.hp_search_backend == HPSearchBackend.SIGOPT:
    #         logger.info(f"SigOpt Assignments: {trial.assignments}")
    #     if self.args.deepspeed:
    #         # Rebuild the deepspeed config to reflect the updated training parameters
    #         from transformers.deepspeed import HfDeepSpeedConfig

    #         self.args.hf_deepspeed_config = HfDeepSpeedConfig(self.args)

    # TODO: keep this for later.
    # def _report_to_hp_search(
    #     self, trial: Union["optuna.Trial", Dict[str, Any]], epoch: int, metrics: Dict[str, float]
    # ):
    #     if self.hp_search_backend is None or trial is None:
    #         return
    #     self.objective = self.compute_objective(metrics.copy())
    #     if self.hp_search_backend == HPSearchBackend.OPTUNA:
    #         import optuna

    #         trial.report(self.objective, epoch)
    #         if trial.should_prune():
    #             raise optuna.TrialPruned()
    #     elif self.hp_search_backend == HPSearchBackend.RAY:
    #         from ray import tune

    #         if self.control.should_save:
    #             self._tune_save_checkpoint()
    #         tune.report(objective=self.objective, **metrics)

    # TODO: keep this for later.
    # def _tune_save_checkpoint(self):
    #     from ray import tune

    #     if not self.use_tune_checkpoints:
    #         return
    #     with tune.checkpoint_dir(step=self.state.global_step) as checkpoint_dir:
    #         output_dir = os.path.join(checkpoint_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}")
    #         self.save_model(output_dir)
    #         if self.args.should_save:
    #             self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))
    #             torch.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
    #             torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))

    def call_model_init(self, trial=None):
        model_init_argcount = number_of_arguments(self.model_init)
        if model_init_argcount == 0:
            model = self.model_init()
        elif model_init_argcount == 1:
            model = self.model_init(trial)
        else:
            raise RuntimeError("model_init should have 0 or 1 argument.")

        if model is None:
            raise RuntimeError("model_init should not return None.")

        return model

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
        total_train_batch_size = args.per_device_train_batch_size * self.ipu_config.batch_size_factor()
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
            # TODO: test if this works.
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
        logger.info(f"  Device Iterations = {self.ipu_config.device_iterations}")
        logger.info(f"  Replication Factor = {self.ipu_config.replication_factor}")
        logger.info(f"  Gradient Accumulation steps = {self.ipu_config.gradient_accumulation_steps}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
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

                tr_loss_step = self.training_step(model, inputs)

                if args.logging_nan_inf_filter and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step)):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                # TODO: see how to enable this (if necessary), slows down training a lot.
                self.current_flos += float(self.floating_point_ops(inputs))

                # if (step + 1) % args.gradient_accumulation_steps == 0 or (
                #     # last step in epoch but step is always smaller than gradient_accumulation_steps
                #     steps_in_epoch <= args.gradient_accumulation_steps
                #     and (step + 1) == steps_in_epoch
                # ):

                # TODO: check how gradient clipping is done with poptorch optimizers.
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

                if optimizer_was_run:
                    self.lr_scheduler.step()
                    self.training_model.setOptimizer(self.optimizer)

                # model.zero_grad()
                self.state.global_step += 1
                self.state.epoch = epoch + (step + 1) / steps_in_epoch
                self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

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
        self.model.deparallelize()
        load_result = self.model.load_state_dict(state_dict, strict=False)
        self.model.parallelize()
        if not self.args.fp32:
            self.model.half()

        # TODO: check if this is needed.
        # if self.training_model and self.training_model.isAttachedToDevice():
        #     self.training_model.copyWeightsToDevice()

        # if self.inference_model and self.inference_model.isAttachedToDevice():
        #     self.inference_model.copyWeightsToDevice()
        if len(load_result.missing_keys) != 0:
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
            # self._report_to_hp_search(trial, epoch, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def _load_rng_state(self, checkpoint):
        # Load RNG states from `checkpoint`
        if checkpoint is None:
            return

        # TODO: validate that.
        local_rank = -1
        # local_rank = xm.get_local_ordinal() if is_torch_tpu_available() else self.args.local_rank
        if local_rank != -1:
            rng_file = os.path.join(checkpoint, f"rng_state_{local_rank}.pth")
            if not os.path.isfile(os.path.join(checkpoint, rng_file)):
                logger.info(
                    f"Didn't find an RNG file for process {local_rank}, if you are resuming a training that "
                    "wasn't launched in a distributed fashion, reproducibility is not guaranteed."
                )
                return
        else:
            rng_file = os.path.join(checkpoint, "rng_state.pth")
            if not os.path.isfile(rng_file):
                logger.info(
                    "Didn't find an RNG file, if you are resuming a training that was launched in a distributed "
                    "fashion, reproducibility is not guaranteed."
                )
                return

        checkpoint_rng_state = torch.load(rng_file)
        random.setstate(checkpoint_rng_state["python"])
        np.random.set_state(checkpoint_rng_state["numpy"])
        torch.random.set_rng_state(checkpoint_rng_state["cpu"])
        # TODO: set poptorch rng state?
        # if torch.cuda.is_available():
        #     if self.args.local_rank != -1:
        #         torch.cuda.random.set_rng_state(checkpoint_rng_state["cuda"])
        #     else:
        #         torch.cuda.random.set_rng_state_all(checkpoint_rng_state["cuda"])
        # if is_torch_tpu_available():
        #     xm.set_rng_state(checkpoint_rng_state["xla"])

    def _save_checkpoint(self, model, trial, metrics=None):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is not None and trial is not None:
            if self.hp_search_backend == HPSearchBackend.OPTUNA:
                run_id = trial.number
            elif self.hp_search_backend == HPSearchBackend.RAY:
                from ray import tune

                run_id = tune.get_trial_id()
            elif self.hp_search_backend == HPSearchBackend.SIGOPT:
                run_id = trial.id
            run_name = self.hp_name(trial) if self.hp_name is not None else f"run-{run_id}"
            run_dir = os.path.join(self.args.output_dir, run_name)
        else:
            run_dir = self.args.output_dir
            self.store_flos()

        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir)
        if self.args.should_save:
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
            with warnings.catch_warnings(record=True) as caught_warnings:
                torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
            reissue_pt_warnings(caught_warnings)

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        # Save the Trainer state
        if self.args.should_save:
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        # Save RNG state in non-distributed training
        # TODO: save IPU RNG state once it is available
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        }

        # A process can arrive here before the process 0 has a chance to save the model, in which case output_dir may
        # not yet exist.
        os.makedirs(output_dir, exist_ok=True)
        torch.save(rng_states, os.path.join(output_dir, "rng_state.pth"))

        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)

    def _load_optimizer_and_scheduler(self, checkpoint):
        """If optimizer and scheduler states exist, load them."""
        if checkpoint is None:
            return

        if os.path.isfile(os.path.join(checkpoint, OPTIMIZER_NAME)) and os.path.isfile(
            os.path.join(checkpoint, SCHEDULER_NAME)
        ):
            # TODO: make sure the optimizer and scheduler are properly loaded to the IPU.
            self.optimizer.load_state_dict(torch.load(os.path.join(checkpoint, OPTIMIZER_NAME)))
            with warnings.catch_warnings(record=True) as caught_warnings:
                self.lr_scheduler.load_state_dict(torch.load(os.path.join(checkpoint, SCHEDULER_NAME)))
            reissue_pt_warnings(caught_warnings)

    # TODO: keep this for later.
    # def hyperparameter_search(
    #     self,
    #     hp_space: Optional[Callable[["optuna.Trial"], Dict[str, float]]] = None,
    #     compute_objective: Optional[Callable[[Dict[str, float]], float]] = None,
    #     n_trials: int = 20,
    #     direction: str = "minimize",
    #     backend: Optional[Union["str", HPSearchBackend]] = None,
    #     hp_name: Optional[Callable[["optuna.Trial"], str]] = None,
    #     **kwargs,
    # ) -> BestRun:
    #     """
    #     Launch an hyperparameter search using ``optuna`` or ``Ray Tune`` or ``SigOpt``. The optimized quantity is
    #     determined by :obj:`compute_objective`, which defaults to a function returning the evaluation loss when no
    #     metric is provided, the sum of all metrics otherwise.

    #     .. warning::

    #         To use this method, you need to have provided a ``model_init`` when initializing your
    #         :class:`~transformers.Trainer`: we need to reinitialize the model at each new run. This is incompatible
    #         with the ``optimizers`` argument, so you need to subclass :class:`~transformers.Trainer` and override the
    #         method :meth:`~transformers.Trainer.create_optimizer_and_scheduler` for custom optimizer/scheduler.

    #     Args:
    #         hp_space (:obj:`Callable[["optuna.Trial"], Dict[str, float]]`, `optional`):
    #             A function that defines the hyperparameter search space. Will default to
    #             :func:`~transformers.trainer_utils.default_hp_space_optuna` or
    #             :func:`~transformers.trainer_utils.default_hp_space_ray` or
    #             :func:`~transformers.trainer_utils.default_hp_space_sigopt` depending on your backend.
    #         compute_objective (:obj:`Callable[[Dict[str, float]], float]`, `optional`):
    #             A function computing the objective to minimize or maximize from the metrics returned by the
    #             :obj:`evaluate` method. Will default to :func:`~transformers.trainer_utils.default_compute_objective`.
    #         n_trials (:obj:`int`, `optional`, defaults to 100):
    #             The number of trial runs to test.
    #         direction(:obj:`str`, `optional`, defaults to :obj:`"minimize"`):
    #             Whether to optimize greater or lower objects. Can be :obj:`"minimize"` or :obj:`"maximize"`, you should
    #             pick :obj:`"minimize"` when optimizing the validation loss, :obj:`"maximize"` when optimizing one or
    #             several metrics.
    #         backend(:obj:`str` or :class:`~transformers.training_utils.HPSearchBackend`, `optional`):
    #             The backend to use for hyperparameter search. Will default to optuna or Ray Tune or SigOpt, depending
    #             on which one is installed. If all are installed, will default to optuna.
    #         kwargs:
    #             Additional keyword arguments passed along to :obj:`optuna.create_study` or :obj:`ray.tune.run`. For
    #             more information see:

    #             - the documentation of `optuna.create_study
    #               <https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.create_study.html>`__
    #             - the documentation of `tune.run
    #               <https://docs.ray.io/en/latest/tune/api_docs/execution.html#tune-run>`__
    #             - the documentation of `sigopt <https://app.sigopt.com/docs/endpoints/experiments/create>`__

    #     Returns:
    #         :class:`transformers.trainer_utils.BestRun`: All the information about the best run.
    #     """
    #     if backend is None:
    #         backend = default_hp_search_backend()
    #         if backend is None:
    #             raise RuntimeError(
    #                 "At least one of optuna or ray should be installed. "
    #                 "To install optuna run `pip install optuna`. "
    #                 "To install ray run `pip install ray[tune]`. "
    #                 "To install sigopt run `pip install sigopt`."
    #             )
    #     backend = HPSearchBackend(backend)
    #     if backend == HPSearchBackend.OPTUNA and not is_optuna_available():
    #         raise RuntimeError("You picked the optuna backend, but it is not installed. Use `pip install optuna`.")
    #     if backend == HPSearchBackend.RAY and not is_ray_tune_available():
    #         raise RuntimeError(
    #             "You picked the Ray Tune backend, but it is not installed. Use `pip install 'ray[tune]'`."
    #         )
    #     if backend == HPSearchBackend.SIGOPT and not is_sigopt_available():
    #         raise RuntimeError("You picked the sigopt backend, but it is not installed. Use `pip install sigopt`.")
    #     self.hp_search_backend = backend
    #     if self.model_init is None:
    #         raise RuntimeError(
    #             "To use hyperparameter search, you need to pass your model through a model_init function."
    #         )

    #     self.hp_space = default_hp_space[backend] if hp_space is None else hp_space
    #     self.hp_name = hp_name
    #     self.compute_objective = default_compute_objective if compute_objective is None else compute_objective

    #     backend_dict = {
    #         HPSearchBackend.OPTUNA: run_hp_search_optuna,
    #         HPSearchBackend.RAY: run_hp_search_ray,
    #         HPSearchBackend.SIGOPT: run_hp_search_sigopt,
    #     }
    #     best_run = backend_dict[backend](self, n_trials, direction, **kwargs)

    #     self.hp_search_backend = None
    #     return best_run

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log :obj:`logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (:obj:`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def _prepare_input(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        """
        Prepares one :obj:`data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, dict):
            return type(data)(**{k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            # TODO: check that.
            # kwargs = dict(device=self.args.device)
            # return data.to(**kwargs)
            return data
        return data

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs)
        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        return inputs

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
        inputs = self._prepare_inputs(inputs)
        loss = self.compute_loss(model, inputs)
        loss = loss.mean()
        return loss

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    # TODO: should we keep this?
    def is_local_process_zero(self) -> bool:
        """
        Whether or not this process is the local (e.g., on one machine if training in a distributed fashion on several
        machines) main process.
        """
        return self.args.local_process_index == 0

    # TODO: should we keep this?
    def is_world_process_zero(self) -> bool:
        """
        Whether or not this process is the global main process (when training in a distributed fashion on several
        machines, this is only going to be :obj:`True` for one process).
        """
        # Special case for SageMaker ModelParallel since there process_index is dp_process_index, not the global
        # process index.
        # if is_sagemaker_mp_enabled():
        #     return smp.rank() == 0
        # else:
        return self.args.process_index == 0

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

        if not isinstance(self.model, PreTrainedModel):
            logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
            if state_dict is None:
                state_dict = self.model.state_dict()
            torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.deparallelize()
            self.model.save_pretrained(output_dir, state_dict=state_dict)
            self.model.parallelize()

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        self.ipu_config.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def store_flos(self):
        # Storing the number of floating-point operations that went into the model
        # TODO: Validate that this is right.
        self.state.total_flos += self.current_flos * self.ipu_config.batch_size_factor()
        self.current_flos = 0
        # Original Trainer implementation:
        # if self.args.local_rank != -1:
        #     self.state.total_flos += (
        #         distributed_broadcast_scalars([self.current_flos], device=self.args.device).sum().item()
        #     )
        #     self.current_flos = 0
        # else:
        #     self.state.total_flos += self.current_flos
        #     self.current_flos = 0

    def _sorted_checkpoints(
        self, output_dir=None, checkpoint_prefix=PREFIX_CHECKPOINT_DIR, use_mtime=False
    ) -> List[str]:
        ordering_and_checkpoint_path = []

        glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{checkpoint_prefix}-*")]

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
                if regex_match is not None and regex_match.groups() is not None:
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        # Make sure we don't delete the best model.
        if self.state.best_model_checkpoint is not None:
            best_model_index = checkpoints_sorted.index(str(Path(self.state.best_model_checkpoint)))
            for i in range(best_model_index, len(checkpoints_sorted) - 2):
                checkpoints_sorted[i], checkpoints_sorted[i + 1] = checkpoints_sorted[i + 1], checkpoints_sorted[i]
        return checkpoints_sorted

    def _rotate_checkpoints(self, use_mtime=False, output_dir=None) -> None:
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=use_mtime, output_dir=output_dir)
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        # If save_total_limit=1 with load_best_model_at_end=True, we could end up deleting the last checkpoint, which
        # we don't do to allow resuming.
        save_total_limit = self.args.save_total_limit
        if (
            self.state.best_model_checkpoint is not None
            and self.args.save_total_limit == 1
            and checkpoints_sorted[-1] != self.state.best_model_checkpoint
        ):
            save_total_limit = 2

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
            shutil.rmtree(checkpoint)

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement the
                :obj:`__len__` method.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        output = self.evaluation_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.per_device_eval_batch_size * self.ipu_config.batch_size_factor(for_inference=True)
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def predict(
        self, test_dataset: Dataset, ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = "test"
    ) -> PredictionOutput:
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in :obj:`evaluate()`.

        Args:
            test_dataset (:obj:`Dataset`):
                Dataset to run the predictions on. If it is an :obj:`datasets.Dataset`, columns not accepted by the
                ``model.forward()`` method are automatically removed. Has to implement the method :obj:`__len__`
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"test"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "test_bleu" if the prefix is "test" (default)

        .. note::

            If your predictions or labels have different sequence length (for instance because you're doing dynamic
            padding in a token classification task) the predictions will be padded (on the right) to allow for
            concatenation into one array. The padding index is -100.

        Returns: `NamedTuple` A namedtuple with the following keys:

            - predictions (:obj:`np.ndarray`): The predictions on :obj:`test_dataset`.
            - label_ids (:obj:`np.ndarray`, `optional`): The labels (if the dataset contained some).
            - metrics (:obj:`Dict[str, float]`, `optional`): The potential dictionary of metrics (if the dataset
              contained labels).
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()

        output = self.evaluation_loop(
            test_dataloader, description="Prediction", ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
        )
        total_batch_size = self.args.eval_batch_size * self.ipu_config.batch_size_factor(for_inference=True)
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return PredictionOutput(predictions=output.predictions, label_ids=output.label_ids, metrics=output.metrics)

    def evaluation_loop(
        self,
        dataloader: poptorch.DataLoader,
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
            loss, logits, labels = self.prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys, is_last_batch=step == len(dataloader) - 1
            )

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
        model.detachFromDevice()

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def _nested_gather(self, tensors, name=None):
        """
        Gather value of `tensors` (tensor or list/tuple of nested tensors) and convert them to numpy before
        concatenating them to `gathered`
        """
        if tensors is None:
            return
        # TODO: this can be removed.
        # if is_torch_tpu_available():
        #     if name is None:
        #         name = "nested_gather"
        #     tensors = nested_xla_mesh_reduce(tensors, name)
        # elif is_sagemaker_mp_enabled():
        #     tensors = smp_gather(tensors)
        # elif self.args.local_rank != -1:
        #     tensors = distributed_concat(tensors)
        return tensors

    # Copied from Accelerate.
    def _pad_across_processes(self, tensor, pad_index=-100):
        """
        Recursively pad the tensors in a nested list/tuple/dictionary of tensors from all devices to the same size so
        they can safely be gathered.
        """
        if isinstance(tensor, (list, tuple)):
            return type(tensor)(self._pad_across_processes(t, pad_index=pad_index) for t in tensor)
        elif isinstance(tensor, dict):
            return type(tensor)({k: self._pad_across_processes(v, pad_index=pad_index) for k, v in tensor.items()})
        elif not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"Can't pad the values of type {type(tensor)}, only of nested list/tuple/dicts of tensors."
            )

        if len(tensor.shape) < 2:
            return tensor
        # Gather all sizes
        size = torch.tensor(tensor.shape, device=tensor.device)[None]
        sizes = self._nested_gather(size).cpu()

        max_size = max(s[1] for s in sizes)
        if tensor.shape[1] == max_size:
            return tensor

        # Then pad to the maximum size
        old_size = tensor.shape
        new_size = list(old_size)
        new_size[1] = max_size
        new_tensor = tensor.new_zeros(tuple(new_size)) + pad_index
        new_tensor[:, : old_size[1]] = tensor
        return new_tensor

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        is_last_batch: bool = False,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if has_labels:
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                # If last batch is incomplete, some losses might be NaN because nothing was computed on the
                # corresponding POD, ignoring them is necessary to not mess up evaluation loss computation
                if is_last_batch:
                    loss = loss[~loss.isnan()]
                loss = loss.mean().detach()
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                else:
                    logits = outputs[1:]
            else:
                loss = None
                outputs = model(**inputs)
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                else:
                    logits = outputs
                # TODO: this needs to be fixed and made cleaner later.
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)

    def floating_point_ops(self, inputs: Dict[str, Union[torch.Tensor, Any]]):
        """
        For models that inherit from :class:`~transformers.PreTrainedModel`, uses that method to compute the number of
        floating point operations for every backward + forward pass. If using another model, either implement such a
        method in the model or subclass and override this method.

        Args:
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

        Returns:
            :obj:`int`: The number of floating-point operations.
        """
        # Using self.original_model because self.model is the underlying model used by the IPUs
        # and calling floating_point_ops on it slows things down a lot.
        if hasattr(self.original_model, "floating_point_ops"):
            return self.original_model.floating_point_ops(inputs)
        else:
            return 0

    def init_git_repo(self):
        """
        Initializes a git repo in :obj:`self.args.hub_model_id`.
        """
        if not self.is_world_process_zero():
            return
        use_auth_token = True if self.args.hub_token is None else self.args.hub_token
        if self.args.hub_model_id is None:
            repo_name = Path(self.args.output_dir).absolute().name
        else:
            repo_name = self.args.hub_model_id
        if "/" not in repo_name:
            repo_name = get_full_repo_name(repo_name, token=self.args.hub_token)

        try:
            self.repo = Repository(
                self.args.output_dir,
                clone_from=repo_name,
                use_auth_token=use_auth_token,
            )
        except EnvironmentError:
            if self.args.overwrite_output_dir:
                # Try again after wiping output_dir
                shutil.rmtree(self.args.output_dir)
                self.repo = Repository(
                    self.args.output_dir,
                    clone_from=repo_name,
                    use_auth_token=use_auth_token,
                )
            else:
                raise

        self.repo.git_pull()

        # By default, ignore the checkpoint folders
        if (
            not os.path.exists(os.path.join(self.args.output_dir, ".gitignore"))
            and self.args.hub_strategy != HubStrategy.ALL_CHECKPOINTS
        ):
            with open(os.path.join(self.args.output_dir, ".gitignore"), "w", encoding="utf-8") as writer:
                writer.writelines(["checkpoint-*/"])

        self.push_in_progress = None

    def create_model_card(
        self,
        language: Optional[str] = None,
        license: Optional[str] = None,
        tags: Optional[str] = None,
        model_name: Optional[str] = None,
        finetuned_from: Optional[str] = None,
        tasks: Optional[str] = None,
        dataset_tags: Optional[Union[str, List[str]]] = None,
        dataset: Optional[Union[str, List[str]]] = None,
        dataset_args: Optional[Union[str, List[str]]] = None,
    ):
        training_summary = IPUTrainingSummary.from_trainer(
            self,
            language=language,
            license=license,
            tags=tags,
            model_name=model_name,
            finetuned_from=finetuned_from,
            tasks=tasks,
            dataset_tags=dataset_tags,
            dataset=dataset,
            dataset_args=dataset_args,
        )
        model_card = training_summary.to_model_card()
        with open(os.path.join(self.args.output_dir, "README.md"), "w") as f:
            f.write(model_card)

    def _push_from_checkpoint(self, checkpoint_folder):
        # Only push from one node.
        if not self.is_world_process_zero() or self.args.hub_strategy == HubStrategy.END:
            return
        # If we haven't finished the last push, we don't do this one.
        if self.push_in_progress is not None and not self.push_in_progress.is_done:
            return

        output_dir = self.args.output_dir
        # To avoid a new synchronization of all model weights, we just copy the file from the checkpoint folder
        modeling_files = [CONFIG_NAME, WEIGHTS_NAME, IPU_CONFIG_NAME]
        for modeling_file in modeling_files:
            if os.path.isfile(os.path.join(checkpoint_folder, modeling_file)):
                shutil.copy(os.path.join(checkpoint_folder, modeling_file), os.path.join(output_dir, modeling_file))
        # Saving the tokenizer is fast and we don't know how many files it may have spawned, so we resave it to be sure.
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        # Same for the training arguments
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

        try:
            if self.args.hub_strategy == HubStrategy.CHECKPOINT:
                # Temporarily move the checkpoint just saved for the push
                tmp_checkpoint = os.path.join(output_dir, "last-checkpoint")
                # We have to remove the "last-checkpoint" dir if it exists, otherwise the checkpoint is moved as a
                # subfolder.
                if os.path.isdir(tmp_checkpoint):
                    shutil.rmtree(tmp_checkpoint)
                shutil.move(checkpoint_folder, tmp_checkpoint)

            if self.args.save_strategy == IntervalStrategy.STEPS:
                commit_message = f"Training in progress, step {self.state.global_step}"
            else:
                commit_message = f"Training in progress, epoch {int(self.state.epoch)}"
            _, self.push_in_progress = self.repo.push_to_hub(
                commit_message=commit_message, blocking=False, auto_lfs_prune=True
            )
        finally:
            if self.args.hub_strategy == HubStrategy.CHECKPOINT:
                # Move back the checkpoint to its place
                shutil.move(tmp_checkpoint, checkpoint_folder)

    def push_to_hub(self, commit_message: Optional[str] = "End of training", blocking: bool = True, **kwargs) -> str:
        """
        Upload `self.model` and `self.tokenizer` to the 🤗 model hub on the repo `self.args.hub_model_id`.

        Parameters:
            commit_message (:obj:`str`, `optional`, defaults to :obj:`"End of training"`):
                Message to commit while pushing.
            blocking (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether the function should return only when the :obj:`git push` has finished.
            kwargs:
                Additional keyword arguments passed along to :meth:`~transformers.Trainer.create_model_card`.

        Returns:
            The url of the commit of your model in the given repository if :obj:`blocking=False`, a tuple with the url
            of the commit and an object to track the progress of the commit if :obj:`blocking=True`
        """

        if self.args.should_save:
            if self.args.hub_model_id is None:
                model_name = Path(self.args.output_dir).name
            else:
                model_name = self.args.hub_model_id.split("/")[-1]
        # Needs to be executed on all processes for TPU training, but will only save on the processed determined by
        # self.args.should_save.
        self.save_model()

        # Only push from one node.
        if not self.is_world_process_zero():
            return

        git_head_commit_url = self.repo.push_to_hub(
            commit_message=commit_message, blocking=blocking, auto_lfs_prune=True
        )
        # push separately the model card to be independant from the rest of the model
        if self.args.should_save:
            self.create_model_card(model_name=model_name, **kwargs)
            try:
                self.repo.push_to_hub(
                    commit_message="update model card README.md", blocking=blocking, auto_lfs_prune=True
                )
            except EnvironmentError as exc:
                logger.error(f"Error pushing update to the model card. Please read logs and retry.\n${exc}")

        return git_head_commit_url
