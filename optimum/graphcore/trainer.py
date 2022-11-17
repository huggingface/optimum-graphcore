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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from tqdm.auto import tqdm


# Integrations must be imported before ML frameworks:
from transformers.integrations import (  # isort: split
    get_reporting_integration_callbacks,
)

import numpy as np
import torch
from packaging import version
from torch import nn, optim
from torch.utils.data import Dataset, RandomSampler, SequentialSampler, SubsetRandomSampler
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
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    MODEL_FOR_PRETRAINING_MAPPING,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES,
)
from transformers.optimization import get_scheduler
from transformers.pytorch_utils import is_torch_less_than_1_11
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import OPTIMIZER_NAME, SCHEDULER_NAME, TRAINER_STATE_NAME, TRAINING_ARGS_NAME
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
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
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
    EvalLoopOutput,
    EvalPrediction,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    TrainerMemoryTracker,
    TrainOutput,
    denumpify_detensorize,
    get_last_checkpoint,
    has_length,
    set_seed,
    speed_metrics,
)
from transformers.utils import (
    CONFIG_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    find_labels,
    get_full_repo_name,
    is_datasets_available,
)

from .data.data_collator import pad_on_batch_axis
from .ipu_configuration import IPU_CONFIG_NAME, IPUConfig
from .modelcard import IPUTrainingSummary
from .modeling_utils import to_pipelined
from .trainer_utils import _WorkerInit
from .training_args import IPUTrainingArguments


if is_datasets_available():
    import datasets


if TYPE_CHECKING:
    import optuna

logger = logging.get_logger(__name__)

_is_torch_generator_available = False

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

TIED_WEIGHT_MODELS = set(
    MODEL_FOR_PRETRAINING_MAPPING.values()
    + MODEL_FOR_MASKED_LM_MAPPING.values()
    + MODEL_FOR_CAUSAL_LM_MAPPING.values()
)


@dataclass
class IPUTrainerState(TrainerState):
    start_time: float = -1.0


class IPUTrainer:
    """
    IPUTrainer is a simple but feature-complete training and eval loop  on Graphcore IPUs for PyTorch, optimized for
    🤗 Transformers.

    Args:
        model ([`transformers.PreTrainedModel`] or `torch.nn.Module`, *optional*):
            The model to train, evaluate or use for predictions. If not provided, a `model_init` must be passed.

            <Tip>

            [`IPUTrainer`] is optimized to work with the [`transformers.PreTrainedModel`] provided by the 🤗 Transformers
            library. You can still use your own models defined as `torch.nn.Module` as long as they work the same way as
            the 🤗 Transformers models.

            </Tip>

        args ([`IPUTrainingArguments`], *optional*):
            The arguments to tweak for training. Will default to a basic instance of [`IPUTrainingArguments`] with the
            `output_dir` set to a directory named *tmp_trainer* in the current directory if not provided.
        data_collator ([`transformers.data.data_collator.DataCollator`], *optional*):
            The function to use to form a batch from a list of elements of `train_dataset` or `eval_dataset`. Will
            default to [`transformers.data.default_data_collator`] if no `tokenizer` is provided, an instance of
            [`~transformers.data.DataCollatorWithPadding`] otherwise.
        train_dataset (`torch.utils.data.Dataset` or `torch.utils.data.IterableDataset`, *optional*):
            The dataset to use for training. If it is a [`~datasets.Dataset`], columns not accepted by the
            `model.forward()` method are automatically removed.

            Note that if it's a `torch.utils.data.IterableDataset` with some randomization and you are training in a
            distributed fashion, your iterable dataset should either use a internal attribute `generator` that is a
            `torch.Generator` for the randomization that must be identical on all processes (and the Trainer will
            manually set the seed of this `generator` at each epoch) or have a `set_epoch()` method that internally
            sets the seed of the RNGs used.
        eval_dataset (Union[`torch.utils.data.Dataset`, Dict[str, `torch.utils.data.Dataset`]), *optional*):
             The dataset to use for evaluation. If it is a [`~datasets.Dataset`], columns not accepted by the
             `model.forward()` method are automatically removed. If it is a dictionary, it will evaluate on each
             dataset prepending the dictionary key to the metric name.
        tokenizer ([`transformers.PreTrainedTokenizerBase`], *optional*):
            The tokenizer used to preprocess the data. If provided, will be used to automatically pad the inputs the
            maximum length when batching inputs, and it will be saved along the model to make it easier to rerun an
            interrupted training or reuse the fine-tuned model.
        model_init (`Callable[[], transformers.PreTrainedModel]`, *optional*):
            A function that instantiates the model to be used. If provided, each call to [`IPUTrainer.train`] will start
            from a new instance of the model as given by this function.

            The function may have zero argument, or a single one containing the optuna/Ray Tune/SigOpt trial object, to
            be able to choose different architectures according to hyper parameters (such as layer count, sizes of
            inner layers, dropout probabilities etc). **Note: this feature is not supported for now.**

        compute_metrics (`Callable[[~transformers.trainer_utils.EvalPrediction], Dict]`, *optional*):
            The function that will be used to compute metrics at evaluation. Must take a
            [`~transformers.trainer_utils.EvalPrediction`] and return a dictionary string to metric values.
        callbacks (List of [`transformers.trainer_callback.TrainerCallback`], *optional*):
            A list of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](callback).

            If you want to remove one of the default callbacks used, use the [`Trainer.remove_callback`] method.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*): A tuple
            containing the optimizer and the scheduler to use. Will default to an instance of `poptorch.AdamW` on your model
            and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`, *optional*):
            A function that preprocess the logits right before caching them at each evaluation step. Must take two
            tensors, the logits and the labels, and return the logits once processed as desired. The modifications made
            by this function will be reflected in the predictions received by `compute_metrics`.

            Note that the labels (second parameter) will be `None` if the dataset does not have them.
    """

    from transformers.trainer_pt_utils import log_metrics, metrics_format, save_metrics, save_state

    from .trainer_pt_utils import _get_learning_rate

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        ipu_config: IPUConfig = None,
        args: IPUTrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        eval_data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
        force_to_pipelined: bool = False,
    ):
        if args is None:
            output_dir = "tmp_trainer"
            logger.info(f"No `TrainingArguments` passed, using `output_dir={output_dir}`.")
            args = IPUTrainingArguments(output_dir=output_dir)
        self.args = args
        # Seed must be set before instantiating the model when using model
        set_seed(self.args.seed)
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
                raise RuntimeError("`model_init` is not supported by the `IPUTrainer` yet")
            else:
                raise RuntimeError("`IPUTrainer` requires either a `model` or `model_init` argument")
        else:
            if model_init is not None:
                warnings.warn(
                    "`IPUTrainer` requires either a `model` or `model_init` argument, but not both. "
                    "`model_init` will overwrite your model when calling the `train` method. This will become a fatal error in the next release.",
                    FutureWarning,
                )
            self.model_init = model_init

        # TODO: not sure about setting the data_collator?
        default_collator = default_data_collator if tokenizer is None else DataCollatorWithPadding(tokenizer)
        self.data_collator = data_collator if data_collator is not None else default_collator
        # If no eval_data_collator is specified then use the train data_collator
        self.eval_data_collator = eval_data_collator if eval_data_collator is not None else self.data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        self.ipu_config = copy.deepcopy(ipu_config).for_pod_type(self.args.pod_type)
        if self.ipu_config.replication_factor > 1 or self.ipu_config.inference_replication_factor > 1:
            os.environ["TOKENIZERS_PARALLELISM"] = "true"
        if args.ipu_config_overrides:
            logger.info(f"Overriding IPU config: {args.ipu_config_overrides}")
            self.ipu_config.update_from_string(args.ipu_config_overrides)
        self.ipu_config.seed = self.args.seed
        self.opts = self.ipu_config.to_options(compile_only=args.compile_only)
        self.eval_opts = self.ipu_config.to_options(for_inference=True, compile_only=args.compile_only)

        # If batch axis padding enabled, wrap train/eval data collators with `pad_on_batch_axis` wrapper
        if self.args.pad_on_batch_axis:
            logger.info(
                "Padding on batch axis enabled, each batch feeded to the compiled model during training will have the proper size"
            )
            if self.args.do_train:
                data_collator_wrapper = pad_on_batch_axis(
                    self.args.per_device_train_batch_size * self.ipu_config.batch_size_factor()
                )
                self.data_collator = data_collator_wrapper(self.data_collator)

            if self.args.do_eval:
                data_collator_wrapper = pad_on_batch_axis(
                    self.args.per_device_eval_batch_size * self.ipu_config.batch_size_factor(for_inference=True),
                )
                self.eval_data_collator = data_collator_wrapper(self.eval_data_collator)

        self.model = to_pipelined(model, self.ipu_config, force=force_to_pipelined)
        self.model.parallelize()

        self.original_model = model

        if not self.args.fp32:
            self.model = self.model.half()

        self.training_model = None
        self.inference_model = None

        self.compute_metrics = compute_metrics
        self.preprocess_logits_for_metrics = preprocess_logits_for_metrics
        self.optimizer, self.lr_scheduler = optimizers
        if model_init is not None and (self.optimizer is not None or self.lr_scheduler is not None):
            raise RuntimeError(
                "Passing a `model_init` is incompatible with providing the `optimizers` argument. "
                "You should subclass `Trainer` and override the `create_optimizer_and_scheduler` method."
            )

        if self.optimizer is not None and not isinstance(self.optimizer, poptorch.optim.Optimizer):
            self.optimizer = self.pytorch_optimizer_to_poptorch(self.optimizer, model, self.model)

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

        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)

        if not callable(self.data_collator) and callable(getattr(self.data_collator, "collate_batch", None)):
            raise ValueError("The `data_collator` should be a simple callable (function, class with `__call__`).")

        if not callable(self.eval_data_collator) and callable(getattr(self.eval_data_collator, "collate_batch", None)):
            raise ValueError("The `eval_data_collator` should be a simple callable (function, class with `__call__`).")

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

        self.state = IPUTrainerState()
        self.control = TrainerControl()
        # Internal variable to count flos in each process, will be accumulated in `self.state.total_flos` then
        # returned to 0 every time flos need to be logged
        self.current_flos = 0
        self.hp_search_backend = None
        self.use_tune_checkpoints = False
        default_label_names = find_labels(model.__class__)
        self.label_names = default_label_names if self.args.label_names is None else self.args.label_names
        self.control = self.callback_handler.on_init_end(self.args, self.state, self.control)

        # very last
        self._memory_tracker.stop_and_update_metrics()

        # If compile-only then compile and exit
        if args.compile_only:
            logger.info("Called with compile_only=True. Compiling models then exiting.")
            if args.do_train:
                train_dl = self.get_train_dataloader()
                model = self.wrap_model(self.model)
                self.compile_model(model, next(iter(train_dl)), log=True)
            if args.do_eval:
                # Same thing with _wrap_and_compile_for_evaluation
                eval_dl = self.get_eval_dataloader()
                model = self._wrap_and_compile_model_for_evaluation(eval_dl, False)
            logger.info("Exiting after compiling models with compile_only=True")
            sys.exit(0)

    def pytorch_optimizer_to_poptorch(
        self,
        optimizer: optim.Optimizer,
        model: Union[PreTrainedModel, nn.Module],
        pipelined_model: Union[PreTrainedModel, nn.Module],
    ) -> poptorch.optim.Optimizer:
        """
        Converts a PyTorch optimizer to a PopTorch optimizer.

        Args:
            optimizer (`torch.optim.Optimizer`):
                The PyTorch optimizer to convert.
            model (`[transformers.PreTrainedModel]` or `torch.nn.Module`):
                The original model the optimizer has parameter references to.
            pipelined_model (`[transformers.PreTrainedModel] or `torch.nn.Module`):
                The pipelined version of the model, its parameters will be used by the poptorch optimizer.

        Returns:
            `poptorch.optim.Optimizer`: The converted poptorch optimizer.
        """
        first_order_type = torch.float32 if self.args.fp32 else torch.float16
        optimizer_kwargs = {
            "loss_scaling": self.args.loss_scaling,
            "accum_type": first_order_type,
            "first_order_momentum_accum_type": first_order_type,
            "second_order_momentum_accum_type": torch.float32,
        }
        # TODO: disabled max_grad_norm because it make things fail, fix it.
        max_grad_norm = self.args.max_grad_norm
        self.args.max_grad_norm = None
        pytorch_to_poptorch_mapping = {
            optim.SGD: (poptorch.optim.SGD, {"loss_scaling": self.args.loss_scaling}),
            optim.Adam: (poptorch.optim.Adam, {"max_grad_norm": self.args.max_grad_norm, **optimizer_kwargs}),
            optim.AdamW: (poptorch.optim.AdamW, {"max_grad_norm": self.args.max_grad_norm, **optimizer_kwargs}),
            optim.RMSprop: (poptorch.optim.RMSprop, optimizer_kwargs),
        }
        self.args.max_grad_norm = max_grad_norm
        poptorch_optimizer_cls, kwargs = pytorch_to_poptorch_mapping.get(optimizer.__class__, (None, {}))
        if poptorch_optimizer_cls is None:
            raise KeyError(f"Could not find a poptorch counterpart for optimizer {optimizer.__class__.__name__}")

        # Some dummy value that should be overriden by the real value with .load_state_dict, using some absurd value to
        # make clear if the value is not properly overriden.
        dummy_lr = 1e4
        poptorch_optimizer = poptorch_optimizer_cls(optimizer.param_groups, lr=dummy_lr, **kwargs)
        poptorch_optimizer.load_state_dict({"ipu_state": None, "ipu_param": None, **optimizer.state_dict()})

        # Currently poptorch_optimizer contains references to the original model parameters, so we need to change those
        # to references to the pipelined model parameters.
        id2name = {id(param): name for name, param in model.named_parameters()}
        name2param = dict(pipelined_model.named_parameters())
        for group in poptorch_optimizer.param_groups:
            for idx, param in enumerate(group["params"]):
                group["params"][idx] = name2param[id2name[id(param)]]

        return poptorch_optimizer

    def compile_model(
        self,
        model: poptorch.PoplarExecutor,
        sample_batch: Union[Dict[str, torch.Tensor], Tuple[torch.Tensor]],
        log: bool = False,
    ):
        """
        Compiles the model with PopTorch.

        Args:
            model (`poptorch.PoplarExecutor`):
                The model to compile (already wrapped).
            sample_batch (`Dict[str, torch.Tensor]` or `Tuple[torch.Tensor]`):
                The inputs to use the compilation, this will set the input shapes that the compiled model can accept.
            log (`bool`, *optional*, defaults to `False`):
                Whether to log that compilation is happening or not.
        """
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
        Adds a callback to the current list of [`~transformer.TrainerCallback`].

        Args:
           callback (`type` or [`~transformer.TrainerCallback`]):
               A [`~transformer.TrainerCallback`] class or an instance of a [`~transformer.TrainerCallback`]. In the
               first case, will instantiate a member of that class.
        """
        self.callback_handler.add_callback(callback)

    def pop_callback(self, callback):
        """
        Removes a callback from the current list of [`~transformer.TrainerCallback`] and returns it.

        If the callback is not found, returns `None` (and no error is raised).

        Args:
           callback (`type` or [`~transformer.TrainerCallback`]):
               A [`~transformer.TrainerCallback`] class or an instance of a [`~transformer.TrainerCallback`]. In the
               first case, will pop the first member of that class found in the list of callbacks.

        Returns:
            [`~transformer.TrainerCallback`]: The callback removed, if found.
        """
        return self.callback_handler.pop_callback(callback)

    def remove_callback(self, callback):
        """
        Removes a callback from the current list of [`~transformer.TrainerCallback`].

        Args:
           callback (`type` or [`~transformer.TrainerCallback`]):
               A [`~transformer.TrainerCallback`] class or an instance of a [`~transformer.TrainerCallback`]. In the
               first case, will remove the first member of that class found in the list of callbacks.
        """
        self.callback_handler.remove_callback(callback)

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))

    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        if not self.args.remove_unused_columns:
            return dataset
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns

        ignored_columns = list(set(dataset.column_names) - set(self._signature_columns))
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set "
            logger.info(
                f"The following columns {dset_description} don't have a corresponding argument in "
                f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
            )

        columns = [k for k in signature_columns if k in dataset.column_names]

        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)

    def _get_collator_with_removed_columns(
        self, data_collator: Callable, description: Optional[str] = None
    ) -> Callable:
        """Wraps the data collator in a callable removing unused columns."""
        if not self.args.remove_unused_columns:
            return data_collator
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns

        remove_columns_collator = RemoveColumnsCollator(
            data_collator=data_collator,
            signature_columns=signature_columns,
            logger=logger,
            description=description,
            model_name=self.model.__class__.__name__,
        )
        return remove_columns_collator

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if not isinstance(self.train_dataset, collections.abc.Sized):
            return None
        generator = None
        if _is_torch_generator_available:
            generator = torch.Generator()
            # for backwards compatibility, we generate a seed here (which is sampled from a generator seeded with
            # `args.seed`) if data_seed isn't provided.
            # Further on in this method, we default to `args.seed` instead.
            if self.args.data_seed is None:
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
            else:
                seed = self.args.data_seed
            generator.manual_seed(seed)

        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed
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

            return LengthGroupedSampler(
                combined_batch_size,
                dataset=self.train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
                generator=generator,
            )

        else:
            return RandomSampler(self.train_dataset)

    def get_train_dataloader(self) -> poptorch.DataLoader:
        """
        Returns the training `poptorch.DataLoader`.

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        poptorch_specific_kwargs = {
            "auto_distributed_partitioning": not isinstance(train_dataset, torch.utils.data.IterableDataset),
            "mode": self.args.dataloader_mode,
            "worker_init_fn": _WorkerInit(123),
        }

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            return poptorch.DataLoader(
                self.opts,
                train_dataset,
                batch_size=self.args.train_batch_size,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                drop_last=self.args.dataloader_drop_last,
                pin_memory=self.args.dataloader_pin_memory,
                **poptorch_specific_kwargs,
            )

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
        return SequentialSampler(eval_dataset)

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> poptorch.DataLoader:
        """
        Returns the evaluation `poptorch.DataLoader`.

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        poptorch_specific_kwargs = {
            "auto_distributed_partitioning": not isinstance(eval_dataset, torch.utils.data.IterableDataset),
            "mode": DataLoaderMode.Sync,
            "worker_init_fn": _WorkerInit(123),
        }

        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.eval_data_collator

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        if isinstance(eval_dataset, torch.utils.data.IterableDataset):
            return poptorch.DataLoader(
                self.eval_opts,
                eval_dataset,
                batch_size=self.args.per_device_eval_batch_size,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                **poptorch_specific_kwargs,
            )

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
        Returns the test `poptorch.DataLoader`.

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            test_dataset (`torch.utils.data.Dataset`, *optional*):
                The test dataset to use. If it is a [`~datasets.Dataset`], columns not accepted by the
                `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        poptorch_specific_kwargs = {
            "auto_distributed_partitioning": not isinstance(test_dataset, torch.utils.data.IterableDataset),
            "mode": DataLoaderMode.Sync,
            "worker_init_fn": _WorkerInit(123),
        }

        data_collator = self.eval_data_collator
        if is_datasets_available() and isinstance(test_dataset, datasets.Dataset):
            test_dataset = self._remove_unused_columns(test_dataset, description="test")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="test")

        if isinstance(test_dataset, torch.utils.data.IterableDataset):
            return poptorch.DataLoader(
                self.eval_opts,
                test_dataset,
                batch_size=self.args.per_device_eval_batch_size,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                **poptorch_specific_kwargs,
            )

        test_sampler = self._get_eval_sampler(test_dataset)

        # We use the same batch_size as for eval.
        return poptorch.DataLoader(
            self.eval_opts,
            test_dataset,
            sampler=test_sampler,
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
        Trainer's init through `optimizers`, or subclass and override this method (or `create_optimizer` and/or
        `create_scheduler`) in a subclass.
        """
        self.create_optimizer()
        self.create_scheduler(num_training_steps=num_training_steps, optimizer=self.optimizer)

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
            decay_parameters = {name for name in decay_parameters if "bias" not in name}
            if self.args.lamb or self.args.lamb_no_bias_correction:
                bias_parameters = {n for n, _ in self.model.named_parameters() if "bias" in n}
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        # Disable LAMB updates for bias parameters
                        "params": [p for n, p in self.model.named_parameters() if n in bias_parameters],
                        "weight_decay": 0.0,
                        "max_weight_norm": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in self.model.named_parameters()
                            if n not in decay_parameters and n not in bias_parameters
                        ],
                        "weight_decay": 0.0,
                    },
                ]
                optimizer_cls = LAMB
                optimizer_kwargs = {
                    "max_weight_norm": None,
                    "bias_correction": not self.args.lamb_no_bias_correction,
                    "eps": self.args.adam_epsilon,
                }
            else:
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
                optimizer_cls = AdamW
                optimizer_kwargs = {
                    # TODO: disabled max_grad_norm because it make things fail, fix it.
                    #  "max_grad_norm": self.args.max_grad_norm,
                    "betas": (self.args.adam_beta1, self.args.adam_beta2),
                    "eps": self.args.adam_epsilon,
                    "bias_correction": False,
                }

            first_order_type = torch.float32 if self.args.fp32 else torch.float16
            optimizer_kwargs["lr"] = self.args.learning_rate
            optimizer_kwargs["loss_scaling"] = self.args.loss_scaling
            optimizer_kwargs["accum_type"] = first_order_type
            optimizer_kwargs["first_order_momentum_accum_type"] = first_order_type
            optimizer_kwargs["second_order_momentum_accum_type"] = torch.float32

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if self.args.lamb or self.args.lamb_no_bias_correction:
                self.optimizer.variable_attrs.markAsConstant("max_weight_norm")

            self.optimizer.variable_attrs.markAsConstant("weight_decay")

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

    def num_examples(self, dataloader: poptorch.DataLoader) -> int:
        """
        Helper to get number of samples in a `poptorch.DataLoader` by accessing its dataset. When
        dataloader.dataset does not exist or has no length, estimates as best it can
        """
        return len(dataloader.dataset)

    def wrap_model(self, model: Union[PreTrainedModel, PoplarExecutor], training=True) -> PoplarExecutor:
        """
        Wraps a model for PopTorch, either for training or for inference.

        Args:
            model ([`transformers.PreTrainedModel`] or `poptorch.PoplarExecutor`):
                The model to wrap.
            training (`bool`, *optional*, defaults to `True`):
                Whether to wrap the model for training or not.

        Returns:
            `poptorch.PoplarExecutor`: The wrapped model.

        """
        wrapped = None
        if isinstance(model, PoplarExecutor):
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

    def _detach_training_model(self):
        """
        Detach training model from IPUs
        """
        if type(self.original_model) in TIED_WEIGHT_MODELS:
            # Work-around bug with models with tied-weights
            # TODO: Remove this when bug fixed
            self.optimizer_state = self.optimizer.state_dict()
            self.training_model.destroy()
            self.model = poptorch._impl.unwrapIfWrapped(self.model)
            for obj in self.model.buffers():
                if "PoptorchBuffer" in str(obj.__class__):
                    obj.__class__ = obj.__class__.__bases__[0]
            self.training_model = None
        else:
            self.training_model.detachFromDevice()

    def _detach_inference_model(self):
        """
        Detach inference model from IPUs
        """
        if type(self.original_model) in TIED_WEIGHT_MODELS:
            # Work-around bug with models with tied-weights
            # TODO: Remove this when bug fixed
            self.inference_model.destroy()
            self.inference_model = None
            self.model = poptorch._impl.unwrapIfWrapped(self.model)
            for obj in self.model.buffers():
                if "PoptorchBuffer" in str(obj.__class__):
                    obj.__class__ = obj.__class__.__bases__[0]
        else:
            self.inference_model.detachFromDevice()

    def _reattach_training_model(self):
        """
        Reattach training model from IPUs
        """
        if type(self.original_model) in TIED_WEIGHT_MODELS:
            # Work-around bug with models with tied-weights
            # TODO: Remove this when bug fixed
            self.training_model = self.wrap_model(self.model.train().half())
        else:
            self.training_model.attachToDevice()

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
            resume_from_checkpoint (`str` or `bool`, *optional*):
                If a `str`, local path to a saved checkpoint as saved by a previous instance of [`IPUTrainer`]. If a
                `bool` and equals `True`, load the last checkpoint in *args.output_dir* as saved by a previous instance
                of [`Trainer`]. If present, training will resume from the model/optimizer/scheduler states loaded here.
            trial (`optuna.Trial` or `Dict[str, Any]`, *optional*):
                The trial run or the hyperparameter dictionary for hyperparameter search.
                **Note**: Feature not supported for now.
            ignore_keys_for_eval (`List[str]`, *optional*)
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions for evaluation during the training.
            kwargs:
                Additional keyword arguments used to hide deprecated arguments
        """
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

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

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")

        if resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint)

        return self._inner_training_loop(
            args=args, resume_from_checkpoint=resume_from_checkpoint, ignore_keys_for_eval=ignore_keys_for_eval
        )

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        # For now, it will always be None.
        if batch_size is None:
            batch_size = args.per_device_train_batch_size

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = batch_size * self.ipu_config.batch_size_factor()

        len_dataloader = None
        if has_length(train_dataloader):
            # No need to divide by the number of gradient accumulation steps as poptorch already accounts for that.
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
            # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
            # the best we can do.
            num_train_samples = max_steps * total_train_batch_size
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = IPUTrainerState()
        if trial is not None:
            raise ValueError("Hyperparameter tuning is not supported by the IPUTrainer.")
            trial = None
        self.state.is_hyper_param_search = trial is not None

        self.training_model = self.wrap_model(self.model)

        # TODO: handle optimizer and scheduler creation
        # if delay_optimizer_creation:
        #     self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        self.compile_model(self.training_model, next(iter(train_dataloader)), log=True)

        # Train!
        num_examples = (
            self.num_examples(train_dataloader)
            if has_length(train_dataloader)
            else total_train_batch_size * args.max_steps
        )
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
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
            self.state = IPUTrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            if self.state.start_time < 0:
                self.state.start_time = start_time
            start_time = self.state.start_time
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
                if args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = None
        self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.start_time = start_time

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
                    train_dataloader.sampler, RandomSampler
                )
                if is_torch_less_than_1_11 or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    # That was before PyTorch 1.11 however...
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    _ = list(train_dataloader.sampler)

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
                len(epoch_iterator)
                if has_length(train_dataloader)
                else args.max_steps * args.gradient_accumulation_steps
            )

            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            step = -1
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

                self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                tr_loss_step = self.training_step(self.training_model, inputs)

                if args.logging_nan_inf_filter and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step)):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                # TODO: see how to enable this (if necessary), slows down training a lot.
                self.current_flos += float(self.floating_point_ops(inputs))

                # Optimizer step
                optimizer_was_run = True

                if optimizer_was_run:
                    self.lr_scheduler.step()
                    self.training_model.setOptimizer(self.optimizer)

                self.state.global_step += 1
                self.state.epoch = epoch + (step + 1) / steps_in_epoch
                self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                self._maybe_log_save_evaluate(tr_loss, self.training_model, epoch, ignore_keys_for_eval)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, self.training_model, epoch, ignore_keys_for_eval)

            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            self._load_best_model()

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
        self._detach_training_model()

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):

        if model is None:
            model = self.model

        if not os.path.isfile(os.path.join(resume_from_checkpoint, WEIGHTS_NAME)) and not os.path.isfile(
            os.path.join(resume_from_checkpoint, WEIGHTS_INDEX_NAME)
        ):
            raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

        logger.info(f"Loading model from {resume_from_checkpoint}.")

        if os.path.isfile(os.path.join(resume_from_checkpoint, CONFIG_NAME)):
            config = PretrainedConfig.from_json_file(os.path.join(resume_from_checkpoint, CONFIG_NAME))
            checkpoint_version = config.transformers_version
            if checkpoint_version is not None and checkpoint_version != __version__:
                logger.warning(
                    f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
                    f"Transformers but your current version is {__version__}. This is not recommended and could "
                    "yield to errors or unwanted behaviors."
                )

        if os.path.isfile(os.path.join(resume_from_checkpoint, WEIGHTS_NAME)):
            # We load the model state dict on the CPU to avoid an OOM error.
            state_dict = torch.load(os.path.join(resume_from_checkpoint, WEIGHTS_NAME), map_location="cpu")
            # workaround for FSDP bug https://github.com/pytorch/pytorch/issues/82963
            # which takes *args instead of **kwargs
            load_result = model.load_state_dict(state_dict, False)
            # release memory
            del state_dict
            self._issue_warnings_after_load(load_result)

    def _load_best_model(self):
        logger.info(f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).")
        best_model_path = os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME)
        if os.path.exists(best_model_path):
            # We load the model state dict on the CPU to avoid an OOM error.
            state_dict = torch.load(best_model_path, map_location="cpu")
            self._load_state_dict_in_model(state_dict)
        else:
            logger.warning(
                f"Could not locate the best model at {best_model_path}, if you are running a distributed training "
                "on multiple nodes, you should activate `--save_on_each_node`."
            )

    def _load_state_dict_in_model(self, state_dict):
        self.model.deparallelize()
        load_result = self.model.load_state_dict(state_dict, strict=False)
        self.model.parallelize()
        if not self.args.fp32:
            self.model.half()

        if len(load_result.missing_keys) != 0:
            if self.model._keys_to_ignore_on_save is not None and set(load_result.missing_keys) != set(
                self.model._keys_to_ignore_on_save
            ):
                logger.warn(f"There were missing keys in the checkpoint model loaded: {load_result.missing_keys}.")
        if len(load_result.unexpected_keys) != 0:
            logger.warn(f"There were unexpected keys in the checkpoint model loaded: {load_result.unexpected_keys}.")

    def _issue_warnings_after_load(self, load_result):

        if len(load_result.missing_keys) != 0:
            if self.model._keys_to_ignore_on_save is not None and set(load_result.missing_keys) == set(
                self.model._keys_to_ignore_on_save
            ):
                self.model.tie_weights()
            else:
                logger.warning(f"There were missing keys in the checkpoint model loaded: {load_result.missing_keys}.")
        if len(load_result.unexpected_keys) != 0:
            logger.warning(
                f"There were unexpected keys in the checkpoint model loaded: {load_result.unexpected_keys}."
            )

    def _maybe_log_save_evaluate(self, tr_loss, model, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            logs: Dict[str, float] = {}

            tr_loss_scalar = tr_loss.mean().item()

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
            self._detach_training_model()
            if isinstance(self.eval_dataset, dict):
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._reattach_training_model()

        if self.control.should_save:
            self._save_checkpoint(model, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def _load_rng_state(self, checkpoint):
        # Load RNG states from `checkpoint`
        if checkpoint is None:
            return

        # TODO: validate that.
        local_rank = -1
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
        # TODO: enable this when SDK 2.5 is out.
        # self.training_model.rng_state = checkpoint_rng_state["ipu"]

    def _save_checkpoint(self, model, metrics=None):
        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        run_dir = self.args.output_dir
        self.store_flos()

        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir, _internal_call=True)
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
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
            # TODO: enable this when SDK 2.5 is out.
            # "ipu": self.training_model.rng_state,
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
            self.optimizer.load_state_dict(torch.load(os.path.join(checkpoint, OPTIMIZER_NAME)))
            with warnings.catch_warnings(record=True) as caught_warnings:
                self.lr_scheduler.load_state_dict(torch.load(os.path.join(checkpoint, SCHEDULER_NAME)))
            reissue_pt_warnings(caught_warnings)

            self.training_model.setOptimizer(self.optimizer)

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def _prepare_input(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, dict):
            return type(data)(**{k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            return data
        return data

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError(
                "The batch received was empty, your model won't be able to train on it. Double-check that your "
                f"training dataset contains keys expected by the model: {','.join(self._signature_columns)}."
            )
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
            model (`poptorch.PoplarExecutor`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        inputs = self._prepare_inputs(inputs)
        loss = self.compute_loss(model, inputs)
        loss = loss.mean()
        return loss

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by `IPUTrainer`. By default, all models return the loss in the first element.

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
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def is_world_process_zero(self) -> bool:
        # Needed only because log_metrics use it.
        return True

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Will save the model, so you can reload it using `from_pretrained()`.

        Will only save from the main process.
        """
        if output_dir is None:
            output_dir = self.args.output_dir

        if self.args.should_save:
            self._save(output_dir)

        # Push to the Hub when `save_model` is called by the user.
        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(commit_message="Model save")

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        # Updating self.model weights with the weights stored on device.
        # TODO: can this be deleted? I would makle things faster.
        if self.training_model is not None and self.training_model.isAttachedToDevice():
            self.training_model.copyWeightsToHost()

        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            logger.info("Trainer.model is not a `transformers.PreTrainedModel`, only saving its state dict.")
            if state_dict is None:
                state_dict = self.model.state_dict()
            torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            rng_state = torch.random.get_rng_state()
            self.model.deparallelize()
            self.model.save_pretrained(output_dir, state_dict=state_dict)
            self.model.parallelize()
            torch.random.set_rng_state(rng_state)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        self.ipu_config.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def store_flos(self):
        # Storing the number of floating-point operations that went into the model
        # TODO: Validate that this is right. (It's most likely wrong)
        self.state.total_flos += self.current_flos * self.ipu_config.batch_size_factor()
        self.current_flos = 0

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
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        prediction_loss_only = True if self.compute_metrics is None else None
        if prediction_loss_only is None:
            prediction_loss_only = self.args.prediction_loss_only

        # Running this here (even though it is being recalled in self.evaluation_loop to make compilation happen here.
        # That way, compilation will not mess inference speed metrics.
        _ = self._wrap_and_compile_model_for_evaluation(eval_dataloader, prediction_loss_only)

        start_time = time.time()

        output = self.evaluation_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        # If we are using padded data collator, dropped the padded part of the output
        if self.args.pad_on_batch_axis:
            eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
            dataset_len = len(eval_dataset)
            output = output._replace(predictions=tuple(pred[:dataset_len] for pred in output.predictions))
            output = output._replace(num_samples=dataset_len)

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
        will also return metrics, like in `evaluate()`.

        Args:
            test_dataset (`Dataset`):
                Dataset to run the predictions on. If it is an `datasets.Dataset`, columns not accepted by the
                `model.forward()` method are automatically removed. Has to implement the method `__len__`
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"test"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "test_bleu" if the prefix is "test" (default)

        <Tip>

        If your predictions or labels have different sequence length (for instance because you're doing dynamic padding
        in a token classification task) the predictions will be padded (on the right) to allow for concatenation into
        one array. The padding index is -100.

        </Tip>

        Returns: *NamedTuple* A namedtuple with the following keys:

            - predictions (`np.ndarray`): The predictions on `test_dataset`.
            - label_ids (`np.ndarray`, *optional*): The labels (if the dataset contained some).
            - metrics (`Dict[str, float]`, *optional*): The potential dictionary of metrics (if the dataset contained
              labels).
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()

        output = self.evaluation_loop(
            test_dataloader, description="Prediction", ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
        )

        # If we are using padded data collator, dropped the padded part of the output
        if self.args.pad_on_batch_axis:
            dataset_len = len(test_dataset)
            output = output._replace(predictions=tuple([pred[:dataset_len] for pred in output.predictions]))
            output = output._replace(num_samples=dataset_len)

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

    def _wrap_and_compile_model_for_evaluation(self, dataloader, prediction_loss_only):
        model = self.wrap_model(self.model, training=False)
        self.compile_model(model, next(iter(dataloader)), log=True)
        return model

    def evaluation_loop(
        self,
        dataloader: poptorch.DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `IPUTrainer.evaluate()` and `IPUTrainer.predict()`.

        Works both with or without labels.
        """
        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        self.inference_model = self._wrap_and_compile_model_for_evaluation(dataloader, prediction_loss_only)

        batch_size = dataloader.batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        self.callback_handler.eval_dataloader = dataloader
        # eval_dataset = getattr(dataloader, "dataset", None)

        if self.args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on IPU (accumulated for eval_accumulation_steps, legacy code for IPUs)
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
            # If dataset is not sized, is_last_batch is False because we cannot know.
            is_last_batch = (
                step == len(dataloader) - 1 if isinstance(dataloader.dataset, collections.abc.Sized) else False
            )
            loss, logits, labels = self.prediction_step(
                self.inference_model,
                inputs,
                prediction_loss_only,
                ignore_keys=ignore_keys,
                is_last_batch=is_last_batch,
            )

            # Update containers on host
            if loss is not None:
                loss = loss.mean(dim=0, keepdim=True)
                # If only one IPU is used, loss is a zero dimensional tensor, we unsqueeze to be able to concatenate.
                if loss.dim() == 0:
                    loss = loss.unsqueeze(0)
                losses_host = loss if losses_host is None else torch.cat((losses_host, loss), dim=0)
            if logits is not None:
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            if labels is not None:
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)

            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

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
        # In the original Trainer, TODO: should we use this instead?
        # if has_length(eval_dataset):
        #     num_samples = len(eval_dataset)
        # # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # # methods. Therefore we need to make sure it also has the attribute.
        # elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
        #     num_samples = eval_dataset.num_examples
        # else:
        #     if has_length(dataloader):
        #         num_samples = self.num_examples(dataloader)
        #     else:  # both len(dataloader.dataset) and len(dataloader) fail
        #         num_samples = observed_num_examples
        # if num_samples == 0 and observed_num_examples > 0:
        #     num_samples = observed_num_examples
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
        self._detach_inference_model()

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def prediction_step(
        self,
        model: poptorch.PoplarExecutor,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        is_last_batch: bool = False,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`poptorch.PoplarExecutor`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`Lst[str]`, *optional*):
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
                loss = loss.detach()
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
        For models that inherit from [`transformers.PreTrainedModel`], uses that method to compute the number of
        floating point operations for every backward + forward pass. If using another model, either implement such a
        method in the model or subclass and override this method.

        Args:
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

        Returns:
            `int`: The number of floating-point operations.
        """
        # Using self.original_model because self.model is the underlying model used by the IPUs
        # and calling floating_point_ops on it slows things down a lot.
        if hasattr(self.original_model, "floating_point_ops"):
            return self.original_model.floating_point_ops(inputs)
        else:
            return 0

    def init_git_repo(self, at_init: bool = False):
        """
        Initializes a git repo in `self.args.hub_model_id`.

        Args:
            at_init (`bool`, *optional*, defaults to `False`):
                Whether this function is called before any training or not. If `self.args.overwrite_output_dir` is
                `True` and `at_init` is `True`, the path to the repo (which is `self.args.output_dir`) might be wiped
                out.
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
                private=self.args.hub_private_repo,
            )
        except EnvironmentError:
            if self.args.overwrite_output_dir and at_init:
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
        tags: Union[str, List[str], None] = None,
        model_name: Optional[str] = None,
        finetuned_from: Optional[str] = None,
        tasks: Union[str, List[str], None] = None,
        dataset_tags: Union[str, List[str], None] = None,
        dataset: Union[str, List[str], None] = None,
        dataset_args: Union[str, List[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            language (`str`, *optional*):
                The language of the model (if applicable)
            license (`str`, *optional*):
                The license of the model. Will default to the license of the pretrained model used, if the original
                model given to the `Trainer` comes from a repo on the Hub.
            tags (`str` or `List[str]`, *optional*):
                Some tags to be included in the metadata of the model card.
            model_name (`str`, *optional*):
                The name of the model.
            finetuned_from (`str`, *optional*):
                The name of the model used to fine-tune this one (if applicable). Will default to the name of the repo
                of the original model given to the `Trainer` (if it comes from the Hub).
            tasks (`str` or `List[str]`, *optional*):
                One or several task identifiers, to be included in the metadata of the model card.
            dataset_tags (`str` or `List[str]`, *optional*):
                One or several dataset tags, to be included in the metadata of the model card.
            dataset (`str` or `List[str]`, *optional*):
                One or several dataset identifiers, to be included in the metadata of the model card.
            dataset_args (`str` or `List[str]`, *optional*):
               One or several dataset arguments, to be included in the metadata of the model card.
        """
        if not self.is_world_process_zero():
            return

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
        if self.args.hub_strategy == HubStrategy.END:
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
        Upload *self.model* and *self.tokenizer* to the 🤗 model hub on the repo *self.args.hub_model_id*.

        Parameters:
            commit_message (`str`, *optional*, defaults to `"End of training"`):
                Message to commit while pushing.
            blocking (`bool`, *optional*, defaults to `True`):
                Whether the function should return only when the `git push` has finished.
            kwargs:
                Additional keyword arguments passed along to [`~Trainer.create_model_card`].

        Returns:
            The url of the commit of your model in the given repository if `blocking=False`, a tuple with the url of
            the commit and an object to track the progress of the commit if `blocking=True`
        """
        # If a user calls manually `push_to_hub` with `self.args.push_to_hub = False`, we try to create the repo but
        # it might fail.
        if not hasattr(self, "repo"):
            self.init_git_repo()

        if self.args.should_save:
            if self.args.hub_model_id is None:
                model_name = Path(self.args.output_dir).name
            else:
                model_name = self.args.hub_model_id.split("/")[-1]

        # Needs to be executed on all processes for TPU training, but will only save on the processed determined by
        # self.args.should_save.
        self.save_model(_internal_call=True)

        # Cancel any async push in progress if blocking=True. The commits will all be pushed together.
        if blocking and self.push_in_progress is not None and not self.push_in_progress.is_done:
            self.push_in_progress._process.kill()
            self.push_in_progress = None

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
