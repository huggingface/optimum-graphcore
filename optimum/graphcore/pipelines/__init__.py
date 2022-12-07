# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
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

from typing import Any, Dict, List, Optional, Union

import torch

import poptorch
import transformers.pipelines
from optimum.graphcore import IPUConfig
from optimum.graphcore.modeling_utils import to_pipelined
from transformers import (
    AudioClassificationPipeline,
    AutomaticSpeechRecognitionPipeline,
    AutoModelForAudioClassification,
    AutoModelForCTC,
    AutoModelForImageClassification,
    AutoModelForMaskedLM,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    ImageClassificationPipeline,
    Pipeline,
    PreTrainedTokenizer,
    ProcessorMixin,
    QuestionAnsweringPipeline,
    TextClassificationPipeline,
)
from transformers.feature_extraction_utils import PreTrainedFeatureExtractor
from transformers.modeling_utils import PreTrainedModel
from transformers.onnx.utils import get_preprocessor
from transformers.pipelines import get_task
from transformers.utils import HUGGINGFACE_CO_RESOLVE_ENDPOINT, logging

from .fill_mask import IPUFillMaskPipeline
from .token_classification import IPUTokenClassificationPipeline
from .zero_shot_classification import IPUZeroShotClassificationPipeline


class IncompatibleIPUConfigError(Exception):
    """An exception used when an IPU Config is incompatible with a model"""

    pass


logger = logging.get_logger(__name__)

TASK_ALIASES = {
    "sentiment-analysis": "text-classification",
    "ner": "token-classification",
}
SUPPORTED_TASKS = {
    "audio-classification": {
        "impl": AudioClassificationPipeline,
        "class": (AutoModelForAudioClassification,),
        "default": {
            "model": ("superb/hubert-base-superb-ks", "d7e0efe"),
            "ipu_config": "Graphcore/hubert-base-ipu",
        },
        "type": "audio",
    },
    "automatic-speech-recognition": {
        "impl": AutomaticSpeechRecognitionPipeline,
        # TODO: support AutoModelForSpeechSeq2Seq
        "class": (AutoModelForCTC,),
        "default": {
            "model": ("facebook/wav2vec2-base-960h", "55bb623"),
            "ipu_config": "Graphcore/wav2vec2-ctc-base-ipu",
        },
        "type": "multimodal",
    },
    "fill-mask": {
        "impl": IPUFillMaskPipeline,
        "class": (AutoModelForMaskedLM,),
        "default": {
            "model": ("distilroberta-base", "ec58a5b"),
            "ipu_config": "Graphcore/roberta-base-ipu",
            "max_length": 128,
        },
        "type": "text",
    },
    "image-classification": {
        "impl": ImageClassificationPipeline,
        "class": (AutoModelForImageClassification,),
        "default": {
            "model": ("google/vit-base-patch16-224", "5dca96d"),
            "ipu_config": "Graphcore/vit-base-ipu",
        },
        "type": "image",
    },
    "question-answering": {
        "impl": QuestionAnsweringPipeline,
        "class": (AutoModelForQuestionAnswering,),
        "default": {
            "model": ("distilbert-base-cased-distilled-squad", "626af31"),
            "ipu_config": "Graphcore/distilbert-base-ipu",
        },
        "type": "text",
    },
    "text-classification": {
        "impl": TextClassificationPipeline,
        "class": (AutoModelForSequenceClassification,),
        "default": {
            "model": ("distilbert-base-uncased-finetuned-sst-2-english", "af0f99b"),
            "ipu_config": "Graphcore/distilbert-base-ipu",
            "max_length": 128,
        },
        "type": "text",
    },
    "token-classification": {
        "impl": IPUTokenClassificationPipeline,
        "class": (AutoModelForTokenClassification,),
        "default": {
            "model": ("dbmdz/bert-large-cased-finetuned-conll03-english", "f2482bf"),
            "ipu_config": "Graphcore/bert-large-ipu",
            "max_length": 128,
        },
        "type": "text",
    },
    "zero-shot-classification": {
        "impl": IPUZeroShotClassificationPipeline,
        "class": (AutoModelForSequenceClassification,),
        "default": {
            "model": ("roberta-large-mnli", "130fb28"),
            "ipu_config": "Graphcore/roberta-large-ipu",
            "max_length": 128,
        },
        "type": "text",
    },
}

NO_FEATURE_EXTRACTOR_TASKS = set()
NO_TOKENIZER_TASKS = set()
for task, values in SUPPORTED_TASKS.items():
    if values["type"] == "text":
        NO_FEATURE_EXTRACTOR_TASKS.add(task)
    elif values["type"] in {"audio", "image"}:
        NO_TOKENIZER_TASKS.add(task)
    elif values["type"] != "multimodal":
        raise ValueError(f"SUPPORTED_TASK {task} contains invalid type {values['type']}")


def list_tasks() -> List[str]:
    """Lists the supported tasks and their aliases"""
    return sorted([*{*SUPPORTED_TASKS, *TASK_ALIASES}])


def get_poplar_executor(
    model: PreTrainedModel,
    ipu_config: Union[str, dict] = None,
    fp16: bool = True,
) -> PreTrainedModel:
    ipu_config_arg = ipu_config

    if isinstance(ipu_config, str):
        ipu_config = IPUConfig.from_pretrained(ipu_config)
    elif isinstance(ipu_config, dict):
        ipu_config = IPUConfig.from_dict(ipu_config)
    else:
        raise ValueError("ipu_config must be a string or a dictionary.")
    ipu_config.inference_device_iterations = 1
    # TODO: inference_replication_factor should be adaptive, especially for batching.
    ipu_config.inference_replication_factor = 1
    try:
        model = to_pipelined(model, ipu_config, force=False)
        model.parallelize()
    except Exception as error:
        new_message = (
            "The model and ipu_config seem to be incompatible,"
            " please try a different IPU config or customizing it for the model."
            f" The config provided is '{ipu_config_arg}'"
        )
        raise IncompatibleIPUConfigError(new_message) from error
    if fp16:
        model.half()
    opts = ipu_config.to_options(for_inference=True)
    opts.setExecutionStrategy(poptorch.ShardedExecution())
    model = poptorch.inferenceModel(model.eval(), opts)
    return model


def get_inference_context(self):
    return torch.no_grad


def check_model_type(self, supported_models: Union[List[str], dict]):
    """
    Check if the model class is in supported by the pipeline.

    Args:
        supported_models (`List[str]` or `dict`):
            The list of models supported by the pipeline, or a dictionary with model class values.
    """
    if not isinstance(supported_models, list):  # Create from a model mapping
        supported_models_names = []
        for config, model in supported_models.items():
            # Mapping can now contain tuples of models for the same configuration.
            if isinstance(model, tuple):
                supported_models_names.extend([_model.__name__ for _model in model])
            else:
                supported_models_names.append(model.__name__)
        supported_models = supported_models_names
    if self.model._user_model.__class__.__bases__[0].__name__ not in supported_models:
        logger.error(
            f"The model '{self.model._user_model.__class__.__bases__[0].__name__}' is not supported for {self.task}. Supported models are"
            f" {supported_models}."
        )


def pipeline(
    task: str = None,
    model: Optional[Any] = None,
    ipu_config: Union[str, dict] = None,
    tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
    feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None,
    revision: Optional[str] = None,
    use_auth_token: Optional[Union[str, bool]] = None,
    pipeline_class: Optional[Any] = None,
    fp16: bool = True,
    **kwargs,
) -> Pipeline:
    """Utility factory method to build a [ Pipeline ] for IPU models.

    Arguments:
        task : The task, see docs for ``transformers.pipeline`` for supported options.
        model : A pre-trained model, see docs for ``transformers.pipeline`` for supported options.
        ipu_config : An IPU config, can either be the path to a model from the HuggingFace Hub
            which defines a ``ipu_config.json`` or a dictionary with the same options.
        tokenizer : The tokenizer, see docs for ``transformers.pipeline`` for supported options.
        feature_extractor : The feature extractor, see docs for ``transformers.pipeline`` for supported options.
        revision : Revision of the model.
        use_auth_token : An authorization token to use for these calls to the hub.
        pipeline_class : Override the Pipeline class defined by the task.
        fp16 : Whether to use Float 16 or not.

        **kwargs: Additional keyword arguments that are passed to the ``transformers.pipeline`` function

    Returns:
        The pipeline object for the specified task.
    """

    if task is None and model is None:
        raise RuntimeError(
            "Impossible to instantiate a pipeline without either a task or a model "
            "being specified. "
            "Please provide a task class or a model"
        )
    if task is None and model is not None:
        if not isinstance(model, str):
            raise RuntimeError(
                "Inferring the task automatically requires to check the hub with a model_id defined as a `str`."
                f"{model} is not a valid model_id."
            )
        task = get_task(model, use_auth_token)

    if task in TASK_ALIASES:
        task = TASK_ALIASES[task]

    targeted_task = "translation" if task.startswith("translation") else task

    if targeted_task not in SUPPORTED_TASKS:
        raise ValueError(f"Task {targeted_task} is not supported. Supported tasks are {list(SUPPORTED_TASKS.keys())}")

    # These will never require a tokenizer.
    # the model on the other hand might have a tokenizer, but
    # the files could be missing from the hub, instead of failing
    # on such repos, we just force to not load it.
    load_tokenizer = targeted_task not in NO_TOKENIZER_TASKS
    load_feature_extractor = targeted_task not in NO_FEATURE_EXTRACTOR_TASKS

    if pipeline_class is None:
        pipeline_class = SUPPORTED_TASKS[targeted_task]["impl"]

    if ipu_config is None and not isinstance(model, poptorch._poplar_executor.PoplarExecutor):
        ipu_config = SUPPORTED_TASKS[targeted_task]["default"]["ipu_config"]

    if model is None:
        model_id, revision = SUPPORTED_TASKS[targeted_task]["default"]["model"]
        logger.warning(
            f"No model was supplied, defaulted to {model_id} and revision"
            f" {revision} ({HUGGINGFACE_CO_RESOLVE_ENDPOINT}/{model_id}).\n"
            "Using a pipeline without specifying a model name and revision in production is not recommended."
        )
        model = SUPPORTED_TASKS[targeted_task]["class"][0].from_pretrained(model_id, revision=revision)
        model = get_poplar_executor(model, ipu_config, fp16)
    elif isinstance(model, str):
        model_id = model
        model = SUPPORTED_TASKS[targeted_task]["class"][0].from_pretrained(model_id, revision=revision)
        model = get_poplar_executor(model, ipu_config, fp16)
    elif isinstance(model, PreTrainedModel):
        model = get_poplar_executor(model, ipu_config, fp16)
        if tokenizer is None and load_tokenizer:
            raise ValueError("If you pass a model as a PreTrainedModel, you must pass a tokenizer as well")
        if feature_extractor is None and load_feature_extractor:
            raise ValueError("If you pass a model as a PreTrainedModel, you must pass a feature extractor as well")
    elif isinstance(model, poptorch._poplar_executor.PoplarExecutor):
        if tokenizer is None and load_tokenizer:
            raise ValueError(
                "If you pass a model as a poptorch._poplar_executor.PoplarExecutor, you must pass a tokenizer as well"
            )
        if feature_extractor is None and load_feature_extractor:
            raise ValueError(
                "If you pass a model as a poptorch._poplar_executor.PoplarExecutor, you must pass a feature extractor as well"
            )
    else:
        raise ValueError(
            f"""Model {model} is not supported. Please provide a valid model either as string, PreTrainedModel or
            poptorch._poplar_executor.PoplarExecutor. You can also provide non model then a default one will be used"""
        )

    if (tokenizer is None and load_tokenizer) or (feature_extractor is None and load_feature_extractor):
        preprocessor = get_preprocessor(model_id)
        if tokenizer is None and load_tokenizer:
            if isinstance(preprocessor, ProcessorMixin):
                tokenizer = preprocessor.tokenizer
            else:
                tokenizer = preprocessor
        if feature_extractor is None and load_feature_extractor:
            if isinstance(preprocessor, ProcessorMixin):
                feature_extractor = preprocessor.feature_extractor
            else:
                feature_extractor = preprocessor

    # Override Pipeline methods
    Pipeline.get_inference_context = get_inference_context
    Pipeline.check_model_type = check_model_type

    # Override pipelines' _forward
    old_forward = pipeline_class._forward

    def new_forward(self, model_inputs, *args, **kwargs):
        # Support change in batch size
        if self.model._executable_inputs:
            compiled_bs = self.model._executable_inputs.args[0].shape[0]
            for input in model_inputs.values():
                if isinstance(input, torch.Tensor):
                    input_bs = input.shape[0]
                    break
            if compiled_bs != input_bs:
                self.model.destroy()
        if fp16:
            # Support fp16
            for key, input in model_inputs.items():
                if isinstance(input, torch.Tensor) and input.dtype == torch.float32:
                    model_inputs[key] = input.half()
        return old_forward(self, model_inputs, *args, **kwargs)

    pipeline_class._forward = new_forward

    # Auto padding for some tasks
    if "max_length" in SUPPORTED_TASKS[targeted_task]["default"]:
        kwargs["padding"] = kwargs.get("padding", "max_length")
        default_max_length = SUPPORTED_TASKS[targeted_task]["default"]["max_length"]
        if kwargs.get("max_length") is None:
            logger.warning(
                f"No padding arguments specified, so pad to {default_max_length} by default. "
                f"Inputs longer than {default_max_length} will be truncated."
                " To change this behaviour, pass the `padding='max_length'` and"
                "`max_length=<your desired input length>` arguments to the pipeline function"
            )
        # question-answering already has its own default padding length `max_seq_len` defined, so we just enable padding to max length.
        if targeted_task in {"question-answering"}:
            kwargs["padding"] = "max_length"
            logger.warning(
                "No padding arguments specified, so pad to 384 by default. Inputs longer than 384 will be truncated."
            )
        kwargs["max_length"] = kwargs.get("max_length", default_max_length)

    # question-answering already has its own default padding length `max_seq_len` defined, so we just enable padding to max length.
    if targeted_task in {"question-answering"}:
        kwargs["padding"] = kwargs.get("padding", "max_length")
        logger.warning(
            "No padding arguments specified, so pad to 384 by default. Inputs longer than 384 will be truncated."
        )

    # Set pad_token for models that do not have pad_token
    if model.config.model_type in {"gpt2"}:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    return transformers.pipelines.pipeline(
        task,
        model=model,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        use_auth_token=use_auth_token,
        pipeline_class=pipeline_class,
        **kwargs,
    )
