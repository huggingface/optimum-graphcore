import poptorch

from optimum.graphcore import IPUConfig
from optimum.graphcore.modeling_utils import to_pipelined

from typing import Any, Optional, Union

from transformers import (
    Pipeline,
    PreTrainedTokenizer,
    ProcessorMixin,
    AutoModelForAudioClassification,
    AutoModelForCTC,
    AutoModelForImageClassification,
    AutoModelForMaskedLM,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AudioClassificationPipeline,
    AutomaticSpeechRecognitionPipeline,
    FillMaskPipeline,
    ImageClassificationPipeline,
    QuestionAnsweringPipeline,
    TextClassificationPipeline,
    TokenClassificationPipeline,
    ZeroShotClassificationPipeline,
)
from transformers import pipeline as transformers_pipeline
from transformers.feature_extraction_utils import PreTrainedFeatureExtractor
from transformers.modeling_utils import PreTrainedModel
from transformers.onnx.utils import get_preprocessor


TASK_ALIASES = {
    "sentiment-analysis": "text-classification",
    "ner": "token-classification",
}
SUPPORTED_TASKS = {
    "audio-classification": {
        "impl": AudioClassificationPipeline,
        "class": (AutoModelForAudioClassification,),
        "default": {
            "model": "superb/hubert-base-superb-ks",
            "ipu_config": "Graphcore/hubert-base-ipu",
        },
        "type": "audio",
    },
    "automatic-speech-recognition": {
        "impl": AutomaticSpeechRecognitionPipeline,
        # "class": (AutoModelForCTC, AutoModelForSpeechSeq2Seq),
        "class": (AutoModelForCTC,),
        "default": {
            "model": "facebook/wav2vec2-base-960h",
            "ipu_config": "Graphcore/wav2vec2-ctc-base-ipu",
        },
        "type": "multimodal",
    },
    "fill-mask": {
        "impl": FillMaskPipeline,
        "class": (AutoModelForMaskedLM,),
        "default": {
            "model": "roberta-base",
            "ipu_config": "Graphcore/roberta-base-ipu",
        },
        "type": "text",
    },
    "image-classification": {
        "impl": ImageClassificationPipeline,
        "class": (AutoModelForImageClassification,),
        "default": {
            "model": "google/vit-base-patch16-224",
            "ipu_config": "Graphcore/vit-base-ipu",
        },
        "type": "image",
    },
    "question-answering": {
        "impl": QuestionAnsweringPipeline,
        "class": (AutoModelForQuestionAnswering,),
        "default": {
            "model": "deepset/roberta-base-squad2",
            "ipu_config": "Graphcore/roberta-base-ipu",
        },
        "type": "text",
    },
    "text-classification": {
        "impl": TextClassificationPipeline,
        "class": (AutoModelForSequenceClassification,),
        "default": {
            "model": "cardiffnlp/twitter-roberta-base-sentiment",
            "ipu_config": "Graphcore/roberta-base-ipu",
            "padding_length": 128,
        },
        "type": "text",
    },
    "token-classification": {
        "impl": TokenClassificationPipeline,
        "class": (AutoModelForTokenClassification,),
        "default": {
            "model": "dbmdz/bert-large-cased-finetuned-conll03-english",
            "ipu_config": "Graphcore/bert-large-ipu",
        },
        "type": "text",
    },
    "zero-shot-classification": {
        "impl": ZeroShotClassificationPipeline,
        "class": (AutoModelForSequenceClassification,),
        "default": {
            "model": "roberta-large-mnli",
            "ipu_config": "Graphcore/roberta-large-ipu",
            "padding_length": 128,
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

def get_poplar_executor(model: PreTrainedModel, ipu_config: Union[str, dict] = None):
    if isinstance(ipu_config, str):
        ipu_config = IPUConfig.from_pretrained(ipu_config)
    elif isinstance(ipu_config, str):
        ipu_config = IPUConfig.from_dict(ipu_config)
    else:
        raise ValueError("ipu_config must be a string or a dictionary.")
    ipu_config.inference_device_iterations = 1
    model = to_pipelined(model, ipu_config, force=False)
    model.parallelize()
    model.half()
    opts = ipu_config.to_options(for_inference=True)
    opts.setExecutionStrategy(poptorch.ShardedExecution())
    model = poptorch.inferenceModel(model.eval(), opts)
    return model

import torch
def get_inference_context(self):
    return torch.no_grad

from typing import List
from transformers.utils import logging
logger = logging.get_logger(__name__)
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
    use_fast: bool = True,
    use_auth_token: Optional[Union[str, bool]] = None,
    **kwargs,
) -> Pipeline:

    if task in TASK_ALIASES:
        task = TASK_ALIASES[task]

    targeted_task = "translation" if task.startswith("translation") else task

    if targeted_task not in list(SUPPORTED_TASKS.keys()):
        raise ValueError(f"Task {targeted_task} is not supported. Supported tasks are { list(SUPPORTED_TASKS.keys())}")

    # copied from transformers.pipelines.__init__.py l.609
    if targeted_task in NO_TOKENIZER_TASKS:
        # These will never require a tokenizer.
        # the model on the other hand might have a tokenizer, but
        # the files could be missing from the hub, instead of failing
        # on such repos, we just force to not load it.
        load_tokenizer = False
    else:
        load_tokenizer = True
    if targeted_task in NO_FEATURE_EXTRACTOR_TASKS:
        load_feature_extractor = False
    else:
        load_feature_extractor = True

    if model is None:
        model_id = SUPPORTED_TASKS[targeted_task]["default"]["model"]
        model = SUPPORTED_TASKS[targeted_task]["class"][0].from_pretrained(model_id)
        model = get_poplar_executor(model, ipu_config if ipu_config else SUPPORTED_TASKS[targeted_task]["default"]["ipu_config"])
    elif isinstance(model, str):
        model_id = model
        model = SUPPORTED_TASKS[targeted_task]["class"][0].from_pretrained(model)
        model = get_poplar_executor(model, ipu_config if ipu_config else SUPPORTED_TASKS[targeted_task]["default"]["ipu_config"])
    elif isinstance(model, poptorch._poplar_executor.PoplarExecutor):
        if tokenizer is None and load_tokenizer:
            raise ValueError("If you pass a model as a poptorch._poplar_executor.PoplarExecutor, you must pass a tokenizer as well")
        if feature_extractor is None and load_feature_extractor:
            raise ValueError("If you pass a model as a poptorch._poplar_executor.PoplarExecutor, you must pass a feature extractor as well")
    else:
        raise ValueError(
            f"""Model {model} is not supported. Please provide a valid model either as string or poptorch._poplar_executor.PoplarExecutor.
            You can also provide non model then a default one will be used"""
        )

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

    # Override Pipeline __call__ to support auto padding
    old_call = Pipeline.__call__
    def new_call(self, *args, **kwargs):
        if "padding_length" in SUPPORTED_TASKS[targeted_task]["default"] and "padding" not in kwargs:
            kwargs["padding"] = "max_length"
            kwargs["max_length"] = SUPPORTED_TASKS[targeted_task]["default"]["padding_length"]
        return old_call(self, *args, **kwargs)
    Pipeline.__call__ = new_call

    # Override pipelines' _forward to support fp16
    pipeline_class = SUPPORTED_TASKS[targeted_task]["impl"]
    old_forward =pipeline_class._forward
    def new_forward(self, model_inputs, *args, **kwargs):
        for key, input in model_inputs.items():
            if isinstance(input, torch.Tensor) and input.dtype == torch.float32:
                model_inputs[key] = input.half()
        return old_forward(self, model_inputs, *args, **kwargs)
    pipeline_class._forward = new_forward

    return transformers_pipeline(
        task,
        model=model,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        use_fast=use_fast,
        use_auth_token=use_auth_token,
        **kwargs,
    )