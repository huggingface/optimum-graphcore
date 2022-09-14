import poptorch

import torch

from optimum.graphcore import IPUConfig
from optimum.graphcore.modeling_utils import to_pipelined

from typing import Any, Optional, Union

from transformers import (
    AutoModelForSequenceClassification,
    Pipeline,
    PreTrainedTokenizer,
    TextClassificationPipeline,
)
from transformers import pipeline as transformers_pipeline
from transformers.feature_extraction_utils import PreTrainedFeatureExtractor
from transformers.modeling_utils import PreTrainedModel
from transformers.onnx.utils import get_preprocessor


SUPPORTED_TASKS = {
    "text-classification": {
        "impl": TextClassificationPipeline,
        "class": (AutoModelForSequenceClassification,),
        "default": "cardiffnlp/twitter-roberta-base-sentiment",
        "default_ipu_config": "Graphcore/roberta-base-ipu",
        "type": "text",
    },
}

NO_FEATURE_EXTRACTOR_TASKS = set()
NO_TOKENIZER_TASKS = set()
for task, values in SUPPORTED_TASKS.items():
    if values["type"] == "text":
        NO_FEATURE_EXTRACTOR_TASKS.add(task)
    elif values["type"] == "image":
        NO_TOKENIZER_TASKS.add(task)
    else:
        raise ValueError(f"Supported types are 'text' and 'image', got {values['type']}")

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

def get_inference_context(self):
        return torch.no_grad

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
        model_id = SUPPORTED_TASKS[targeted_task]["default"]
        model = SUPPORTED_TASKS[targeted_task]["class"][0].from_pretrained(model_id)
        model = get_poplar_executor(model, ipu_config if ipu_config else SUPPORTED_TASKS[targeted_task]["default_ipu_config"])
    elif isinstance(model, str):
        model_id = model
        model = SUPPORTED_TASKS[targeted_task]["class"][0].from_pretrained(model)
        model = get_poplar_executor(model, ipu_config if ipu_config else SUPPORTED_TASKS[targeted_task]["default_ipu_config"])
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

    if tokenizer is None and load_tokenizer:
        tokenizer = get_preprocessor(model_id)
    if feature_extractor is None and load_feature_extractor:
        feature_extractor = get_preprocessor(model_id)

    # Modify transformers.pipelines
    # Override get_inference_context()
    Pipeline.get_inference_context = get_inference_context

    return transformers_pipeline(
        task,
        model=model,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        use_fast=use_fast,
        use_auth_token=use_auth_token,
        **kwargs,
    )