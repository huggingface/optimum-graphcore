# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

#  Copyright 2021 The HuggingFace Team. All rights reserved.
#  Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import ctypes
import os
import poptorch

from .ipu_configuration import IPUConfig
from .models.bart import PipelinedBartForConditionalGeneration, PipelinedBartForSequenceClassification
from .models.bert import (
    PipelinedBertModel,
    PipelinedBertForMaskedLM,
    PipelinedBertForMultipleChoice,
    PipelinedBertForPreTraining,
    PipelinedBertForQuestionAnswering,
    PipelinedBertForSequenceClassification,
    PipelinedBertForTokenClassification,
)
from .models.convnext import PipelinedConvNextForImageClassification
from .models.distilbert import (
    PipelinedDistilBertForMaskedLM,
    PipelinedDistilBertForMultipleChoice,
    PipelinedDistilBertForQuestionAnswering,
    PipelinedDistilBertForSequenceClassification,
    PipelinedDistilBertForTokenClassification,
)
from .models.gpt2 import (
    PipelinedGPT2ForSequenceClassification,
    PipelinedGPT2ForTokenClassification,
    PipelinedGPT2LMHeadModel,
)

from .models.hubert import PipelinedHubertForSequenceClassification
from .models.lxmert import PipelinedLxmertForQuestionAnswering
from .models.mt5 import PipelinedMT5ForConditionalGeneration
from .models.roberta import (
    PipelinedRobertaForMaskedLM,
    PipelinedRobertaForMultipleChoice,
    PipelinedRobertaForQuestionAnswering,
    PipelinedRobertaForSequenceClassification,
    PipelinedRobertaForTokenClassification,
)
from .models.t5 import (
    PipelinedT5ForConditionalGeneration,
    PipelinedT5EncoderModel,
)
from .models.vit import PipelinedViTForImageClassification
from .models.wav2vec2 import PipelinedWav2Vec2ForPreTraining

from .models.mpnet import PipelinedMPNetModel, PipelinedMPNetForMaskedLM

from .pipelines import IPUFillMaskPipeline, IPUTokenClassificationPipeline, pipeline
from .trainer import IPUTrainer, IPUTrainerState
from .trainer_seq2seq import IPUSeq2SeqTrainer
from .training_args import IPUTrainingArguments
from .training_args_seq2seq import IPUSeq2SeqTrainingArguments
from .version import __version__


# Disable poptorch compiler warnings by default
poptorch.setLogLevel("ERR")


# # Load the custom ops
# def _load_custom_ops():
#     import ctypes
#     import pkg_resources
#     import sysconfig
#     from pathlib import Path

#     root = Path(pkg_resources.get_distribution("optimum-graphcore").location).absolute()
#     names = ["custom_ops.so", str(Path("custom_ops.so").with_suffix(sysconfig.get_config_vars()["SO"]))]
#     paths = [
#         root / "build" / names[0],
#         root / names[1],
#     ]
#     print("CUSTOM OPS", paths)
#     for path in paths:
#         if path.exists():
#             ctypes.cdll.LoadLibrary(str(path))
#             print("Loading:", path)
#             return
#     # Search recursively in build dir if not found in first level
#     for name in names:
#         for path in (root / "build").rglob(str(name)):
#             if path.exists():
#                 ctypes.cdll.LoadLibrary(str(path))
#                 print("Loading:", path)
#                 return
#     raise ImportError(f"Cannot find extension library {name} - tried {[str(p) for p in paths]}")  # pragma: no cover


# _load_custom_ops()
