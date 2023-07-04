# coding=utf-8
# Copyright 2021 HuggingFace Inc.
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
import pytest
from transformers import T5Config, T5EncoderModel
from optimum.graphcore.models.t5.configuration_t5 import T5EncoderConfig

MODELS_TO_TEST_MAPPING = {
    "bart": ("facebook/bart-base", "Graphcore/bart-base-ipu"),
    "bert": ("bert-base-uncased", "Graphcore/bert-base-ipu"),
    "convnext": ("facebook/convnext-base-224", "Graphcore/convnext-base-ipu"),
    "deberta": ("microsoft/deberta-base", "Graphcore/deberta-base-ipu"),
    "distilbert": ("distilbert-base-uncased", "Graphcore/distilbert-base-ipu"),
    "gpt2": ("gpt2", "Graphcore/gpt2-small-ipu"),
    "groupbert": ("Graphcore/groupbert-base-uncased", "Graphcore/groupbert-base-uncased"),
    "hubert": {
        "default": ("facebook/hubert-base-ls960", "Graphcore/hubert-base-ipu"),
        "ctc": ("facebook/hubert-base-ls960", "Graphcore/wav2vec2-ctc-base-ipu"),
    },
    "lxmert": ("unc-nlp/lxmert-base-uncased", "Graphcore/lxmert-base-ipu"),
    "roberta": ("roberta-base", "Graphcore/roberta-base-ipu"),
    "t5": ("t5-small", "Graphcore/t5-small-ipu"),
    "mt5": ("google/mt5-small", "Graphcore/mt5-small-ipu"),
    "vit": ("google/vit-base-patch16-224-in21k", "Graphcore/vit-base-ipu"),
    "wav2vec2": {
        "default": ("facebook/wav2vec2-base", "Graphcore/wav2vec2-base-ipu"),
        "ctc": ("facebook/wav2vec2-base", "Graphcore/wav2vec2-ctc-base-ipu"),
    },
    "whisper": ("openai/whisper-tiny", "Graphcore/whisper-tiny-ipu"),
}

# Register models that don't have a mapping in upstream transformers
MODEL_MAPPING_EXTRA = {T5EncoderConfig: T5EncoderModel}
CONFIG_MAPPING_EXTRA = {"t5encoder": T5EncoderConfig}
MODELS_TO_TEST_MAPPING_EXTRA = {
    "t5encoder": ("sentence-transformers/sentence-t5-base", "Graphcore/sentence-t5-base"),
}


def skip_unsupported(feature):
    return pytest.mark.skip(f"Skipping since {feature} is not yet supported in Optimum Graphcore")
