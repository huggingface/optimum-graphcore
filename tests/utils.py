# coding=utf-8
# Copyright 2021 HuggingFace Inc.
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
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
import unittest
from collections import namedtuple

import pytest
from transformers import T5EncoderModel
from transformers.testing_utils import parse_flag_from_env

from optimum.graphcore import IPUConfig
from optimum.graphcore.models.t5.configuration_t5 import T5EncoderConfig


_run_high_memory_usage_tests = parse_flag_from_env("RUN_HIGH_MEMORY", default=False)

model_test_config = namedtuple(
    "ModelTestConfig",
    ["model", "ipu_config", "test_examples_config"],
    defaults=[
        "hf-internal-testing/tiny-random-t5",
        "Graphcore/internal-testing-tiny-ipu",
        {},
    ],
)

MODELS_TO_TEST_MAPPING = {
    "bart": model_test_config("facebook/bart-base", "Graphcore/bart-base-ipu"),
    "bert": model_test_config("bert-base-uncased", "Graphcore/bert-base-ipu"),
    "convnext": model_test_config("facebook/convnext-base-224", "Graphcore/convnext-base-ipu"),
    "deberta": model_test_config("microsoft/deberta-base", "Graphcore/deberta-base-ipu"),
    "distilbert": model_test_config("distilbert-base-uncased", "Graphcore/distilbert-base-ipu"),
    "gpt2": model_test_config("gpt2", "Graphcore/gpt2-small-ipu"),
    "groupbert": model_test_config("Graphcore/groupbert-base-uncased", "Graphcore/groupbert-base-uncased"),
    "hubert": {
        "default": model_test_config("facebook/hubert-base-ls960", "Graphcore/hubert-base-ipu"),
        "ctc": model_test_config("facebook/hubert-base-ls960", "Graphcore/wav2vec2-ctc-base-ipu"),
    },
    "lxmert": model_test_config("unc-nlp/lxmert-base-uncased", "Graphcore/lxmert-base-ipu"),
    "roberta": model_test_config("roberta-base", "Graphcore/roberta-base-ipu"),
    "t5": model_test_config("t5-small", "Graphcore/t5-small-ipu"),
    "mt5": {
        "default": model_test_config("google/mt5-small", "Graphcore/mt5-small-ipu"),
        "translation": model_test_config(
            "google/mt5-small",
            IPUConfig.from_pretrained(
                "Graphcore/mt5-small-ipu",
                embedding_serialization_factor=None,
                serialized_embedding_splits_per_ipu=[4, 4, 0, 0],
                layers_per_ipu=[0, 0, 16, 0],
            ),
            {"extra_command_line_arguments": ["--max_source_length 128", "--max_target_length 128"]},
        ),
        "summarization": model_test_config(
            "google/mt5-small",
            IPUConfig.from_pretrained(
                "Graphcore/mt5-small-ipu",
                ipus_per_replica=8,
                layers_per_ipu=[0, 1, 2, 2, 5, 6, 0, 0],
                embedding_serialization_factor=None,
                projection_serialization_factor=None,
                serialized_embedding_splits_per_ipu=[4, 4, 0, 0, 0, 0, 0, 0],
                serialized_projection_splits_per_ipu=[0, 0, 0, 0, 0, 0, 4, 4],
            ),
        ),
    },
    "mpnet": model_test_config("sentence-transformers/all-mpnet-base-v2", "Graphcore/mpnet-base-ipu"),
    "vit": model_test_config("google/vit-base-patch16-224-in21k", "Graphcore/vit-base-ipu"),
    "wav2vec2": {
        "default": model_test_config("facebook/wav2vec2-base", "Graphcore/wav2vec2-base-ipu"),
        "ctc": model_test_config("facebook/wav2vec2-base", "Graphcore/wav2vec2-ctc-base-ipu"),
    },
    "whisper": model_test_config("openai/whisper-tiny", "Graphcore/whisper-tiny-ipu"),
}

# Register models that don't have a mapping in upstream transformers
MODEL_MAPPING_EXTRA = {T5EncoderConfig: T5EncoderModel}
CONFIG_MAPPING_EXTRA = {"t5encoder": T5EncoderConfig}
MODELS_TO_TEST_MAPPING_EXTRA = {
    "t5encoder": model_test_config("sentence-transformers/sentence-t5-base", "Graphcore/sentence-t5-base"),
}


def skip_unsupported(feature):
    return pytest.mark.skip(f"Skipping since {feature} is not yet supported in Optimum Graphcore")


def high_memory_usage(test_case):
    """
    Decorator marking a test as using a large amount of DRAM.

    This test is skipped by default. Set the RUN_HIGH_MEMORY environment variable to a truthy value to run them.
    """
    return unittest.skipUnless(_run_high_memory_usage_tests, "test requires high resident memory in DRAM")(test_case)
