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
import unittest

import torch

from optimum.graphcore import IPUConfig
from optimum.graphcore.modeling_utils import _PRETRAINED_TO_PIPELINED_REGISTRY
from parameterized import parameterized
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
    MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
    MODEL_FOR_PRETRAINING_MAPPING,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    MODEL_MAPPING,
    AutoTokenizer,
)


REVERSE_CONFIG_MAPPING = {v: k for k, v in CONFIG_MAPPING.items()}


def _get_models_to_test(model_to_test_names):
    def find_config_class_from_pretrained_class(pretrained_class):
        mappings = [
            MODEL_FOR_CAUSAL_LM_MAPPING,
            MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
            MODEL_FOR_MASKED_LM_MAPPING,
            MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
            MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
            MODEL_FOR_PRETRAINING_MAPPING,
            MODEL_FOR_QUESTION_ANSWERING_MAPPING,
            MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
            MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
            MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
            MODEL_MAPPING,
        ]
        config_class = None
        for mapping in mappings:
            for k, v in mapping.items():
                if v is pretrained_class:
                    config_class = k
                    break
            if config_class is not None:
                break
        if config_class is None:
            # TODO: what is better than ValueError in this case?
            raise ValueError(f"could not find a configuration class from {pretrained_class}")
        return config_class

    models_to_test = []
    for pretrained_class, pipelined_class in _PRETRAINED_TO_PIPELINED_REGISTRY.items():
        test_name = f"{pretrained_class.__name__}"
        config_class = find_config_class_from_pretrained_class(pretrained_class)
        model_name_or_path, ipu_config_name_or_path = model_to_test_names[REVERSE_CONFIG_MAPPING[config_class]]
        models_to_test.append(
            (test_name, model_name_or_path, ipu_config_name_or_path, pretrained_class, pipelined_class, config_class)
        )
    return models_to_test


MODELS_TO_TEST_NAMES = {
    "bert": ("bert-base-uncased", "Graphcore/bert-base-ipu"),
    "roberta": ("roberta-base", "Graphcore/roberta-base-ipu"),
    "vit": ("google/vit-base-patch16-224", "Graphcore/vit-base-ipu"),
}
MODELS_TO_TEST = _get_models_to_test(MODELS_TO_TEST_NAMES)


class PipelinedModelsTester(unittest.TestCase):
    def _generate_input_for_model_class(self, model_name_or_path, model_class):
        # TODO: add support for labels.
        inputs = None
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if model_class in MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING.values():
            raise NotImplementedError
        elif model_class in MODEL_FOR_MULTIPLE_CHOICE_MAPPING.values():
            prompt = "This is a fake prompt."
            choice0 = "Here is the first choice"
            choice1 = "Here is the second choice"
            inputs = tokenizer([prompt, prompt], [choice0, choice1], return_tensors="pt", padding=True)
            inputs = {k: v.unsqueeze(0) for k, v in inputs.items()}
        # TODO: do we really need this case?
        elif model_class in MODEL_FOR_QUESTION_ANSWERING_MAPPING.values():
            inputs = tokenizer("Who was Jim Henson?", "Jim Henson was a nice puppet", return_tensors="pt")
        else:
            inputs = tokenizer(
                "This is a test to check that pretrained and pipeline model outputs match.", return_tensors="pt"
            )
        return inputs

    @parameterized.expand(MODELS_TO_TEST)
    def test_pretrained_and_pipelined_models_match(
        self, test_name, model_name_or_path, ipu_config_name_or_path, pretrained_class, pipelined_class, config_class
    ):
        config = config_class.from_pretrained(model_name_or_path)
        ipu_config = IPUConfig.from_pretrained(ipu_config_name_or_path)
        pretrained_model = pretrained_class(config).eval()
        pipelined_model = pipelined_class.from_transformers(pretrained_model, ipu_config).eval()

        inputs = self._generate_input_for_model_class(model_name_or_path, pretrained_class)
        pretrained_model_outputs = pretrained_model(**inputs, return_dict=False)
        pipelined_model_outputs = pipelined_model(**inputs)

        for pretrained_output, pipelined_output in zip(pretrained_model_outputs, pipelined_model_outputs):
            self.assertTrue(torch.allclose(pretrained_output, pipelined_output))

    @parameterized.expand(MODELS_TO_TEST)
    def test_parallelize_deparallelize(
        self, test_name, model_name_or_path, ipu_config_name_or_path, pretrained_class, pipelined_class, config_class
    ):
        ipu_config = IPUConfig.from_pretrained(ipu_config_name_or_path)
        model = pipelined_class.from_pretrained_transformers(model_name_or_path, ipu_config)
        model.parallelize()
        model.deparallelize()
        model.parallelize()
