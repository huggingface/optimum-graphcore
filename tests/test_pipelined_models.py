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
from unittest import TestCase

import torch
from PIL import Image

import requests
import transformers
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
    AutoFeatureExtractor,
    AutoTokenizer,
)

from .utils import MODELS_TO_TEST_MAPPING


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


MODELS_TO_TEST = _get_models_to_test(MODELS_TO_TEST_MAPPING)


class PipelinedModelsTester(TestCase):
    def _generate_input_for_model_class(self, model_name_or_path, model_class):
        # TODO: add support for labels.
        inputs = None
        extractor = None
        if model_class in [*MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING.values()]:
            extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
        else:
            extractor = AutoTokenizer.from_pretrained(model_name_or_path)
        if model_class in MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING.values():
            encoder_input_text = "This is the text that is going to be fed to the encoder."
            decoder_input_text = "This is part, on the other end, will be fed to the decoder."
            inputs = extractor(encoder_input_text, return_tensors="pt")
            decoder_inputs = extractor(decoder_input_text, return_tensors="pt")
            inputs["decoder_input_ids"] = decoder_inputs["input_ids"]
        elif model_class in MODEL_FOR_MULTIPLE_CHOICE_MAPPING.values():
            prompt = "This is a fake prompt."
            choice0 = "Here is the first choice"
            choice1 = "Here is the second choice"
            inputs = extractor([prompt, prompt], [choice0, choice1], return_tensors="pt", padding=True)
            inputs = {k: v.unsqueeze(0) for k, v in inputs.items()}
        # TODO: do we really need this case?
        elif model_class in MODEL_FOR_QUESTION_ANSWERING_MAPPING.values():
            # LXMERT does visual question answering so it requires visual features as input
            if model_class == transformers.models.lxmert.modeling_lxmert.LxmertForQuestionAnswering:
                inputs = extractor("What is the man wearing?", return_tensors="pt")
                inputs["visual_feats"] = torch.rand(1, 36, 2048)
                inputs["visual_pos"] = torch.rand(1, 36, 4)
            else:
                inputs = extractor("Who was Jim Henson?", "Jim Henson was a nice puppet", return_tensors="pt")
        elif model_class in MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING.values():
            url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            image = Image.open(requests.get(url, stream=True).raw)
            inputs = extractor(images=image, return_tensors="pt")
        else:
            inputs = extractor(
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
        # The forward method can be different in train and eval mode for some models (seq2seq for instance), so we make
        # sure to use the proper one.
        pipelined_forward_function = getattr(pipelined_model, "_forward_for_train", pipelined_model.forward)
        pipelined_model_outputs = pipelined_forward_function(**inputs)

        for idx, t in enumerate(zip(pretrained_model_outputs, pipelined_model_outputs)):
            pretrained_output, pipelined_output = t
            self.assertTrue(
                torch.allclose(pretrained_output, pipelined_output),
                f"Pretrained and pipelined model {idx}th outputs do not match, max difference = {(pretrained_output - pipelined_output).abs().max()}",
            )

    @parameterized.expand(MODELS_TO_TEST)
    def test_parallelize_deparallelize(
        self, test_name, model_name_or_path, ipu_config_name_or_path, pretrained_class, pipelined_class, config_class
    ):
        ipu_config = IPUConfig.from_pretrained(ipu_config_name_or_path)
        model = pipelined_class.from_pretrained_transformers(model_name_or_path, ipu_config)
        model.parallelize()
        model.deparallelize()
        model.parallelize()
