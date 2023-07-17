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
import copy
from unittest import TestCase

import requests
import torch
import transformers
from datasets import load_dataset
from parameterized import parameterized
from PIL import Image
from torch.nn.utils.weight_norm import WeightNorm
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    MODEL_FOR_CTC_MAPPING,
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
    MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
    MODEL_FOR_PRETRAINING_MAPPING,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    MODEL_MAPPING,
    AutoFeatureExtractor,
    AutoProcessor,
    AutoTokenizer,
)

from optimum.graphcore import IPUConfig
from optimum.graphcore.modeling_utils import _PRETRAINED_TO_PIPELINED_REGISTRY

from .utils import CONFIG_MAPPING_EXTRA, MODEL_MAPPING_EXTRA, MODELS_TO_TEST_MAPPING, MODELS_TO_TEST_MAPPING_EXTRA


MODELS_TO_TEST_MAPPING.update(MODELS_TO_TEST_MAPPING_EXTRA)
[CONFIG_MAPPING.register(k, v) for k, v in CONFIG_MAPPING_EXTRA.items()]
REVERSE_CONFIG_MAPPING = {v: k for k, v in CONFIG_MAPPING.items()}


def _get_models_to_test(model_to_test_names):
    def find_config_class_from_pretrained_class(pretrained_class):
        mappings = [
            MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING,
            MODEL_FOR_CAUSAL_LM_MAPPING,
            MODEL_FOR_CTC_MAPPING,
            MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
            MODEL_FOR_MASKED_LM_MAPPING,
            MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
            MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
            MODEL_FOR_PRETRAINING_MAPPING,
            MODEL_FOR_QUESTION_ANSWERING_MAPPING,
            MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
            MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
            MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
            MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
            MODEL_MAPPING,
            MODEL_MAPPING_EXTRA,
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
        names = model_to_test_names[REVERSE_CONFIG_MAPPING[config_class]]
        if isinstance(names, dict):
            task = "ctc" if "CTC" in test_name else "default"
            names = names.get(task, "default")
            model_name_or_path = names.model
            ipu_config = names.ipu_config
        else:
            model_name_or_path = names.model
            ipu_config = names.ipu_config
        models_to_test.append(
            (test_name, model_name_or_path, ipu_config, pretrained_class, pipelined_class, config_class)
        )
    return models_to_test


MODELS_TO_TEST = _get_models_to_test(MODELS_TO_TEST_MAPPING)


class PipelinedModelsTester(TestCase):
    # Copied from transformers hubert tests.
    def _load_superb(self, task, num_samples):
        ds = load_dataset("anton-l/superb_dummy", task, split="test")

        return ds[:num_samples]

    def _load_librispeech(self, num_samples):
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        # automatic decoding with librispeech
        speech_samples = ds.sort("id").select(range(num_samples))[:num_samples]["audio"]

        return [x["array"] for x in speech_samples]

    def _generate_input_for_model_class(self, model_name_or_path, model_class):
        # TODO: add support for labels.
        inputs = None
        extractor = None
        if model_class in [
            *MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING.values(),
            *MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING.values(),
            *MODEL_FOR_CTC_MAPPING.values(),
        ]:
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
        elif model_class in MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING.values():
            input_data = self._load_superb("ic", 1)
            inputs = extractor(input_data["speech"], return_tensors="pt")
        elif model_class in MODEL_FOR_CTC_MAPPING.values() or (
            model_class in MODEL_FOR_PRETRAINING_MAPPING.values()
            and model_class == transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForPreTraining
        ):
            # Wav2Vec2 does speech pretraining, so it requires speech data as input
            extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
            input_speech = self._load_librispeech(1)
            inputs = extractor(input_speech, return_tensors="pt")
        elif model_class in MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING.values():
            input_speech = self._load_librispeech(1)
            processor = AutoProcessor.from_pretrained(model_name_or_path)
            inputs = processor(audio=input_speech, text="This part of the speech", return_tensors="pt")
        else:
            inputs = extractor(
                "This is a test to check that pretrained and pipeline model outputs match.", return_tensors="pt"
            )
        return inputs

    @parameterized.expand(MODELS_TO_TEST)
    def test_pretrained_and_pipelined_models_match(
        self, test_name, model_name_or_path, ipu_config, pretrained_class, pipelined_class, config_class
    ):
        config = config_class.from_pretrained(model_name_or_path)
        if isinstance(ipu_config, str):
            ipu_config = IPUConfig.from_pretrained(ipu_config)

        # Serialized layers split large modules into submodules with the result
        # aggregated/combined across submodules. Since torch uses intra-op parallelism,
        # intermediate thread results may be different when comparing computation on a single module
        # vs the serialized module. Since floating point addition is sensitive to the order of accumulation of
        # intermediate results, results from serialized layers will be marginally different from the original layer.
        # The code below turns off intra-op parallelism since the aim of this test is to test functional correctnesss
        model_using_serialized_splits_per_ipu = (
            ipu_config.serialized_embedding_splits_per_ipu or ipu_config.serialized_projection_splits_per_ipu
        )
        if model_using_serialized_splits_per_ipu:
            torch_original_intra_op_thread_count = torch.get_num_threads()
            torch.set_num_threads(1)

        pretrained_model = pretrained_class(config).eval()
        pipelined_model = pipelined_class.from_transformers(pretrained_model, ipu_config).eval()

        inputs = self._generate_input_for_model_class(model_name_or_path, pretrained_class)
        pretrained_model_outputs = pretrained_model(**inputs, return_dict=True)

        pipelined_model.parallelize(**ipu_config.inference_parallelize_kwargs)
        pipelined_model_outputs = pipelined_model.forward(**inputs, return_dict=True)
        for idx, k in enumerate(pretrained_model_outputs.keys()):
            pretrained_output, pipelined_output = pretrained_model_outputs[k], pipelined_model_outputs[k]
            # Handle tuple outputs. Outputs such as past_key_values are returned as tuples.
            if isinstance(pretrained_output, tuple):
                for x, y in zip(pretrained_output, pipelined_output):
                    if isinstance(x, tuple):
                        for x1, y1 in zip(x, y):
                            self.assertTrue(
                                torch.allclose(x1, y1, atol=1e-5),
                                f"Pretrained and pipelined model {idx}th outputs do not match, max difference = {(x1 - y1).abs().max()}",
                            )
                    else:
                        self.assertTrue(
                            torch.allclose(x, y, atol=1e-5),
                            f"Pretrained and pipelined model {idx}th outputs do not match, max difference = {(x - y).abs().max()}",
                        )

            else:
                self.assertTrue(
                    torch.allclose(pretrained_output, pipelined_output, atol=1e-5),
                    f"Pretrained and pipelined model {idx}th outputs do not match, max difference = {(pretrained_output - pipelined_output).abs().max()}",
                )

        pipelined_model.deparallelize()
        pipelined_model_outputs = pipelined_model.forward(**inputs)
        for idx, k in enumerate(pretrained_model_outputs.keys()):
            pretrained_output, pipelined_output = pretrained_model_outputs[k], pipelined_model_outputs[k]
            # Handle tuple outputs. Outputs such as past_key_values are returned as tuples.
            if isinstance(pretrained_output, tuple):
                for x, y in zip(pretrained_output, pipelined_output):
                    if isinstance(x, tuple):
                        for x1, y1 in zip(x, y):
                            self.assertTrue(
                                torch.allclose(x1, y1, atol=1e-5),
                                f"Pretrained and pipelined model {idx}th outputs do not match, max difference = {(x1 - y1).abs().max()}",
                            )
                    else:
                        self.assertTrue(
                            torch.allclose(x, y, atol=1e-5),
                            f"Pretrained and pipelined model {idx}th outputs do not match, max difference = {(x - y).abs().max()}",
                        )

            else:
                self.assertTrue(
                    torch.allclose(pretrained_output, pipelined_output, atol=1e-5),
                    f"Pretrained and pipelined model {idx}th outputs do not match, max difference = {(pretrained_output - pipelined_output).abs().max()}",
                )

        if model_using_serialized_splits_per_ipu:
            torch.set_num_threads(torch_original_intra_op_thread_count)

    @parameterized.expand(MODELS_TO_TEST)
    def test_parallelize_deparallelize(
        self, test_name, model_name_or_path, ipu_config, pretrained_class, pipelined_class, config_class
    ):
        if isinstance(ipu_config, str):
            ipu_config = IPUConfig.from_pretrained(ipu_config)

        model = pipelined_class.from_pretrained_transformers(model_name_or_path, ipu_config)

        # Remove the weight-norm hook, if present, because it doesn't work with deepcopy
        # https://github.com/pytorch/pytorch/issues/28594
        for module in model.modules():
            for _, hook in module._forward_pre_hooks.items():
                if isinstance(hook, WeightNorm):
                    delattr(module, hook.name)

        modules_before = copy.deepcopy(model).modules()
        model.parallelize(**ipu_config.parallelize_kwargs)
        model.deparallelize()
        modules_after = copy.deepcopy(model).modules()
        # Confirm that parallelize then deparallelize won't change the model's modules
        for mod_before, mod_after in zip(modules_before, modules_after):
            self.assertEqual(type(mod_before), type(mod_after))

        model.parallelize(**ipu_config.parallelize_kwargs)
