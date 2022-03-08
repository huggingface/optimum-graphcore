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

import json
import os
import subprocess
from functools import wraps
from pathlib import Path
from parameterized import parameterized
from tempfile import TemporaryDirectory
from typing import Dict, List, Tuple, Type
from unittest import TestCase

from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
)

from optimum.graphcore.modeling_utils import _PRETRAINED_TO_PIPELINED_REGISTRY

from .utils import MODELS_TO_TEST_MAPPING


def _get_supported_model_names_for_script(
    model_names_mapping: Dict[str, Tuple[str]], task_mapping: Dict[str, str]
) -> List[Tuple[str]]:

    def is_valid_model_type(model_type: str, model_class: Type) -> bool:
        in_task_mapping = CONFIG_MAPPING[model_type] in task_mapping
        if in_task_mapping:
            return task_mapping[CONFIG_MAPPING[model_type]] in _PRETRAINED_TO_PIPELINED_REGISTRY
        return False

    return [(model_type, names) for (model_type, names) in model_names_mapping.items() if is_valid_model_type(model_type, names)]


_SCRIPT_TO_MODEL_MAPPING = {
    "run_mlm": _get_supported_model_names_for_script(MODELS_TO_TEST_MAPPING, MODEL_FOR_MASKED_LM_MAPPING),
    "run_swag": _get_supported_model_names_for_script(MODELS_TO_TEST_MAPPING, MODEL_FOR_MULTIPLE_CHOICE_MAPPING),
    "run_qa": _get_supported_model_names_for_script(MODELS_TO_TEST_MAPPING, MODEL_FOR_QUESTION_ANSWERING_MAPPING),
    "run_summarization": _get_supported_model_names_for_script(
        MODELS_TO_TEST_MAPPING, MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING
    ),
    "run_translation": _get_supported_model_names_for_script(
        MODELS_TO_TEST_MAPPING, MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING
    ),
    "run_glue": _get_supported_model_names_for_script(
        MODELS_TO_TEST_MAPPING, MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING
    ),
    "run_ner": _get_supported_model_names_for_script(MODELS_TO_TEST_MAPPING, MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING),
    "run_image_classification": _get_supported_model_names_for_script(
        MODELS_TO_TEST_MAPPING, MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING
    ),
}


class ExampleTestMeta(type):

    def __new__(cls, name, bases, attrs, example_name=None):
        models_to_test = []
        if example_name is not None:
            models_to_test = _SCRIPT_TO_MODEL_MAPPING.get(example_name)
            if models_to_test is None:
                raise ValueError(f"could not find models for example {example_name}")
        for model_type, names in models_to_test:
            model_name, ipu_config_name = names
            attrs[f"test_{example_name}_{model_type}"] = cls._create_test(model_name, ipu_config_name)
        attrs["EXAMPLE_NAME"] = example_name
        return super().__new__(cls, name, bases, attrs)

    def _create_command_line(
        self,
        script: str,
        task: str,
        model_name: str,
        ipu_config_name: str,
        output_dir: str,
        do_eval: bool = True,
        lr: float = 1e-5,
        train_batch_size: int = 2,
        eval_batch_size: int = 2,
        num_epochs: int = 3,
    ) -> List[str]:
        do_eval_option = "--do-eval \\" if do_eval else ""
        task_option = f"--dataset_name {task} \\" if task else ""
        return f"""
            {script} \
             --model_name_or_path {model_name} \
             --ipu_config_name {ipu_config_name} \
             {task_option}
             --do_train \
             {do_eval_option}
             --output_dir {output_dir} \
             --overwrite_output_dir \
             --learning_rate {lr} \
             --per_device_train_batch_size {train_batch_size} \
             --per_device_eval_batch_size {eval_batch_size} \
             --save_strategy epochs \
             --num_epochs {num_epochs} \
             """.split()

    @classmethod
    def _create_test(cls, model_name: str, ipu_config_name: str):

        def test(self):
            return True
            with TemporaryDirectory() as tmp_dir:
                cmd_line = self._create_command_line(
                    str(self.EXAMPLE_DIR / self.EXAMPLE_NAME) + ".py",
                    self.TASK_NAME,
                    model_name,
                    ipu_config_name,
                    tmp_dir,
                    do_eval=self.EVAL_IS_SUPPORTED,
                )
                p = subprocess.Popen(cmd_line)
                return_code = p.wait()
                # TODO: not sure about that.
                self.assertEqual(return_code, 0)

                if self.EVAL_IS_SUPPORTED:
                    with open(tmp_dir / "all_results.json") as fp:
                        results = json.load(fp)
                    self.assertGreaterEqual(results["eval_accuracy"], self.EVAL_ACCURACY_THRESHOLD)

        return test


class ExampleTesterBase(TestCase):
    EXAMPLE_DIR = Path(os.path.dirname(__file__)) / "examples"
    EXAMPLE_NAME = None
    TASK_NAME = None
    EVAL_IS_SUPPORTED = True
    EVAL_ACCURACY_THRESHOLD = 0.75

    # def _create_command_line(
    #     self,
    #     script: str,
    #     task: str,
    #     model_name: str,
    #     ipu_config_name: str,
    #     output_dir: str,
    #     do_eval: bool = True,
    #     lr: float = 1e-5,
    #     train_batch_size: int = 2,
    #     eval_batch_size: int = 2,
    #     num_epochs: int = 3,
    # ) -> List[str]:
    #     do_eval_option = "--do-eval \\" if do_eval else ""
    #     return f"""
    #         {script} \
    #          --model_name_or_path {model_name} \
    #          --ipu_config_name {ipu_config_name} \
    #          --do_train \
    #          {do_eval_option}
    #          --output_dir {output_dir} \
    #          --overwrite_output_dir \
    #          --learning_rate {lr} \
    #          --per_device_train_batch_size {train_batch_size} \
    #          --per_device_eval_batch_size {eval_batch_size} \
    #          --save_strategy epochs \
    #          --num_epochs {num_epochs} \
    #          """.split()


class TextClassificationExampleTester(ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_glue"):
    TASK_NAME = "sst2"


class TokenClassificationExampleTester(TestCase, metaclass=ExampleTestMeta, example_name="run_ner"):
    TASK_NAME = "conll2003"


class MultipleChoiceExampleTester(TestCase, metaclass=ExampleTestMeta, example_name="run_swag"):
    pass


class QuestionAnsweringExampleTester(TestCase, metaclass=ExampleTestMeta, example_name="run_qa"):
    TASK_NAME = "squad"


class SummarizationExampleTester(TestCase, metaclass=ExampleTestMeta, example_name="run_summarization"):
    TASK_NAME = "cnn_dailymail"


class TranslationExampleTester(TestCase, metaclass=ExampleTestMeta, example_name="run_translation"):
    TASK_NAME = "wmt16"


class ImageClassificationExampleTester(TestCase, metaclass=ExampleTestMeta, example_name="run_image_classification"):
    TASK_NAME = "cifar10"
