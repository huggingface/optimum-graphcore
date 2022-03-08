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
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Dict, List, Optional, Tuple, Type
from unittest import TestCase

from optimum.graphcore.modeling_utils import _PRETRAINED_TO_PIPELINED_REGISTRY
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

from .utils import MODELS_TO_TEST_MAPPING


def _get_supported_models_for_script(
    models_to_test: Dict[str, Tuple[str]], task_mapping: Dict[str, str]
) -> List[Tuple[str]]:
    """
    Filters models that can perform the task from models_to_test.

    Args:
        models_to_test: mapping between a model type and a tuple (model_name_or_path, ipu_config_name).
        task_mapping: mapping bewteen a model config and a model class.

    Returns:
        A list of models that are supported for the task.
        Each element of the list follows the same format: (model_type, (model_name_or_path, ipu_config_name)).
    """

    def is_valid_model_type(model_type: str, model_class: Type) -> bool:
        in_task_mapping = CONFIG_MAPPING[model_type] in task_mapping
        if in_task_mapping:
            return task_mapping[CONFIG_MAPPING[model_type]] in _PRETRAINED_TO_PIPELINED_REGISTRY
        return False

    return [
        (model_type, names) for (model_type, names) in models_to_test.items() if is_valid_model_type(model_type, names)
    ]


_SCRIPT_TO_MODEL_MAPPING = {
    "run_mlm": _get_supported_models_for_script(MODELS_TO_TEST_MAPPING, MODEL_FOR_MASKED_LM_MAPPING),
    "run_swag": _get_supported_models_for_script(MODELS_TO_TEST_MAPPING, MODEL_FOR_MULTIPLE_CHOICE_MAPPING),
    "run_qa": _get_supported_models_for_script(MODELS_TO_TEST_MAPPING, MODEL_FOR_QUESTION_ANSWERING_MAPPING),
    "run_summarization": _get_supported_models_for_script(
        MODELS_TO_TEST_MAPPING, MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING
    ),
    "run_translation": _get_supported_models_for_script(
        MODELS_TO_TEST_MAPPING, MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING
    ),
    "run_glue": _get_supported_models_for_script(MODELS_TO_TEST_MAPPING, MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING),
    "run_ner": _get_supported_models_for_script(MODELS_TO_TEST_MAPPING, MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING),
    "run_image_classification": _get_supported_models_for_script(
        MODELS_TO_TEST_MAPPING, MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING
    ),
}


class ExampleTestMeta(type):
    """
    Metaclass that takes care of creating the proper example tests for a given task.

    It uses example_name to figure out which models support this task, and create a run example test for each of these
    models.
    """

    def __new__(cls, name, bases, attrs, example_name=None):
        models_to_test = []
        if example_name is not None:
            models_to_test = _SCRIPT_TO_MODEL_MAPPING.get(example_name)
            if models_to_test is None:
                raise AttributeError(f"could not create class because no model was found for example {example_name}")
        for model_type, names in models_to_test:
            model_name, ipu_config_name = names
            attrs[f"test_{example_name}_{model_type}"] = cls._create_test(model_name, ipu_config_name)
        attrs["EXAMPLE_NAME"] = example_name
        return super().__new__(cls, name, bases, attrs)

    @classmethod
    def _create_test(cls, model_name: str, ipu_config_name: str) -> Callable[[], None]:
        """
        Creates a test function that runs an example for a specific (model_name, ipu_config_name) pair.

        Args:
            model_name: the model_name_or_path.
            ipu_config_name: the ipu config name.

        Returns:
            The test function that runs the example.
        """

        def test(self):
            example_script = Path(self.EXAMPLE_DIR).glob(f"*/{self.EXAMPLE_NAME}.py")
            example_script = list(example_script)
            if len(example_script) == 0:
                raise RuntimeError(f"could not find {self.EXAMPLE_NAME}.py in examples located in {self.EXAMPLE_DIR}")
            elif len(example_script) > 1:
                raise RuntimeError(f"found more than {self.EXAMPLE_NAME}.py in examples located in {self.EXAMPLE_DIR}")
            else:
                example_script = example_script[0]

            with TemporaryDirectory() as tmp_dir:
                cmd_line = self._create_command_line(
                    example_script,
                    model_name,
                    ipu_config_name,
                    tmp_dir,
                    task=self.TASK_NAME,
                    do_eval=self.EVAL_IS_SUPPORTED,
                )
                import pdb; pdb.set_trace()
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
    """
    Base example tester class.

    Attributes:
        EXAMPLE_DIR (`Pathlike`): the directory containing the examples.
        EXAMPLE_NAME (`str`): the name of the example script without the file extension, e.g. run_qa, run_glue, etc.
        TASK_NAME (`str`): the name of the dataset to use.
        EVAL_IS_SUPPORTED (`bool`): whether evaluation is currently supported on IPUs.
            If True, the example will run evaluation, otherwise it will be skipped.
        EVAL_ACCURACY_THRESHOLD (`float`): the score threshold from which training is assumed to have worked.
    """

    EXAMPLE_DIR = Path(os.path.dirname(__file__)).parent / "examples"
    EXAMPLE_NAME = None
    TASK_NAME = None
    EVAL_IS_SUPPORTED = True
    EVAL_ACCURACY_THRESHOLD = 0.75

    def _create_command_line(
        self,
        script: str,
        model_name: str,
        ipu_config_name: str,
        output_dir: str,
        task: Optional[str] = None,
        do_eval: bool = True,
        lr: float = 1e-5,
        train_batch_size: int = 2,
        eval_batch_size: int = 2,
        num_epochs: int = 3,
    ) -> List[str]:
        do_eval_option = "--do_eval" if do_eval else " "
        task_option = f"--dataset_name {task}" if task else " "
        cmd_line = [
            f"{script}",
            f"--model_name_or_path {model_name}",
            f"--ipu_config_name {ipu_config_name}",
            f"{task_option}",
            "--do_train",
            f"{do_eval_option}",
            f"--output_dir {output_dir}",
            "--overwrite_output_dir true",
            f"--learning_rate {lr}",
            f"--per_device_train_batch_size {train_batch_size}",
            f"--per_device_eval_batch_size {eval_batch_size}",
            "--save_strategy epoch",
            "--ipu_config_overrides 'executable_cache_dir=None'",
            f" --num_train_epochs {num_epochs}",
        ]
        return [x for y in cmd_line for x in y.split()]


class TextClassificationExampleTester(ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_glue"):
    TASK_NAME = "sst2"

    def _create_command_line(
        self,
        script: str,
        model_name: str,
        ipu_config_name: str,
        output_dir: str,
        task: Optional[str] = None,
        do_eval: bool = True,
        lr: float = 1e-5,
        train_batch_size: int = 2,
        eval_batch_size: int = 2,
        num_epochs: int = 3,
    ) -> List[str]:
        do_eval_option = "--do_eval" if do_eval else " "
        task_option = f"--task_name {task}" if task else " "
        cmd_line = [
            f"{script}",
            f"--model_name_or_path {model_name}",
            f"--ipu_config_name {ipu_config_name}",
            f"{task_option}",
            "--do_train",
            f"{do_eval_option}",
            f"--output_dir {output_dir}",
            "--overwrite_output_dir true",
            f"--learning_rate {lr}",
            f"--per_device_train_batch_size {train_batch_size}",
            f"--per_device_eval_batch_size {eval_batch_size}",
            "--save_strategy epoch",
            "--ipu_config_overrides 'executable_cache_dir=None'",
            f" --num_train_epochs {num_epochs}",
        ]
        return [x for y in cmd_line for x in y.split()]


class TokenClassificationExampleTester(ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_ner"):
    TASK_NAME = "conll2003"


class MultipleChoiceExampleTester(ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_swag"):
    pass


class QuestionAnsweringExampleTester(ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_qa"):
    TASK_NAME = "squad"


class SummarizationExampleTester(ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_summarization"):
    TASK_NAME = "cnn_dailymail"
    # TODO: handle prefix pr t5


class TranslationExampleTester(ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_translation"):
    TASK_NAME = "wmt16"
    # TODO: handle prefix pr t5 and dataset_config


class ImageClassificationExampleTester(
    ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_image_classification"
):
    TASK_NAME = "cifar10"
