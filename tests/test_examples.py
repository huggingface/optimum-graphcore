# coding=utf-8
# Copyright 2022 HuggingFace Inc.
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

import glob
import json
import os
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Dict, List, Optional, Tuple, Union
from unittest import TestCase

from filelock import FileLock
from optimum.graphcore.modeling_utils import _PRETRAINED_TO_PIPELINED_REGISTRY
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    MODEL_FOR_CTC_MAPPING,
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
)
from transformers.testing_utils import slow

from .utils import MODELS_TO_TEST_MAPPING


def _get_supported_models_for_script(
    models_to_test: Dict[str, Tuple[str]], task_mapping: Dict[str, str], task: str = "default"
) -> List[Tuple[str]]:
    """
    Filters models that can perform the task from models_to_test.

    Args:
        models_to_test: mapping between a model type and a tuple (model_name_or_path, ipu_config_name).
        task_mapping: mapping bewteen a model config and a model class.
        task: the task to get the model names for.

    Returns:
        A list of models that are supported for the task.
        Each element of the list follows the same format: (model_type, (model_name_or_path, ipu_config_name)).
    """

    def is_valid_model_type(model_type: str) -> bool:
        in_task_mapping = CONFIG_MAPPING[model_type] in task_mapping
        if in_task_mapping:
            return task_mapping[CONFIG_MAPPING[model_type]] in _PRETRAINED_TO_PIPELINED_REGISTRY
        return False

    supported_models = []
    for model_type, model_names in models_to_test.items():
        names = model_names.get(task, model_names["default"]) if isinstance(model_names, dict) else model_names
        if is_valid_model_type(model_type):
            supported_models.append((model_type, names))

    return supported_models


_SCRIPT_TO_MODEL_MAPPING = {
    "run_clm": _get_supported_models_for_script(MODELS_TO_TEST_MAPPING, MODEL_FOR_CAUSAL_LM_MAPPING),
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
    "run_audio_classification": _get_supported_models_for_script(
        MODELS_TO_TEST_MAPPING, MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING
    ),
    "run_speech_recognition_ctc": _get_supported_models_for_script(
        MODELS_TO_TEST_MAPPING, MODEL_FOR_CTC_MAPPING, task="ctc"
    ),
}
# Take LXMERT out of run_qa because it's incompatible
_SCRIPT_TO_MODEL_MAPPING["run_qa"] = [x for x in _SCRIPT_TO_MODEL_MAPPING["run_qa"] if x[0] != "lxmert"]


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

        @slow
        def test(self):
            if self.EXAMPLE_NAME is None:
                raise ValueError("An example name must be provided")
            example_script = Path(self.EXAMPLE_DIR).glob(f"*/{self.EXAMPLE_NAME}.py")
            example_script = list(example_script)
            if len(example_script) == 0:
                raise RuntimeError(f"Could not find {self.EXAMPLE_NAME}.py in examples located in {self.EXAMPLE_DIR}")
            elif len(example_script) > 1:
                raise RuntimeError(f"Found more than {self.EXAMPLE_NAME}.py in examples located in {self.EXAMPLE_DIR}")
            else:
                example_script = example_script[0]

            self._install_requirements(example_script.parent / "requirements.txt")

            with TemporaryDirectory(dir=Path(self.EXAMPLE_DIR)) as tmp_dir:
                os.environ["HF_HOME"] = os.path.join(tmp_dir, "hf_home")
                cmd_line = self._create_command_line(
                    example_script,
                    model_name,
                    ipu_config_name,
                    tmp_dir,
                    task=self.TASK_NAME,
                    dataset_config_name=self.DATASET_CONFIG_NAME,
                    do_eval=self.EVAL_IS_SUPPORTED,
                    lr=self.LEARNING_RATE,
                    train_batch_size=self.TRAIN_BATCH_SIZE,
                    eval_batch_size=self.EVAL_BATCH_SIZE,
                    num_epochs=self.NUM_EPOCHS,
                    inference_device_iterations=self.INFERENCE_DEVICE_ITERATIONS,
                    gradient_accumulation_steps=self.GRADIENT_ACCUMULATION_STEPS,
                    extra_command_line_arguments=self.EXTRA_COMMAND_LINE_ARGUMENTS,
                )
                print()
                print("#### Running command line... ####")
                joined_cmd_line = " ".join(cmd_line)
                print(joined_cmd_line)
                print()
                p = subprocess.run(joined_cmd_line, shell=True)
                self.assertEqual(p.returncode, 0)

                if self.EVAL_IS_SUPPORTED:
                    with open(Path(tmp_dir) / "all_results.json") as fp:
                        results = json.load(fp)
                    threshold_overrides = {}
                    if isinstance(self.EVAL_SCORE_THRESHOLD_OVERRIDES, dict):
                        threshold_overrides = self.EVAL_SCORE_THRESHOLD_OVERRIDES
                    threshold = threshold_overrides.get(model_name, self.EVAL_SCORE_THRESHOLD)
                    if self.EVAL_SCORE_GREATER_IS_BETTER:
                        self.assertGreaterEqual(float(results[self.SCORE_NAME]), threshold)
                    else:
                        self.assertLessEqual(float(results[self.SCORE_NAME]), threshold)

        return test


class ExampleTesterBase(TestCase):
    """
    Base example tester class.

    Attributes:
        EXAMPLE_DIR (`str` or `os.Pathlike`): the directory containing the examples.
        EXAMPLE_NAME (`str`): the name of the example script without the file extension, e.g. run_qa, run_glue, etc.
        TASK_NAME (`str`): the name of the dataset to use.
        EVAL_IS_SUPPORTED (`bool`): whether evaluation is currently supported on IPUs.
            If True, the example will run evaluation, otherwise it will be skipped.
        EVAL_SCORE_THRESHOLD (`float`): the score threshold from which training is assumed to have worked.
        EVAL_SCORE_THRESHOLD_OVERRIDES (`dict`): per-model score threshold overrides
        SCORE_NAME (`str`): the name of the metric to use for checking that the example ran successfully.
        DATASET_PARAMETER_NAME (`str`): the argument name to use for the dataset parameter.
            Most of the time it will be "dataset_name", but for some tasks on a benchmark it might be something else.
        TRAIN_BATCH_SIZE (`int`): the batch size to give to the example script for training.
        EVAL_BATCH_SIZE (`int`): the batch size to give to the example script for evaluation.
        INFERENCE_DEVICE_ITERATIONS (`int`): the number of device iterations to use for evaluation.
        GRADIENT_ACCUMULATION_STEPS (`int`): the number of gradient accumulation to use during training.
        DATALOADER_DROP_LAST (`bool`): whether to drop the last batch if it is a remainder batch.
    """

    EXAMPLE_DIR = Path(os.path.dirname(__file__)).parent / "examples"
    EXAMPLE_NAME = None
    TASK_NAME = None
    DATASET_CONFIG_NAME = None
    EVAL_IS_SUPPORTED = True
    EVAL_SCORE_THRESHOLD = 0.75
    EVAL_SCORE_THRESHOLD_OVERRIDES = None
    EVAL_SCORE_GREATER_IS_BETTER = True
    SCORE_NAME = "eval_accuracy"
    DATASET_PARAMETER_NAME = "dataset_name"
    NUM_EPOCHS = 1
    LEARNING_RATE = 1e-4
    TRAIN_BATCH_SIZE = 2
    EVAL_BATCH_SIZE = 2
    INFERENCE_DEVICE_ITERATIONS = 4
    GRADIENT_ACCUMULATION_STEPS = 64
    EXTRA_COMMAND_LINE_ARGUMENTS = None
    POD_TYPE = "pod8"

    def setUp(self):
        self._create_venv()

    def tearDown(self):
        self._remove_venv()

    def _create_command_line(
        self,
        script: str,
        model_name: str,
        ipu_config_name: str,
        output_dir: str,
        task: Optional[str] = None,
        dataset_config_name: Optional[str] = None,
        do_eval: bool = True,
        lr: float = 1e-4,
        train_batch_size: int = 2,
        eval_batch_size: int = 2,
        num_epochs: int = 2,
        inference_device_iterations: int = 4,
        gradient_accumulation_steps: int = 64,
        extra_command_line_arguments: Optional[List[str]] = None,
    ) -> List[str]:
        do_eval_option = "--do_eval" if do_eval else " "
        task_option = f"--{self.DATASET_PARAMETER_NAME} {task}" if task else " "
        ipu_config_overrides = ",".join(
            [
                "executable_cache_dir=disabled",
                "device_iterations=1",
                f"inference_device_iterations={inference_device_iterations}",
                f"gradient_accumulation_steps={gradient_accumulation_steps}",
            ]
        )

        cmd_line = [
            f"{self.VENV_DIR.name}/bin/python" if self.venv_was_created else "python",
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
            f"--ipu_config_overrides {ipu_config_overrides}",
            f" --num_train_epochs {num_epochs}",
            "--dataloader_num_workers 16",
            "--pad_on_batch_axis",
            "--save_steps -1",
            "--save_total_limit 1",
            "--report_to none",
            f"--pod_type {self.POD_TYPE}",
        ]
        if dataset_config_name is not None:
            cmd_line.append(f"--dataset_config_name {dataset_config_name}")

        if extra_command_line_arguments is not None:
            cmd_line += extra_command_line_arguments

        pattern = re.compile(r"([\"\'].+?[\"\'])|\s")
        return [x for y in cmd_line for x in re.split(pattern, y) if x]

    @property
    def venv_was_created(self):
        return os.path.isdir(self.VENV_DIR.name)

    def _create_venv(self):
        """
        Creates the virtual environment for the example.
        """
        self.VENV_DIR = TemporaryDirectory(prefix="venv_")
        cmd_line = f"python -m venv {self.VENV_DIR.name}".split()
        p = subprocess.run(cmd_line)
        self.assertEqual(p.returncode, 0)

    def _remove_venv(self):
        """
        Creates the virtual environment for the example.
        """
        if self.venv_was_created:
            self.VENV_DIR.cleanup()

    def _get_poptorch_wheel_path(self, sdk_path: Optional[str] = None) -> str:
        """
        Retrieves the path for the poptorch wheel.
        """
        if sdk_path is None:
            sdk_path = os.environ["SDK_PATH"]
        paths = glob.glob(f"{sdk_path}/poptorch-*.whl")
        if len(paths) == 0:
            raise FileNotFoundError(f"Could not find poptorch wheel at {sdk_path}")
        if len(paths) > 1:
            raise RuntimeError(f"Multiple poptorch wheels were found at {sdk_path}")
        return paths[0]

    def _get_enable_path(self, library_name: str, sdk_path: Optional[str] = None) -> str:
        """
        Retrieves the path for the "enable" scripts for either poplar or popart.
        """
        if library_name not in ["poplar", "popart"]:
            raise ValueError(
                f'The library name must either be "poplar" or "popart" but "{library_name}" was provided here.'
            )
        if sdk_path is None:
            sdk_path = os.environ["SDK_PATH"]
        paths = glob.glob(f"{sdk_path}/{library_name}*/enable.sh")
        if len(paths) == 0:
            raise FileNotFoundError(f"Could not find {library_name} enable script at {sdk_path}")
        if len(paths) > 1:
            raise RuntimeError(f"Multiple {library_name} enable scripts were found at {sdk_path}")
        return paths[0]

    def _get_poplar_enable_path(self, sdk_path: Optional[str] = None):
        return self._get_enable_path("poplar", sdk_path=sdk_path)

    def _get_popart_enable_path(self, sdk_path: Optional[str] = None):
        return self._get_enable_path("popart", sdk_path=sdk_path)

    def _install_requirements(self, requirements_filename: Union[str, os.PathLike]):
        """
        Installs the necessary requirements to run the example if the provided file exists, otherwise does nothing.
        """
        pip_name = f"{self.VENV_DIR.name}/bin/pip" if self.venv_was_created else "pip"

        # Update pip
        cmd_line = f"{pip_name} install --upgrade pip".split()
        p = subprocess.run(cmd_line)
        self.assertEqual(p.returncode, 0)

        # Install SDK
        cmd_line = f"{pip_name} install .[testing] {self._get_poptorch_wheel_path()}".split()
        with FileLock("install_optimum_graphcore.lock"):
            p = subprocess.run(cmd_line)
            self.assertEqual(p.returncode, 0)

        # Install requirements
        if not Path(requirements_filename).exists():
            return
        cmd_line = f"{pip_name} install -r {requirements_filename}".split()
        p = subprocess.run(cmd_line)
        self.assertEqual(p.returncode, 0)

    def _cleanup_dataset_cache(self):
        """
        Cleans up the dataset cache to free up space for other tests.
        """
        cmd_line = ["rm" "-r", "/nethome/michaelb/.cache/huggingface/datasets"]
        p = subprocess.run(cmd_line)
        self.assertEqual(p.returncode, 0)


class TextClassificationExampleTester(ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_glue"):
    TASK_NAME = "sst2"
    DATASET_PARAMETER_NAME = "task_name"
    INFERENCE_DEVICE_ITERATIONS = 5


class TokenClassificationExampleTester(ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_ner"):
    TASK_NAME = "conll2003"
    TRAIN_BATCH_SIZE = 1
    EVAL_BATCH_SIZE = 1


class MultipleChoiceExampleTester(ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_swag"):
    # Using a small gradient accumulation steps value because input data is repated for the multiple choice task.
    TRAIN_BATCH_SIZE = 1
    EVAL_BATCH_SIZE = 1
    EVAL_SCORE_THRESHOLD_OVERRIDES = {"distilbert-base-uncased": 0.645, "Graphcore/groupbert-base-uncased": 0.66}


class QuestionAnsweringExampleTester(ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_qa"):
    TASK_NAME = "squad"
    SCORE_NAME = "eval_f1"


class SummarizationExampleTester(ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_summarization"):
    TASK_NAME = "cnn_dailymail"
    DATASET_CONFIG = "3.0.0"
    TRAIN_BATCH_SIZE = 1
    EVAL_BATCH_SIZE = 1
    EVAL_IS_SUPPORTED = False
    EVAL_SCORE_THRESHOLD = 30
    SCORE_NAME = "eval_rougeLsum"
    INFERENCE_DEVICE_ITERATIONS = 6
    EXTRA_COMMAND_LINE_ARGUMENTS = [
        "--dataset_config 3.0.0",
        "--prediction_loss_only",
        "--pad_to_max_length",
        "--max_target_length 200",
        "--max_source_length 1024",
    ]

    def _create_command_line(
        self,
        script: str,
        model_name: str,
        ipu_config_name: str,
        output_dir: str,
        task: Optional[str] = None,
        dataset_config_name: Optional[str] = None,
        do_eval: bool = True,
        lr: float = 1e-4,
        train_batch_size: int = 1,
        eval_batch_size: int = 1,
        num_epochs: int = 2,
        inference_device_iterations: int = 6,
        gradient_accumulation_steps: int = 64,
        extra_command_line_arguments: Optional[List[str]] = None,
    ) -> List[str]:
        if extra_command_line_arguments is None:
            extra_command_line_arguments = []
        if "t5" in model_name:
            extra_command_line_arguments.append("--source_prefix 'summarize: '")
        return super()._create_command_line(
            script,
            model_name,
            ipu_config_name,
            output_dir,
            task=task,
            dataset_config_name=dataset_config_name,
            do_eval=do_eval,
            lr=lr,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_epochs=num_epochs,
            inference_device_iterations=inference_device_iterations,
            gradient_accumulation_steps=gradient_accumulation_steps,
            extra_command_line_arguments=extra_command_line_arguments,
        )


class TranslationExampleTester(ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_translation"):
    TASK_NAME = "wmt16"
    TRAIN_BATCH_SIZE = 1
    EVAL_BATCH_SIZE = 1
    EVAL_IS_SUPPORTED = False
    EVAL_SCORE_THRESHOLD = 22
    SCORE_NAME = "eval_bleu"
    INFERENCE_DEVICE_ITERATIONS = 6
    EXTRA_COMMAND_LINE_ARGUMENTS = [
        "--dataset_config ro-en",
        "--source_lang ro",
        "--target_lang en",
        "--pad_to_max_length",
        "--max_source_length 512",
        "--max_target_length 512",
        "--prediction_loss_only",
    ]

    def _create_command_line(
        self,
        script: str,
        model_name: str,
        ipu_config_name: str,
        output_dir: str,
        task: Optional[str] = None,
        dataset_config_name: Optional[str] = None,
        do_eval: bool = True,
        lr: float = 1e-4,
        train_batch_size: int = 1,
        eval_batch_size: int = 1,
        num_epochs: int = 2,
        inference_device_iterations: int = 6,
        gradient_accumulation_steps: int = 64,
        extra_command_line_arguments: Optional[List[str]] = None,
    ) -> List[str]:
        if extra_command_line_arguments is None:
            extra_command_line_arguments = []
        if "t5" in model_name:
            extra_command_line_arguments.append("--source_prefix 'translate English to Romanian: '")
        return super()._create_command_line(
            script,
            model_name,
            ipu_config_name,
            output_dir,
            task=task,
            dataset_config_name=dataset_config_name,
            do_eval=do_eval,
            lr=lr,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_epochs=num_epochs,
            inference_device_iterations=inference_device_iterations,
            gradient_accumulation_steps=gradient_accumulation_steps,
            extra_command_line_arguments=extra_command_line_arguments,
        )


class ImageClassificationExampleTester(
    ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_image_classification"
):
    TASK_NAME = "cifar10"
    NUM_EPOCHS = 2
    EXTRA_COMMAND_LINE_ARGUMENTS = [
        "--remove_unused_columns false",
        "--dataloader_drop_last true",
        "--ignore_mismatched_sizes",
    ]


class AudioClassificationExampleTester(
    ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_audio_classification"
):
    TASK_NAME = "superb"
    DATASET_CONFIG_NAME = "ks"
    GRADIENT_ACCUMULATION_STEPS = 16
    NUM_EPOCHS = 3
    EXTRA_COMMAND_LINE_ARGUMENTS = ["--max_length_seconds 1", "--attention_mask False"]
    LEARNING_RATE = 3e-5


class SpeechRecognitionExampleTester(
    ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_speech_recognition_ctc"
):
    TASK_NAME = "common_voice"
    DATASET_CONFIG_NAME = "tr"
    TRAIN_BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 8
    EVAL_BATCH_SIZE = 1
    NUM_EPOCHS = 15
    # Here we are evaluating against the loss because it can take a long time to have wer < 1.0
    SCORE_NAME = "eval_loss"
    EVAL_SCORE_THRESHOLD = 4
    EVAL_SCORE_GREATER_IS_BETTER = False
    EXTRA_COMMAND_LINE_ARGUMENTS = [
        "--learning_rate 3e-4",
        "--warmup_steps 400",
        "--mask_time_prob 0.0",
        "--layerdrop 0.0",
        "--freeze_feature_encoder",
        "--text_column_name sentence",
        "--length_column_name input_length",
        '--chars_to_ignore , ? . ! - \\; \\: \\" “ % ‘ ” � ',
    ]
