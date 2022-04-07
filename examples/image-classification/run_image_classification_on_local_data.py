#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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

import glob
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from timm.data.mixup import Mixup, FastCollateMixup

import transforms

import datasets
import numpy as np
import torch
from PIL import Image
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

import transformers
from optimum.graphcore import IPUConfig, IPUTrainer
from optimum.graphcore import IPUTrainingArguments
from transformers import (
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForImageClassification,
    HfArgumentParser,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version


""" Fine-tuning a Transformers model for image classification"""

logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.15.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/image-classification/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def pil_loader(path: str):
    with open(path, "rb") as f:
        im = Image.open(f)
        return im.convert("RGB")


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(
        default="nateraw/image-folder", metadata={"help": "Name of a dataset from the datasets package"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the training data."})
    validation_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the validation data."})
    train_val_split: Optional[float] = field(
        default=0.15, metadata={"help": "Percent to split off of train for validation."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )

    def __post_init__(self):
        data_files = dict()
        if self.train_dir is not None:
            data_files["train"] = self.train_dir
        if self.validation_dir is not None:
            data_files["val"] = self.validation_dir
        self.data_files = data_files if data_files else None


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="google/vit-base-patch16-224-in21k",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    feature_extractor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."},
    )

    disable_feature_extractor: bool = field(
        default=False,
        metadata={
            "help": "Weather or not to disable the feature extractor."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

@dataclass
class TrainingArguments(IPUTrainingArguments):
    """
    Subclass IPUTrainingArguments to pass extra training-related arguments such as data-augmentation.
    """
    input_size: Optional[int] = field(
        default = 224,
        metadata={
            "help": "Image input size."
        },
    )
    disable_mixup: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Disable the pre-processing Mixup function for data augmentation."
        },
    )
    nb_classes: Optional[float] = field(
        default=1000
    )
    smoothing: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "Label smoothing."
        },
    )
    mixup: Optional[float] = field(
        default=1.0
    )
    cutmix: Optional[float] = field(
        default=1.0
    )
    cutmix_minmax: Optional[float] = field(
        default=None
    )
    mixup_prob: Optional[float] = field(
        default=0.1
    )
    mixup_switch_prob: Optional[float] = field(
        default=0.5
    )
    mixup_mode: Optional[str] = field(
        default='batch'
    )


def collate_fn(examples):
    # pixel_values = torch.stack([example["pixel_values"] for example in examples])
    # labels = torch.tensor([example["labels"] for example in examples])

    pixel_values = torch.stack([example[0] for example in examples])
    labels = torch.tensor([example[1] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


# Implement transforms as a functor instead of a function because the Async Dataloader
# can't handle functions with closures because it uses pickle underneath.
class ApplyTransforms:
    """
    Functor that applies image transforms across a batch.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, example_batch):
        # example_batch["pixel_values"] = [self.transforms(img) for img in example_batch["image"]]
        # TODO: is ApplyTransforms still needed since we now transforms already apply to the image features.
        example_batch = self.transforms(example_batch)
        return example_batch




def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Load the accuracy metric from the datasets package
    metric = datasets.load_metric("accuracy")

    # Define our compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p):
        """Computes accuracy on a batch of predictions"""
        return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

    labels = glob.glob(os.path.join(data_args.data_files.get("train", None), "*/"))
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    config = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
        finetuning_task="image-classification",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    config.smoothing=training_args.smoothing

    # ipu_config = IPUConfig.from_pretrained(
    #     training_args.ipu_config_name if training_args.ipu_config_name else model_args.model_name_or_path,
    #     cache_dir=model_args.cache_dir,
    #     revision=model_args.model_revision,
    #     use_auth_token=True if model_args.use_auth_token else None,
    # )
    ipu_config = IPUConfig.from_json_file("convnext_ipuconfig.json")
    model = AutoModelForImageClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_args.feature_extractor_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Define torchvision transforms to be applied to each image.
    _train_transforms, _val_transforms = transforms.get_transforms(model_args.model_name_or_path, training_args, feature_extractor)

    # Initialize our dataset and prepare it for the 'image-classification' task.
    ds = dict()
    train_data_files = data_args.data_files.get("train", None)
    eval_data_files = data_args.data_files.get("val", None)

    if train_data_files:
        ds["train"] = ImageFolder(train_data_files, transform=ApplyTransforms(_train_transforms))

    if eval_data_files:
        ds["validation"] = ImageFolder(eval_data_files, transform=ApplyTransforms(_val_transforms))

    # If we don't have a validation split, split off a percentage of train as validation.
    data_args.train_val_split = None if "validation" in ds.keys() else data_args.train_val_split
    if isinstance(data_args.train_val_split, float) and data_args.train_val_split > 0.0:
        ds_validation_size = int(len(ds["train"]) * data_args.train_val_split)
        ds_train_size = len(ds["train"]) - ds_validation_size
        ds["train"], ds["validation"] = random_split(
            ds["train"],
            [ds_train_size, ds_validation_size],
            generator=torch.Generator().manual_seed(training_args.seed),
        )

    # Prepare label mappings.
    # We'll include these in the model's config to get human readable labels in the Inference API.

    if training_args.do_train:
        if "train" not in ds:
            raise ValueError("--do_train requires a train dataset")
        if data_args.max_train_samples is not None:
            ds["train"] = ds["train"].shuffle(seed=training_args.seed)[: data_args.max_train_samples]

    if training_args.do_eval:
        if "validation" not in ds:
            raise ValueError("--do_eval requires a validation dataset")
        if data_args.max_eval_samples is not None:
            ds["validation"] = ds["validation"].shuffle(seed=training_args.seed)[: data_args.max_val_samples]

    train_collate_fn = collate_fn
    if (training_args.mixup > 0 or training_args.cutmix > 0. or training_args.cutmix_minmax is not None) and not training_args.disable_mixup:

        logger.info("Training with Mixup")
        mixup_fn = Mixup(
        mixup_alpha=training_args.mixup, cutmix_alpha=training_args.cutmix, cutmix_minmax=training_args.cutmix_minmax,
        prob=training_args.mixup_prob, switch_prob=training_args.mixup_switch_prob, mode=training_args.mixup_mode,
        label_smoothing=training_args.smoothing, num_classes=training_args.nb_classes)

        def mixup_collate_fn(examples):
            pixel_values = torch.stack([example[0] for example in examples])
            labels = torch.tensor([example[1] for example in examples])
            pixel_values, labels = mixup_fn(pixel_values, labels)
            return {"pixel_values": pixel_values, "labels": labels}

        train_collate_fn = mixup_collate_fn
    if model_args.disable_feature_extractor:
        logger.info("Model feature extractor disabled")

    # Initalize our trainer
    trainer = IPUTrainer(
        model=model,
        ipu_config=ipu_config,
        args=training_args,
        train_dataset=ds["train"] if training_args.do_train else None,
        eval_dataset=ds["validation"] if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=feature_extractor if not model_args.disable_feature_extractor else None,
        data_collator=train_collate_fn,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        trainer.data_collator=collate_fn
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Write model card and (optionally) push to hub
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "image-classification",
        "dataset": data_args.dataset_name,
        "tags": ["image-classification"],
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
