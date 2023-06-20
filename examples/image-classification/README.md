<!---
Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Image classification examples

This directory contains a script that showcases how to fine-tune any model supported by the [`AutoModelForImageClassification` API](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoModelForImageClassification) using PyTorch. Currently, ViT and ConvNeXT are supported. They can be used to fine-tune models on both datasets from 🤗 `datasets` as well as on [your own custom data](#using-your-own-data).

## Using datasets from 🤗 `datasets`

Here we show how to fine-tune a `ViT` on the [cifar10](https://huggingface.co/datasets/cifar10) dataset.

```
python examples/image-classification/run_image_classification.py \
    --dataset_name cifar10 \
    --output_dir ./cifar10_outputs/ \
    --overwrite_output_dir \
    --model_name_or_path google/vit-base-patch16-224-in21k \
    --ipu_config_name Graphcore/vit-base-ipu \
    --n_ipu 16 \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --weight_decay 0.0 \
    --warmup_ratio 0.25 \
    --loss_scaling 1.0 \
    --num_train_epochs 100 \
    --per_device_train_batch_size 17 \
    --per_device_eval_batch_size 17 \
    --gradient_accumulation_steps 128 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_total_limit 3 \
    --dataloader_mode async_rebatched \
    --dataloader_num_workers 200 \
    --dataloader_drop_last \
    --seed 1337
```

Here we show how to fine-tune a `ViT` on the [beans](https://huggingface.co/datasets/beans) dataset.

👀 See the results here: [nateraw/vit-base-beans](https://huggingface.co/nateraw/vit-base-beans).

```bash
python examples/image-classification/run_image_classification.py \
    --dataset_name beans \
    --output_dir ./beans_outputs/ \
    --model_name_or_path google/vit-base-patch16-224-in21k \
    --ipu_config_name Graphcore/vit-base-ipu \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --learning_rate 3e-4 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --n_ipu 16 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_total_limit 3 \
    --dataloader_num_workers 8 \
    --dataloader_drop_last \
    --seed 1337
```

Here we show how to fine-tune a `ViT` on the [cats_vs_dogs](https://huggingface.co/datasets/cats_vs_dogs) dataset.

👀 See the results here: [nateraw/vit-base-cats-vs-dogs](https://huggingface.co/nateraw/vit-base-cats-vs-dogs).

```bash
python examples/image-classification/run_image_classification.py \
    --dataset_name cats_vs_dogs \
    --output_dir ./cats_vs_dogs_outputs/ \
    --model_name_or_path google/vit-base-patch16-224-in21k \
    --ipu_config_name Graphcore/vit-base-ipu \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --push_to_hub \
    --push_to_hub_model_id vit-base-cats-vs-dogs \
    --learning_rate 2e-4 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 15 \
    --per_device_eval_batch_size 15 \
    --n_ipu 16 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --seed 1337
```

## Using your own data

To use your own dataset, the training script expects the following directory structure:

```bash
root/dog/xxx.png
root/dog/xxy.png
root/dog/[...]/xxz.png

root/cat/123.png
root/cat/nsdf3.png
root/cat/[...]/asd932_.png
```

Once you've prepared your dataset, you can can run the script like this:

```bash
python examples/image-classification/run_image_classification.py \
    --ipu_config_name Graphcore/vit-base-ipu \
    --train_dir <path-to-train-root> \
    --train_val_split 0.1 \
    --output_dir ./outputs/ \
    --do_train \
    --do_eval \
    --num_train_epochs 3 \
    --learning_rate 3e-4 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --n_ipu 16 \
    --dataloader_num_workers 8 \
    --dataloader_drop_last \
    --seed 1337 \
    --remove_unused_columns False
```

### 💡 The above will split the train dir into training and evaluation sets
  - To control the split amount, use the `--train_val_split` flag.
  - To provide your own validation split in its own directory, you can pass the `--validation_dir <path-to-val-root>` flag.

## Sharing your model on 🤗 Hub

0. If you haven't already, [sign up](https://huggingface.co/join) for a 🤗 account

1. Make sure you have `git-lfs` installed and git set up.

```bash
$ apt install git-lfs
$ git config --global user.email "you@example.com"
$ git config --global user.name "Your Name"
```

2. Log in with your HuggingFace account credentials using `huggingface-cli`

```bash
$ huggingface-cli login
# ...follow the prompts
```

3. When running the script, pass the following arguments:

```bash
python examples/image-classification/run_image_classification.py \
    --push_to_hub \
    --push_to_hub_model_id <name-your-model> \
    ...
```
