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

# Audio classification examples

The following examples showcase how to fine-tune `HuBERT` for audio classification using PyTorch.

Speech recognition models that have been pretrained in unsupervised fashion on audio data alone, 
*e.g.* [Wav2Vec2](https://huggingface.co/transformers/master/model_doc/wav2vec2.html), 
[HuBERT](https://huggingface.co/transformers/master/model_doc/hubert.html), 
[XLSR-Wav2Vec2](https://huggingface.co/transformers/master/model_doc/xlsr_wav2vec2.html), have shown to require only 
very little annotated data to yield good performance on speech classification datasets.

## SUPERB Dataset

The following command shows how to fine-tune [hubert-base-ls960](https://huggingface.co/facebook/hubert-base-ls960) on the 🗣️ [Keyword Spotting subset](https://huggingface.co/datasets/superb#ks) of the SUPERB dataset.

```bash
python examples/audio-classification/run_audio_classification.py \
    --model_name_or_path facebook/hubert-base-ls960 \
    --ipu_config_name Graphcore/hubert-base-ipu \
    --dataset_name superb \
    --dataset_config_name ks \
    --output_dir hubert-base-superb-ks \
    --overwrite_output_dir \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --learning_rate 3e-5 \
    --max_length_seconds 1 \
    --attention_mask False \
    --warmup_ratio 0.1 \
    --lr_schedule linear \
    --weight_decay 0.01 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --per_device_eval_batch_size 8 \
    --n_ipu 16 \
    --dataloader_num_workers 64 \
    --dataloader_drop_last True \
    --logging_strategy steps \
    --logging_steps 10 \
    --seed 0
```

## Common Language

The following command shows how to fine-tune [hubert-base-ls960](https://huggingface.co/facebook/hubert-base-ls960) for 🌎 **Language Identification** on the [CommonLanguage dataset](https://huggingface.co/datasets/anton-l/common_language).

```bash
python examples/audio-classification/run_audio_classification.py \
    --model_name_or_path facebook/hubert-base-ls960 \
    --ipu_config_name Graphcore/hubert-base-ipu \
    --dataset_name common_language \
    --audio_column_name audio \
    --label_column_name language \
    --output_dir /tmp/hubert-base-common-language \
    --overwrite_output_dir \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --max_length_seconds 13 \
    --attention_mask False \
    --learning_rate 1e-4 \
    --warmup_ratio 0.25 \
    --lr_schedule linear \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --per_device_eval_batch_size 4 \
    --dataloader_num_workers 64 \
    --dataloader_drop_last True \
    --n_ipu 16 \
    --logging_strategy steps \
    --logging_steps 10 \
    --seed 0
```


## Sharing your model on 🤗 Hub

0. If you haven't already, [sign up](https://huggingface.co/join) for a 🤗 account

1. Make sure you have `git-lfs` installed and git set up.

```bash
$ apt install git-lfs
```

2. Log in with your HuggingFace account credentials using `huggingface-cli`

```bash
$ huggingface-cli login
# ...follow the prompts
```

3. When running the script, pass the following arguments:

```bash
python run_audio_classification.py \
    --push_to_hub \
    --hub_model_id <username/model_id> \
    ...
```
