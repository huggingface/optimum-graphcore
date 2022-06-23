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

# Speech Recognition Pre-Training


## Wav2Vec2 Speech Pre-Training

The script [`run_pretraining.py`](./run_pretraining.py) can be used to pre-train a [Wav2Vec2](https://huggingface.co/transformers/model_doc/wav2vec2.html) model from scratch.

In the script [`run_pretraining.py`](./run_pretraining.py), a Wav2Vec2 model is pre-trained on audio data alone using [Wav2Vec2's contrastive loss objective](https://arxiv.org/abs/2006.11477).

The following examples show how to pre-train a `"base"`-sized Wav2Vec2 model.


---
**NOTE 1**

Wav2Vec2's pre-training is known to be quite unstable.
It is advised to do a couple of test runs with a smaller dataset,
*i.e.* `--dataset_config_name clean`, `--dataset_split_name validation`
to find good hyper-parameters for `learning_rate`, `batch_size`, `warmup_steps`,
and the optimizer.

---

---
**NOTE 2**

When training a model on large datasets it is recommended to run the data preprocessing
in a first run in a **non-distributed** mode via `--preprocessing_only` so that
when running the  model in **distributed** mode in a second step the preprocessed data
can easily be loaded on each distributed device.

---

### Demo

In this demo run we pre-train a `"base-sized"` Wav2Vec2 model simply only on the validation
data of [librispeech_asr](https://huggingface.co/datasets/librispeech_asr).

```bash
python run_pretraining.py \
	--model_name_or_path "facebook/wav2vec2-base" \
	--dataset_name "librispeech_asr" \
	--dataset_config_name "clean" \
	--train_split_name "validation" \
	--ipu_config_name "./base_4-ipu_config.json" \
	--output_dir "./wav2vec2-pretrained-demo" \
	--max_duration_in_seconds 20.0 \
	--min_duration_in_seconds 2.0 \
	--do_train \
	--overwrite_output_dir \
	--layerdrop 0.05 \
	--per_device_train_batch_size 1 \
	--dataloader_mode "async_rebatched" \
	--dataloader_num_workers 8 \
	--num_train_epochs 1 \
	--weight_decay 0.01
```

### Base

To pre-train `"base-sized"` Wav2Vec2 model, *e.g.* [facebook/wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base)
on 100h of training data from the [librispeech_asr](https://huggingface.co/datasets/librispeech_asr), the following command can be run:

```bash
python run_pretraining.py \
	--model_name_or_path "facebook/wav2vec2-base" \
	--dataset_name "librispeech_asr" \
	--dataset_config_name "clean" \
	--train_split_name "train.100" \
	--ipu_config_name "./base_4-ipu_config.json" \
	--output_dir "./wav2vec2-pretrained-base" \
	--max_duration_in_seconds 20.0 \
	--min_duration_in_seconds 2.0 \
	--do_train \
	--overwrite_output_dir \
	--layerdrop 0.05 \
	--per_device_train_batch_size 1 \
	--dataloader_mode "async_rebatched" \
	--dataloader_num_workers 8 \
	--num_train_epochs 10 \
	--warmup_steps 1000 \
	--weight_decay 0.01 \
	--learning_rate 0.001 \
	--adam_beta1 0.9 \
	--adam_beta2 0.98 \
	--adam_epsilon 1e-04
```

If you increase the effective `batch_size`, for example by increasing the `gradient_accumulation_steps`,
it is recommended to increase the `learning_rate` to `0.005` for faster convergence.
