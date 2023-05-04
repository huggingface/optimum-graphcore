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

The following examples show how to pre-train `"base"`- and `"large"`-sized Wav2Vec2 models.


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

## Poplar SDK setup
To check if your Poplar SDK has already been enabled, run:
```bash
 echo $POPLAR_SDK_ENABLED
```

If no path is provided, then follow these steps:
1. Navigate to your Poplar SDK root directory

2. Enable the Poplar SDK with:
```bash
source enable
```

More detailed instructions on setting up your Poplar environment are available in the [Poplar quick start guide](https://docs.graphcore.ai/projects/poplar-quick-start).


## Environment setup
To prepare your environment, follow these steps:

1. Create and activate a Python3 virtual environment:
```bash
python3 -m venv <venv name>
source <venv path>/bin/activate
```

2. Navigate to the Poplar SDK root directory

3. Install the PopTorch (PyTorch) wheel:
```bash
cd <poplar sdk root dir>
pip3 install poptorch...x86_64.whl
```

4. Navigate to this example's root directory

5. Install the Python requirements:
```bash
pip3 install -r requirements.txt
```

6. Install the latest release of the `optimum-graphcore` package as described in [optimum-graphcore/#install](https://github.com/huggingface/optimum-graphcore/#install). For example, to install from source:
```
pip install git+https://github.com/huggingface/optimum-graphcore.git
```

## Demo

In this demo run we pre-train a `"base-sized"` Wav2Vec2 model simply only on the validation
data of [librispeech_asr](https://huggingface.co/datasets/librispeech_asr).

```bash
python run_pretraining.py \
	--model_name_or_path "facebook/wav2vec2-base" \
	--dataset_name "librispeech_asr" \
	--dataset_config_name "clean" \
	--train_split_name "validation" \
	--ipu_config_name "Graphcore/wav2vec2-base-ipu" \
	--output_dir "./wav2vec2-pretrained-demo" \
	--max_duration_in_seconds 15.6 \
	--min_duration_in_seconds 2.0 \
	--do_train \
	--overwrite_output_dir \
	--layerdrop 0.05 \
	--per_device_train_batch_size 1 \
	--dataloader_mode "async_rebatched" \
	--dataloader_num_workers 64 \
	--num_train_epochs 1 \
	--warmup_steps 1000 \
	--weight_decay 0.01 \
	--learning_rate 0.001 \
	--adam_beta1 0.9 \
	--adam_beta2 0.98 \
	--adam_epsilon 1e-04 \
	--max_gumbel_temperature 2.0 \
	--min_gumbel_temperature 0.5 \
	--gumbel_temperature_decay 0.999995 \
	--logging_steps 10 \
	--n_ipu 16
```

## Base

To pre-train `"base-sized"` Wav2Vec2 model, *e.g.* [facebook/wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base)
on 100h of training data from the [librispeech_asr](https://huggingface.co/datasets/librispeech_asr), the following command can be run:

```bash
python run_pretraining.py \
	--model_name_or_path "facebook/wav2vec2-base" \
	--dataset_name "librispeech_asr" \
	--dataset_config_name "clean" \
	--train_split_name "train.100" \
	--ipu_config_name "Graphcore/wav2vec2-base-ipu" \
	--output_dir "./wav2vec2-pretrained-base" \
	--max_duration_in_seconds 15.6 \
	--min_duration_in_seconds 2.0 \
	--do_train \
	--overwrite_output_dir \
	--layerdrop 0.05 \
	--per_device_train_batch_size 1 \
	--dataloader_mode "async_rebatched" \
	--dataloader_num_workers 64 \
	--num_train_epochs 10 \
	--warmup_steps 1000 \
	--weight_decay 0.01 \
	--learning_rate 0.001 \
	--adam_beta1 0.9 \
	--adam_beta2 0.98 \
	--adam_epsilon 1e-04 \
	--max_gumbel_temperature 2.0 \
	--min_gumbel_temperature 0.5 \
	--gumbel_temperature_decay 0.999995 \
	--logging_steps 10 \
	--n_ipu 16
```

If you increase the effective batch size, for example by increasing the `gradient_accumulation_steps`,
it is recommended to increase the `learning_rate` to `0.005` for faster convergence.

## Large

To pre-train `"large-sized"` Wav2Vec2 model, *e.g.* [facebook/wav2vec2-large](https://huggingface.co/facebook/wav2vec2-large)
on 100h of training data from the [librispeech_asr](https://huggingface.co/datasets/librispeech_asr), the following command can be run:

```bash
python run_pretraining.py \
	--model_name_or_path "facebook/wav2vec2-large-960h" \
	--dataset_name "librispeech_asr" \
	--dataset_config_name "clean" \
	--train_split_name "train.100" \
	--ipu_config_name "Graphcore/wav2vec2-large-ipu" \
	--output_dir "./wav2vec2-pretrained-large" \
	--max_duration_in_seconds 15.6 \
	--min_duration_in_seconds 2.0 \
	--do_train \
	--overwrite_output_dir \
	--layerdrop 0.05 \
	--per_device_train_batch_size 1 \
	--dataloader_num_workers 64 \
	--num_train_epochs 10 \
	--warmup_steps 1000 \
	--weight_decay 0.01 \
	--learning_rate 0.001 \
	--adam_beta1 0.9 \
	--adam_beta2 0.98 \
	--adam_epsilon 1e-04 \
	--max_gumbel_temperature 2.0 \
	--min_gumbel_temperature 0.5 \
	--gumbel_temperature_decay 0.999995 \
	--logging_steps 10 \
	--n_ipu 16
```

Similarly to the `"base-sized"` model above, be sure to select optimal `learning_rate` given the effective batch size of your configuration. The effective batch size is defined as `gradient_accumulation_steps * per_device_train_batch_size * replication_factor`. The `replication_factor` is calculated as number of IPUs (`pod_type`) divided by `ipus_per_replica`. See [Graphcore/wav2vec2-large](https://huggingface.co/Graphcore/wav2vec2-large-ipu) for configuration parameters in addition to the command line arguments.
