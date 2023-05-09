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

# Language model training

Fine-tuning (or training from scratch) the library models for language modeling on a text dataset for GPT-2,
BERT, RoBERTa... GPT-2 is trained or fine-tuned using a causal language modeling
(CLM) loss while ALBERT, DistilBERT and RoBERTa are trained or fine-tuned using a masked language modeling (MLM)
loss. BERT model is trained using a combination of MLM and NSP (next sentence prediction).
You can find more information about the differences between those objectives in our [model summary](https://huggingface.co/transformers/model_summary.html).

The following examples, will run on datasets hosted on our [hub](https://huggingface.co/datasets) or with your own
text files for training and validation. We give examples of both below.

## BERT pre-training

The following example pre-trains BERT-base and -large on English Wikipedia *from scratch*. This uses the HuggingFace `IPUTrainer` for training, designed to perform training on GraphCore IPUs. The model is trained on two tasks:

- Masked Language Modeling (MLM)
- Next Sentence Prediction (NSP)

You can train BERT on any dataset with `run_pretraining` as long as the dataset contains a column `next_sentence_label`.

BERT Pre-training is done in two phases - the first is with sequence length 128 for 10500 steps, and the second is with sequence length 512 for 2038 steps.

### GroupBERT-base

Phase 1:
```bash
python examples/language-modeling/run_pretraining.py \
  --model_type groupbert \
  --tokenizer_name bert-base-uncased \
  --ipu_config_name Graphcore/bert-base-ipu \
  --dataset_name Graphcore/wikipedia-bert-128 \
  --do_train \
  --logging_steps 5 \
  --max_seq_length 128 \
  --max_steps 10500 \
  --is_already_preprocessed \
  --dataloader_num_workers 64 \
  --dataloader_mode async_rebatched \
  --lamb \
  --per_device_train_batch_size 10 \
  --gradient_accumulation_steps 1640 \
  --n_ipu 16 \
  --learning_rate 0.012 \
  --loss_scaling 16384 \
  --weight_decay 0.01 \
  --warmup_ratio 0.14 \
  --groupbert_schedule \
  --config_overrides "hidden_dropout_prob=0.0,attention_probs_dropout_prob=0.0,layer_norm_eps=0.001" \
  --ipu_config_overrides "device_iterations=1,matmul_proportion=0.22,layers_per_ipu=[1 3 4 4]" \
  --output_dir output-pretrain-groupbert-base-phase1
```

Phase 2:
```bash
examples/language-modeling/run_pretraining.py \
  --model_type groupbert \
  --tokenizer_name bert-base-uncased \
  --model_name_or_path ./output-pretrain-groupbert-base-phase1 \
  --ipu_config_name Graphcore/bert-base-ipu \
  --dataset_name Graphcore/wikipedia-bert-512 \
  --do_train \
  --logging_steps 5 \
  --max_seq_length 512 \
  --max_steps 2038 \
  --is_already_preprocessed \
  --dataloader_num_workers 64 \
  --dataloader_mode async_rebatched \
  --lamb \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 2048 \
  --n_ipu 16 \
  --learning_rate 0.01 \
  --loss_scaling 128.0 \
  --weight_decay 0.01 \
  --warmup_ratio 0.13 \
  --groupbert_schedule \
  --config_overrides "hidden_dropout_prob=0.0,attention_probs_dropout_prob=0.0,layer_norm_eps=0.001" \
  --ipu_config_overrides device_iterations=1,embedding_serialization_factor=2,matmul_proportion=0.22 \
  --output_dir output-pretrain-groupbert-base-phase2
```
### BERT-base

Phase 1:
```bash
python examples/language-modeling/run_pretraining.py \
  --config_name bert-base-uncased \
  --tokenizer_name bert-base-uncased \
  --ipu_config_name Graphcore/bert-base-ipu \
  --dataset_name Graphcore/wikipedia-bert-128 \
  --do_train \
  --logging_steps 5 \
  --max_seq_length 128 \
  --max_steps 10500 \
  --is_already_preprocessed \
  --dataloader_num_workers 64 \
  --dataloader_mode async_rebatched \
  --lamb \
  --lamb_no_bias_correction \
  --per_device_train_batch_size 32 \
  --gradient_accumulation_steps 512 \
  --n_ipu 16 \
  --learning_rate 0.006 \
  --lr_scheduler_type linear \
  --loss_scaling 16384 \
  --weight_decay 0.01 \
  --warmup_ratio 0.28 \
  --config_overrides "layer_norm_eps=0.001" \
  --ipu_config_overrides "device_iterations=1" \
  --output_dir output-pretrain-bert-base-phase1
```

Phase 2:
```bash
python examples/language-modeling/run_pretraining.py \
  --config_name bert-base-uncased \
  --tokenizer_name bert-base-uncased \
  --model_name_or_path ./output-pretrain-bert-base-phase1 \
  --ipu_config_name Graphcore/bert-base-ipu \
  --dataset_name Graphcore/wikipedia-bert-512 \
  --do_train \
  --logging_steps 5 \
  --max_seq_length 512 \
  --max_steps 2038 \
  --is_already_preprocessed \
  --dataloader_num_workers 128 \
  --dataloader_mode async_rebatched \
  --lamb \
  --lamb_no_bias_correction \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 512 \
  --n_ipu 16 \
  --learning_rate 0.002828 \
  --lr_scheduler_type linear \
  --loss_scaling 128.0 \
  --weight_decay 0.01 \
  --warmup_ratio 0.128 \
  --config_overrides "layer_norm_eps=0.001" \
  --ipu_config_overrides "device_iterations=1,embedding_serialization_factor=2,matmul_proportion=0.22" \
  --output_dir output-pretrain-bert-base-phase2
```

### BERT-large

Phase 1:
```bash
python examples/language-modeling/run_pretraining.py \
  --config_name bert-large-uncased \
  --tokenizer_name bert-large-uncased \
  --ipu_config_name Graphcore/bert-large-ipu \
  --dataset_name Graphcore/wikipedia-bert-128 \
  --do_train \
  --logging_steps 5 \
  --max_seq_length 128 \
  --max_steps 10500 \
  --is_already_preprocessed \
  --dataloader_num_workers 64 \
  --dataloader_mode async_rebatched \
  --lamb \
  --lamb_no_bias_correction \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2048 \
  --n_ipu 16 \
  --learning_rate 0.006 \
  --lr_scheduler_type linear \
  --loss_scaling 32768 \
  --weight_decay 0.01 \
  --warmup_ratio 0.28 \
  --config_overrides "layer_norm_eps=0.001" \
  --ipu_config_overrides "matmul_proportion=[0.14 0.19 0.19 0.19]" \
  --output_dir output-pretrain-bert-large-phase1
```

Phase 2:
```bash
python examples/language-modeling/run_pretraining.py \
  --config_name bert-large-uncased \
  --tokenizer_name bert-large-uncased \
  --model_name_or_path ./output-pretrain-bert-large-phase1 \
  --ipu_config_name Graphcore/bert-large-ipu \
  --dataset_name Graphcore/wikipedia-bert-512 \
  --do_train \
  --logging_steps 5 \
  --max_seq_length 512 \
  --max_steps 2038 \
  --is_already_preprocessed \
  --dataloader_num_workers 96 \
  --dataloader_mode async_rebatched \
  --lamb \
  --lamb_no_bias_correction \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 2048 \
  --n_ipu 16 \
  --learning_rate 0.002828 \
  --lr_scheduler_type linear \
  --loss_scaling 16384 \
  --weight_decay 0.01 \
  --warmup_ratio 0.128 \
  --config_overrides "layer_norm_eps=0.001" \
  --ipu_config_overrides "matmul_proportion=[0.14 0.19 0.19 0.19]" \
  --output_dir output-pretrain-bert-large-phase2
```

### Creating a model on the fly

When training a model from scratch, configuration values may be overridden with the help of `--config_overrides`:


```bash
python run_pretraining.py --config_overrides="hidden_size=1024,num_attention_heads=16,num_hidden_layers=24" [...]
```

### Employing automatic loss scaling (ALS) for half precision training
	
ALS is an experimental feature in the Poplar SDK which brings stability to training large models in half precision, specially when gradient accumulation and reduction across replicas also happen in half precision. 

NB. This feature expects the `poptorch` training option `accumulationAndReplicationReductionType` to be set to `poptorch.ReductionType.Mean`, and for accumulation by the optimizer to be done in half precision (using `accum_type=torch.float16` when instantiating the optimizer), or else it may lead to unexpected behaviour.

To employ ALS, just add the flag `--auto_loss_scaling` to the command. The loss scaling value specified with `--loss_scaling` will be the initial one before ALS updates it during training — you can set it to `1`.


## RoBERTa/BERT and masked language modeling

The following example fine-tunes RoBERTa-base on WikiText-2. We're using the raw WikiText-2. Note that `inference_device_iterations` is overridden.

```bash
python examples/language-modeling/run_mlm.py \
    --model_name_or_path roberta-base  \
    --ipu_config_name Graphcore/roberta-base-ipu \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --do_train \
    --do_eval \
    --num_train_epochs 5 \
    --n_ipu 16 \
    --output_dir /tmp/mlm_output \
    --ipu_config_overrides="inference_device_iterations=5" \
    --dataloader_drop_last
```

The same can be done with the BERT model by changing the flags: `--model_name_or_path bert-base-uncased --ipu_config_name Graphcore/bert-base-ipu`. 
## GPT2 and causal language modeling

The following example fine-tunes GPT2-small on WikiText-2. We're using the raw WikiText-2. Note that some IPU configurations are overridden.

```bash
python examples/language-modeling/run_clm.py \
    --model_name_or_path gpt2 \
    --ipu_config_name Graphcore/gpt2-small-ipu \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_train \
    --do_eval \
    --num_train_epochs 30 \
    --dataloader_num_workers 64 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 128 \
    --output_dir /tmp/clm_gpt2 \
    --logging_steps 1 \
    --learning_rate 1e-5 \
    --lr_scheduler_type linear \
    --loss_scaling 16384 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --config_overrides="activation_function=gelu" \
    --dataloader_drop_last \
    --n_ipu 16
```

To fine-tune GPT2-medium on WikiText-2, we need to override a different set of IPU configurations. Note that `activation_function` is overridden to `gelu`
instead of using the original `gelu_new`, which does not run efficiently on IPUs.

```bash
python examples/language-modeling/run_clm.py \
    --model_name_or_path gpt2-medium \
    --ipu_config_name Graphcore/gpt2-medium-ipu \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_train \
    --do_eval \
    --num_train_epochs 30 \
    --dataloader_num_workers 64 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 256 \
    --output_dir /tmp/clm_gpt2_medium \
    --logging_steps 1 \
    --learning_rate 1e-5 \
    --lr_scheduler_type linear \
    --loss_scaling 16384 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --config_overrides="activation_function=gelu" \
    --dataloader_drop_last \
    --n_ipu 16

```
