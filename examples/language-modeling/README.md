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

Fine-tuning (or training from scratch) the library models for language modeling on a text dataset for GPT, GPT-2,
ALBERT, BERT, DistilBERT, RoBERTa, XLNet... GPT and GPT-2 are trained or fine-tuned using a causal language modeling
(CLM) loss while ALBERT, BERT, DistilBERT and RoBERTa are trained or fine-tuned using a masked language modeling (MLM)
loss. XLNet uses permutation language modeling (PLM), you can find more information about the differences between those
objectives in our [model summary](https://huggingface.co/transformers/model_summary.html).

The following examples, will run on datasets hosted on our [hub](https://huggingface.co/datasets) or with your own
text files for training and validation. We give examples of both below.

## BERT pretraining

The following example pretrains BERT on English Wikipedia. The model is trained on two tasks:

- Masked Language Modeling (MLM)
- Next Sentence Prediction (NSP)

You can train BERT on any dataset with `run_pretraining` as long as the dataset contains a column `next_sentence_label`.

```bash
python run_pretraining.py \
  --config_name bert-base-uncased \
  --tokenizer_name bert-base-uncased \
  --ipu_config_name . \
  --dataset_name Graphcore/wikipedia-bert-128 \
  --do_train \
  --do_eval \
  --output_dir /tmp/test-pretraining
```

This uses the HuggingFace `IPUTrainer` for training, designed to perform training on GraphCore IPUs.

### Creating a model on the fly

When training a model from scratch, configuration values may be overridden with the help of `--config_overrides`:


```bash
python run_pretraining.py --config_overrides="hidden_size=1024,num_attention_heads=16,num_hidden_layers=24" [...]
```

## RoBERTa and masked language modeling

The following example fine-tunes RoBERTa-base on WikiText-2. We're using the raw WikiText-2. Note that some IPU configurations are overridden.

```bash
python run_mlm.py \
    --model_name_or_path roberta-base  \
    --ipu_config_name Graphcore/roberta-base-ipu \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_train \
    --do_eval \
    --num_train_epochs 5 \
    --output_dir /tmp/mlm_output \
    --ipu_config_overrides="optimizer_state_offchip=true,inference_device_iterations=5"
```

To fine-tune RoBERTa-large on WikiText-2, we need to override a different set of IPU configurations.

```bash
python run_mlm.py \
    --model_name_or_path roberta-large  \
    --ipu_config_name Graphcore/roberta-large-ipu \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_train \
    --do_eval \
    --num_train_epochs 5 \
    --output_dir /tmp/mlm_output \
    --ipu_config_overrides="embedding_serialization_factor=5,inference_device_iterations=5,matmul_proportion=[0.08 0.2 0.25 0.25]"
```