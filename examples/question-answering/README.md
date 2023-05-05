<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

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

# SQuAD

Based on the script [`run_qa.py`](https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/run_qa.py).

**Note:** This script only works with models that have a fast tokenizer (backed by the 🤗 Tokenizers library) as it
uses special features of those tokenizers. You can check if your favorite model has a fast tokenizer in
[this table](https://huggingface.co/transformers/index.html#supported-frameworks).

`run_qa.py` allows you to fine-tune any model from our [hub](https://huggingface.co/models) (as long as its architecture has a `ForQuestionAnswering` version in the transformers library, and a `Pipelined` version available in optimum) on the SQUAD dataset or another question-answering dataset of the `datasets` library or your own csv/jsonlines files as long as they are structured the same way as SQUAD. You might need to tweak the data processing inside the script if your data is structured differently.

Note that if your dataset contains samples with no possible answers (like SQUAD version 2), you need to pass along the flag `--version_2_with_negative`.

## Fine-tuning BERT on SQuAD1.0

This example code fine-tunes BERT base on the SQuAD1.0 dataset.
<!-- Add execution time once it is available.
It runs in 4 min (with BERT-base) or 68 min (with BERT-large)
on a single tesla V100 16GB.
-->

```bash
python examples/question-answering/run_qa.py \
  --model_name_or_path Graphcore/bert-base-uncased \
  --ipu_config_name Graphcore/bert-base-ipu \
  --dataset_name squad \
  --do_train \
  --do_eval \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 16 \
  --n_ipu 16 \
  --learning_rate 7e-5 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --seed 42 \
  --lr_scheduler_type linear \
  --loss_scaling 64 \
  --weight_decay 0.01 \
  --warmup_ratio 0.1 \
  --logging_steps 1 \
  --save_steps -1 \
  --dataloader_num_workers 64 \
  --pad_on_batch_axis \
  --output_dir ./output/squad_bert_base \
  --overwrite_output_dir
```

Training with the previously defined hyper-parameters yields the following results:
```bash
f1 = 88.72
exact_match = 81.59
```

# Visual question answering

`run_vqa.py` allows you to fine-tune the LXMERT model for visual question answering datasets such as VQA v2 and GQA, which are available on our [hub](https://huggingface.co/datasets).

## Fine-tuning LXMERT on GQA

```bash
python examples/question-answering/run_vqa.py \
  --model_name_or_path unc-nlp/lxmert-base-uncased \
  --ipu_config_name Graphcore/lxmert-base-ipu \
  --dataset_name Graphcore/gqa-lxmert \
  --do_train \
  --do_eval \
  --max_seq_length 512 \
  --per_device_train_batch_size 1 \
  --num_train_epochs 4 \
  --dataloader_num_workers 64 \
  --logging_steps 5 \
  --learning_rate 1e-5 \
  --lr_scheduler_type linear \
  --loss_scaling 16384 \
  --weight_decay 0.01 \
  --warmup_ratio 0.1 \
  --output_dir /tmp/gqa/ \
  --dataloader_drop_last \
  --replace_qa_head \
  --n_ipu 16
```
