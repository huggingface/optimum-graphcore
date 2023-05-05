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

## Translation

This directory contains examples for finetuning and evaluating transformers on translation tasks.

### Supported Architectures

- `BartForConditionalGeneration`
- `T5ForConditionalGeneration`

`run_translation.py` is a lightweight examples of how to download and preprocess a dataset from the [🤗 Datasets](https://github.com/huggingface/datasets) library or use your own files (jsonlines or csv), then fine-tune one of the architectures above on it.

For custom datasets in `jsonlines` format please see: https://huggingface.co/docs/datasets/loading_datasets.html#json-files
and you also will find examples of these below.


Here is an example of a translation fine-tuning with a Bart model:

```bash
python examples/translation/run_translation.py \
    --model_name_or_path facebook/bart-base \
    --ipu_config_name Graphcore/bart-base-ipu \
    --do_train \
    --do_eval \
    --source_lang en \
    --target_lang ro \
    --dataset_name wmt16 \
    --dataset_config_name ro-en \
    --output_dir /tmp/tst-translation \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --n_ipu 16 \
    --dataloader_drop_last \
    --pad_to_max_length \
    --logging_steps 1 \
    --overwrite_output_dir
```

Some T5 models require special handling.

T5 models `t5-small`, `t5-base`, `t5-large`, `t5-3b` and `t5-11b` must use an additional argument: `--source_prefix "translate {source_lang} to {target_lang}"`. For example:

```bash
python examples/translation/run_translation.py \
    --model_name_or_path t5-small \
    --ipu_config_name Graphcore/t5-small-ipu \
    --do_train \
    --do_eval \
    --source_lang en \
    --target_lang ro \
    --source_prefix "translate English to Romanian: " \
    --dataset_name wmt16 \
    --dataset_config_name ro-en \
    --output_dir /tmp/tst-translation \
    --per_device_train_batch_size=2 \
    --per_device_eval_batch_size=2 \
    --n_ipu 16 \
    --dataloader_drop_last \
    --max_source_length 512 \
    --max_target_length 512 \
    --pad_to_max_length \
    --logging_steps 1 \
    --overwrite_output_dir
```

If you get a terrible BLEU score, make sure that you didn't forget to use the `--source_prefix` argument.

For the aforementioned group of T5 models it's important to remember that if you switch to a different language pair, make sure to adjust the source and target values in all 3 language-specific command line argument: `--source_lang`, `--target_lang` and `--source_prefix`.

And here is how you would use the translation finetuning on your own files, after adjusting the
values for the arguments `--train_file`, `--validation_file` to match your setup:

```bash
python examples/translation/run_translation.py \
    --model_name_or_path t5-small \
    --ipu_config_name Graphcore/t5-small-ipu \
    --do_train \
    --do_eval \
    --source_lang en \
    --target_lang ro \
    --source_prefix "translate English to Romanian: " \
    --dataset_name wmt16 \
    --dataset_config_name ro-en \
    --train_file path_to_jsonlines_file \
    --validation_file path_to_jsonlines_file \
    --output_dir /tmp/tst-translation \
    --per_device_train_batch_size=2 \
    --per_device_eval_batch_size=2 \
    --n_ipu 16 \
    --dataloader_drop_last \
    --max_source_length 512 \
    --max_target_length 512 \
    --pad_to_max_length \
    --logging_steps 1 \
    --overwrite_output_dir
```

The task of translation supports only custom JSONLINES files, with each line being a dictionary with a key `"translation"` and its value another dictionary whose keys is the language pair. For example:

```json
{ "translation": { "en": "Others have dismissed him as a joke.", "ro": "Alții l-au numit o glumă." } }
{ "translation": { "en": "And some are holding out for an implosion.", "ro": "Iar alții așteaptă implozia." } }
```
Here the languages are Romanian (`ro`) and English (`en`).

If you want to use a pre-processed dataset that leads to high BLEU scores, but for the `en-de` language pair, you can use `--dataset_name stas/wmt14-en-de-pre-processed`, as following:

```bash
python examples/translation/run_translation.py \
    --model_name_or_path t5-small \
    --ipu_config_name Graphcore/t5-small-ipu \
    --do_train \
    --do_eval \
    --source_lang en \
    --target_lang de \
    --source_prefix "translate English to German: " \
    --dataset_name stas/wmt14-en-de-pre-processed \
    --output_dir /tmp/tst-translation \
    --per_device_train_batch_size=2 \
    --per_device_eval_batch_size=2 \
    --n_ipu 16 \
    --dataloader_drop_last \
    --max_source_length 512 \
    --max_target_length 512 \
    --pad_to_max_length \
    --logging_steps 1 \
    --overwrite_output_dir
 ```
