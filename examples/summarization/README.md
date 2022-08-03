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

## Summarization

This directory contains examples for finetuning and evaluating transformers on summarization  tasks.

### Supported Architectures

- `BartForConditionalGeneration`
- `T5ForConditionalGeneration`

`run_summarization.py` is a lightweight example of how to download and preprocess a dataset from the [🤗 Datasets](https://github.com/huggingface/datasets) library or use your own files (jsonlines or csv), then fine-tune one of the architectures above on it.

For custom datasets in `jsonlines` format please see: https://huggingface.co/docs/datasets/loading_datasets.html#json-files
and you also will find examples of these below.

Here is an example on a summarization task:
```bash
python examples/summarization/run_summarization.py \
    --model_name_or_path t5-small \
    --ipu_config_name Graphcore/t5-small-ipu \
    --ipu_config_overrides "inference_device_iterations=1,inference_replication_factor=2,sharded_execution_for_inference=True,execute_encoder_on_cpu_for_generation=False" \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --pod_type pod16 \
    --num_train_epochs 2 \
    --max_target_length 200 \
    --logging_steps 1 \
    --learning_rate 1e-4 \
    --lr_scheduler_type constant \
    --max_grad_norm 0.5 \
    --pad_to_max_length \
    --dataloader_drop_last \
    --predict_with_generate \
    --generation_num_beams 2 \
    --output_dir /tmp/t5-summarization \
    --overwrite_output_dir
```

Only T5 models `t5-small`, `t5-base`, `t5-large`, `t5-3b` and `t5-11b` must use an additional argument: `--source_prefix "summarize: "`. To abreviate the training and evaluation you can add the flags: `--max_train_samples 20000 --max_eval_samples 400`.

We used CNN/DailyMail dataset in this example as `t5-small` was trained on it and one can get good scores even when pre-training with a very small sample.

Extreme Summarization (XSum) Dataset is another commonly used dataset for the task of summarization. To use it replace `--dataset_name cnn_dailymail --dataset_config "3.0.0"` with  `--dataset_name xsum`.

And here is how you would use it on your own files, after adjusting the values for the arguments
`--train_file`, `--validation_file`, `--text_column` and `--summary_column` to match your setup:

```bash
python examples/summarization/run_summarization.py \
    --model_name_or_path t5-small \
    --ipu_config_name Graphcore/t5-small-ipu \
    --do_train \
    --do_eval \
    --train_file path_to_csv_or_jsonlines_file \
    --validation_file path_to_csv_or_jsonlines_file \
    --source_prefix "summarize: " \
    --overwrite_output_dir \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=4 \
    --max_target_length 200 \
    --num_train_epochs 2 \
    --pod_type pod16 \
    --learning_rate 1e-4 \
    --lr_scheduler_type constant \
    --max_grad_norm 0.5 \
    --pad_to_max_length \
    --dataloader_drop_last \
    --predict_with_generate \
    --generation_num_beams 2 \
    --output_dir /tmp/t5-summarization \
```

The task of summarization supports custom CSV and JSONLINES formats.

The same tasks can be run with BART models by using arguments `--model_name_or_path facebook/bart-base --ipu_config_name Graphcore/bart-base-ipu` and removing the `--source_prefix` argument. For example, the `cnn_dailymail` summarization:

```
python examples/summarization/run_summarization.py \
    --model_name_or_path facebook/bart-base \
    --ipu_config_name Graphcore/bart-base-ipu \
    --ipu_config_overrides "inference_device_iterations=1,inference_replication_factor=2,sharded_execution_for_inference=True,execute_encoder_on_cpu_for_generation=False,layers_per_ipu=[0 4 4 4]" \
    --do_train True \
    --do_eval True \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --pod_type pod16 \
    --num_train_epochs 2 \
    --logging_steps 1 \
    --learning_rate 1e-4 \
    --lr_scheduler_type constant \
    --max_grad_norm 0.5 \
    --pad_to_max_length \
    --dataloader_drop_last \
    --predict_with_generate \
    --generation_num_beams 2 \
    --output_dir /tmp/bart-summarization \
    --overwrite_output_dir
```

#### Custom CSV Files

If it's a csv file the training and validation files should have a column for the inputs texts and a column for the summaries.

If the csv file has just two columns as in the following example:

```csv
text,summary
"I'm sitting here in a boring room. It's just another rainy Sunday afternoon. I'm wasting my time I got nothing to do. I'm hanging around I'm waiting for you. But nothing ever happens. And I wonder","I'm sitting in a room where I'm waiting for something to happen"
"I see trees so green, red roses too. I see them bloom for me and you. And I think to myself what a wonderful world. I see skies so blue and clouds so white. The bright blessed day, the dark sacred night. And I think to myself what a wonderful world.","I'm a gardener and I'm a big fan of flowers."
"Christmas time is here. Happiness and cheer. Fun for all that children call. Their favorite time of the year. Snowflakes in the air. Carols everywhere. Olden times and ancient rhymes. Of love and dreams to share","It's that time of year again."
```

The first column is assumed to be for `text` and the second is for summary.

If the csv file has multiple columns, you can then specify the names of the columns to use:

```bash
    --text_column text_column_name \
    --summary_column summary_column_name \
```

For example if the columns were:

```csv
id,date,text,summary
```

and you wanted to select only `text` and `summary`, then you'd pass these additional arguments:

```bash
    --text_column text \
    --summary_column summary \
```

#### Custom JSONLINES Files

The second supported format is jsonlines. Here is an example of a jsonlines custom data file.


```json
{"text": "I'm sitting here in a boring room. It's just another rainy Sunday afternoon. I'm wasting my time I got nothing to do. I'm hanging around I'm waiting for you. But nothing ever happens. And I wonder", "summary": "I'm sitting in a room where I'm waiting for something to happen"}
{"text": "I see trees so green, red roses too. I see them bloom for me and you. And I think to myself what a wonderful world. I see skies so blue and clouds so white. The bright blessed day, the dark sacred night. And I think to myself what a wonderful world.", "summary": "I'm a gardener and I'm a big fan of flowers."}
{"text": "Christmas time is here. Happiness and cheer. Fun for all that children call. Their favorite time of the year. Snowflakes in the air. Carols everywhere. Olden times and ancient rhymes. Of love and dreams to share", "summary": "It's that time of year again."}
```

Same as with the CSV files, by default the first value will be used as the text record and the second as the summary record. Therefore you can use any key names for the entries, in this example `text` and `summary` were used.

And as with the CSV files, you can specify which values to select from the file, by explicitly specifying the corresponding key names. In our example this again would be:

```bash
    --text_column text \
    --summary_column summary \
```
