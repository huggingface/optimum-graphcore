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

# Text classification examples

## GLUE tasks

Based on the script [`run_glue.py`](https://github.com/huggingface/transformers/blob/master/examples/pytorch/text-classification/run_glue.py).

Fine-tuning the library models for sequence classification on the GLUE benchmark: [General Language Understanding
Evaluation](https://gluebenchmark.com/). This script can fine-tune any of the models on the [hub](https://huggingface.co/models)
and can also be used for a dataset hosted on our [hub](https://huggingface.co/datasets) or your own data in a csv or a JSON file
(the script might need some tweaks in that case, refer to the comments inside for help).

GLUE is made up of a total of 9 different tasks. Here is how to run the script on one of them:

```bash
export TASK_NAME=mrpc

python run_glue.py \
  --model_name_or_path bert-base-cased \
  --ipu_config_name Graphcore/bert-base \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /tmp/$TASK_NAME/
```

where task name can be one of cola, sst2, mrpc, stsb, qqp, mnli, qnli, rte, wnli.

We get the following results on the dev set of the benchmark with the previous commands (with an exception for MRPC and
WNLI which are tiny and where we used 5 epochs instead of 3). <!-- TODO: perform experiments on the PODs -> Trainings are seeded so you should obtain the same
results with PyTorch 1.6.0 (and close results with different versions), training times are given for information (a
single Titan RTX was used):

| Task  | Metric                       | Result      | Training time |
|-------|------------------------------|-------------|---------------|
| CoLA  | Matthews corr                | 56.53       | 3:17          |
| SST-2 | Accuracy                     | 92.32       | 26:06         |
| MRPC  | F1/Accuracy                  | 88.85/84.07 | 2:21          |
| STS-B | Pearson/Spearman corr.       | 88.64/88.48 | 2:13          |
| QQP   | Accuracy/F1                  | 90.71/87.49 | 2:22:26       |
| MNLI  | Matched acc./Mismatched acc. | 83.91/84.10 | 2:35:23       |
| QNLI  | Accuracy                     | 90.66       | 40:57         |
| RTE   | Accuracy                     | 65.70       | 57            |
| WNLI  | Accuracy                     | 56.34       | 24            |

Some of these results are significantly different from the ones reported on the test set of GLUE benchmark on the
website. For QQP and WNLI, please refer to [FAQ #12](https://gluebenchmark.com/faq) on the website.

-->
The following example fine-tunes BERT on the `imdb` dataset hosted on our [hub](https://huggingface.co/datasets):

```bash
python run_glue.py \
  --model_name_or_path bert-base-cased \
  --ipu_config_name Graphcore/bert-base \
  --dataset_name imdb  \
  --do_train \
  --do_predict \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /tmp/imdb/
```

## XNLI

Based on the script [`run_xnli.py`](https://github.com/huggingface/transformers/examples/pytorch/text-classification/run_xnli.py).

[XNLI](https://www.nyu.edu/projects/bowman/xnli/) is a crowd-sourced dataset based on [MultiNLI](http://www.nyu.edu/projects/bowman/multinli/). It is an evaluation benchmark for cross-lingual text representations. Pairs of text are labeled with textual entailment annotations for 15 different languages (including both high-resource language such as English and low-resource languages such as Swahili).

#### Fine-tuning on XNLI

This example code fine-tunes mBERT (multi-lingual BERT) on the XNLI dataset. <!-- TODO: compute time -> It runs in 106 mins on a single tesla V100 16GB.-->

```bash
python run_xnli.py \
  --model_name_or_path bert-base-multilingual-cased \
  --ipu_config_name Graphcore/bert-base \
  --language de \
  --train_language en \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 32 \
  --learning_rate 5e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 128 \
  --output_dir /tmp/debug_xnli/ \
  --save_steps -1
```

<!-- TODO: insert accuracy
Training with the previously defined hyper-parameters yields the following results on the **test** set:

```bash
acc = 0.7093812375249501
```
-->
