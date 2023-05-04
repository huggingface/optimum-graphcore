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

# Multiple Choice

## Fine-tuning on SWAG with the Trainer

`run_swag` allows you to fine-tune any model from our [hub](https://huggingface.co/models) (as long as its architecture as a `ForMultipleChoice` version in the library) on the SWAG dataset or your own csv/jsonlines files as long as they are structured the same way. To make it works on another dataset, you will need to tweak the `preprocess_function` inside the script.

```
python examples/multiple-choice/run_swag.py \
--model_name_or_path roberta-base \
--ipu_config_name Graphcore/roberta-base-ipu \
--do_train \
--do_eval \
--learning_rate 5e-5 \
--num_train_epochs 3 \
--output_dir ./output/swag_base \
--per_device_eval_batch_size=2 \
--per_device_train_batch_size=2 \
--gradient_accumulation_steps 16 \
--n_ipu 16 \
--report_to none \
--pad_on_batch_axis \
--overwrite_output
```

Training with the defined hyper-parameters yields the following results:
```
***** Eval results *****
eval_acc = 0.8396
eval_loss = 0.439
```
