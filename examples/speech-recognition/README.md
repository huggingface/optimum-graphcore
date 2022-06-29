<!---
Copyright 2021 The HuggingFace Team. All rights reserved.
Copyright 2022 Graphcore Ltd. All rights reserved.

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

# Speech Recognition Fine-Tuning


## Wav2Vec2 Speech Fine-Tuning

## Connectionist Temporal Classification

The script [`run_speech_recognition_ctc.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/speech-recognition/run_speech_recognition_ctc.py) can be used to fine-tune any pretrained [Connectionist Temporal Classification Model](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoModelForCTC) for automatic speech
recognition on one of the [official speech recognition datasets](https://huggingface.co/datasets?task_ids=task_ids:automatic-speech-recognition) or a custom dataset.

Speech recognition models that have been pretrained in unsupervised fashion on audio data alone, *e.g.* [Wav2Vec2](https://huggingface.co/transformers/main/model_doc/wav2vec2.html), [HuBERT](https://huggingface.co/transformers/main/model_doc/hubert.html), [XLSR-Wav2Vec2](https://huggingface.co/transformers/main/model_doc/xlsr_wav2vec2.html), have shown to require only
very little annotated data to yield good performance on automatic speech recognition datasets.

In the script [`run_speech_recognition_ctc`], we first create a vocabulary from all unique characters of both the training data and evaluation data. Then, we preprocesses the speech recognition dataset, which includes correct resampling, normalization and padding. Finally, the pretrained speech recognition model is fine-tuned on the annotated speech recognition datasets using CTC loss.


---
**NOTE**

If you encounter problems with data preprocessing by setting `--preprocessing_num_workers` > 1,
you might want to set the environment variable `OMP_NUM_THREADS` to 1 as follows:

```bash
OMP_NUM_THREADS=1 python run_speech_recognition_ctc ...
```

If the environment variable is not set, the training script might freeze, *i.e.* see: https://github.com/pytorch/audio/issues/1021#issuecomment-726915239

---

### Base

To fine-tune`"base-sized"` Wav2Vec2 model, *e.g.* [facebook/wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base)
on 100h of training data from the [librispeech_asr](https://huggingface.co/datasets/librispeech_asr), the following command can be run:

```bash
python run_speech_recognition_ctc.py \
    --dataset_name="librispeech_asr" \
    --dataset_config_name="clean" \
    --train_split_name="train.100" \
    --eval_split_name="validation" \
    --model_name_or_path="facebook/wav2vec2-base-960h" \
    --ipu_config_name="base_4-ipu_config.json" \
    --mask_time_prob=0.0 \
    --output_dir="./wav2vec2-base-960h" \
    --overwrite_output_dir \
    --length_column_name="input_length" \
    --num_train_epochs="1" \
    --learning_rate="3e-4" \
    --warmup_steps="400" \
    --evaluation_strategy="steps" \
    --text_column_name="text" \
    --save_steps="400" \
    --eval_steps="400" \
    --logging_steps="10" \
    --save_total_limit="1" \
    --freeze_feature_encoder \
    --do_train \
    --do_eval \
    --layerdrop=0.0 \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --adam_beta1=0.9 \
    --adam_beta2=0.98 \
    --adam_epsilon 0.0001 \
    --dataloader_drop_last \
    --dataloader_mode="async_rebatched" \
    --dataloader_num_workers=8 
```

