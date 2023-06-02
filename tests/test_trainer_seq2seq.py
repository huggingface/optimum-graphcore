# coding=utf-8
# Copyright 2020 the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from optimum.graphcore import IPUConfig, IPUSeq2SeqTrainer, IPUSeq2SeqTrainingArguments
from transformers import T5Tokenizer, T5ForConditionalGeneration
import numpy as np

from transformers.testing_utils import TestCasePlus, require_torch, slow
from transformers.utils import is_datasets_available


if is_datasets_available():
    import datasets


class Seq2SeqTrainerTester(TestCasePlus):
    @slow
    @require_torch
    def test_finetune_t5_small(self):
        total_steps_on_device = 5
        batch_size = 2

        ipu_config = IPUConfig.from_pretrained(
            "Graphcore/t5-small-ipu",
            inference_layers_per_ipu=[-1],
            gradient_accumulation_steps=9,
            device_iterations=1,
            inference_device_iterations=1,
        )

        model_checkpoint = "t5-small"
        t5_small = T5ForConditionalGeneration.from_pretrained(model_checkpoint)
        tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
        t5_small.config.max_length = 128

        train_dataset = datasets.load_dataset("cnn_dailymail", "3.0.0", split="train[:1%]")
        val_dataset = datasets.load_dataset("cnn_dailymail", "3.0.0", split="validation[:1%]")

        train_dataset = train_dataset.select(
            range(ipu_config.batch_size_factor() * batch_size * total_steps_on_device)
        )
        val_dataset = val_dataset.select(
            range(ipu_config.batch_size_factor(for_inference=True) * batch_size * total_steps_on_device)
        )

        def _map_to_encoder_decoder_inputs(batch):
            inputs = tokenizer(
                ["summarize: " + article for article in batch["article"]],
                padding="max_length",
                truncation=True,
                max_length=512,
            )
            with tokenizer.as_target_tokenizer():
                outputs = tokenizer(batch["highlights"], padding="max_length", truncation=True, max_length=128)
            batch["input_ids"] = inputs.input_ids
            batch["attention_mask"] = inputs.attention_mask

            batch["decoder_input_ids"] = outputs.input_ids
            batch["labels"] = outputs.input_ids.copy()
            batch["labels"] = [
                [-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]
            ]
            batch["decoder_attention_mask"] = outputs.attention_mask

            assert all([len(x) == 512 for x in inputs.input_ids])
            assert all([len(x) == 128 for x in outputs.input_ids])

            return batch

        def _compute_metrics(pred):
            labels_ids = pred.label_ids
            pred_ids = pred.predictions

            # all unnecessary tokens are removed
            pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

            labels_ids = np.where(labels_ids != -100, labels_ids, tokenizer.pad_token_id)
            label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

            accuracy = sum([int(pred_str[i] == label_str[i]) for i in range(len(pred_str))]) / len(pred_str)

            return {"accuracy": accuracy}

        # map train dataset
        train_dataset = train_dataset.map(
            _map_to_encoder_decoder_inputs,
            batched=True,
            batch_size=batch_size,
            remove_columns=["article", "highlights"],
        )
        train_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
        )

        # same for validation dataset
        val_dataset = val_dataset.map(
            _map_to_encoder_decoder_inputs,
            batched=True,
            batch_size=batch_size,
            remove_columns=["article", "highlights"],
        )
        val_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
        )

        output_dir = self.get_auto_remove_tmp_dir()

        training_args = IPUSeq2SeqTrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            predict_with_generate=True,
            evaluation_strategy="steps",
            do_train=True,
            do_eval=True,
            warmup_steps=0,
            max_steps=total_steps_on_device,
            eval_steps=2,
            logging_steps=2,
        )

        # instantiate trainer
        trainer = IPUSeq2SeqTrainer(
            model=t5_small,
            ipu_config=ipu_config,
            args=training_args,
            compute_metrics=_compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
        )

        # start training
        trainer.train()
