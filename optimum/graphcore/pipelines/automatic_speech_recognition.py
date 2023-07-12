# Copyright 2021 The HuggingFace Team. All rights reserved.
# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from transformers import AutomaticSpeechRecognitionPipeline
from transformers.pipelines.base import (
    DataLoader,
    PipelineChunkIterator,
    PipelineIterator,
    PipelinePackIterator,
    no_collate_fn,
    pad_collate_fn,
)
from transformers.utils import logging


logger = logging.get_logger(__name__)


class IPUAutomaticSpeechRecognitionPipeline(AutomaticSpeechRecognitionPipeline):
    def get_iterator(
        self, inputs, num_workers: int, batch_size: int, preprocess_params, forward_params, postprocess_params
    ):
        if "TOKENIZERS_PARALLELISM" not in os.environ:
            logger.info("Disabling tokenizer parallelism, we're using DataLoader multithreading already")
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if num_workers > 1:
            logger.warning(
                "For ChunkPipeline using num_workers>0 is likely to result in errors since everything is iterable,"
                " setting `num_workers=1` to guarantee correctness."
            )
            num_workers = 1
        dataset = PipelineChunkIterator(inputs, self.preprocess, preprocess_params)
        collate_fn = no_collate_fn if batch_size == 1 else pad_collate_fn(self.tokenizer, self.feature_extractor)
        dataloader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, collate_fn=collate_fn)

        # Change: If the last batch contains fewer than `batch_size` elements, pad it.
        def batch_padding(items, batch_size):
            is_last = items["is_last"]

            actual_batch_size = 1 if isinstance(is_last, bool) else len(is_last)
            if actual_batch_size >= batch_size:
                return items

            n_to_pad = batch_size - actual_batch_size
            is_last = is_last + [None] * n_to_pad

            # Pad input features by duplicating with genuine feature values as opposed to
            # e.g. zeros. This makes it significantly more likely beam search will terminate.
            input_features = items["input_features"]
            new_input_features = input_features.repeat(
                batch_size // actual_batch_size + 1, *([1] * (input_features.ndim - 1))
            )
            new_input_features = new_input_features[:batch_size]

            padded_items = {"is_last": is_last, "input_features": new_input_features}

            stride = items.get("stride", None)
            if stride is not None:
                stride = stride + [stride[-1]] * n_to_pad
                padded_items["stride"] = stride
            return padded_items

        if self.type == "seq2seq_whisper":
            dataloader = PipelineIterator(dataloader, batch_padding, {"batch_size": batch_size})
        model_iterator = PipelinePackIterator(dataloader, self.forward, forward_params, loader_batch_size=batch_size)
        final_iterator = PipelineIterator(model_iterator, self.postprocess, postprocess_params)
        return final_iterator

    def _forward(self, model_inputs, return_timestamps=False, generate_kwargs=None):
        if not self.type == "seq2seq_whisper":
            return super()._forward(model_inputs, return_timestamps=return_timestamps, generate_kwargs=generate_kwargs)

        if generate_kwargs is None:
            generate_kwargs = {}
        if return_timestamps:
            generate_kwargs["return_timestamps"] = return_timestamps
        is_last = model_inputs.pop("is_last")

        if "input_features" in model_inputs:
            inputs = model_inputs.pop("input_features")
        elif "input_values" in model_inputs:
            inputs = model_inputs.pop("input_values")
        else:
            raise ValueError(
                "Seq2Seq speech recognition model requires either a "
                f"`input_features` or `input_values` key, but only has {model_inputs.keys()}"
            )

        attention_mask = model_inputs.pop("attention_mask", None)
        tokens = self.model.generate(inputs=inputs, attention_mask=attention_mask, **generate_kwargs)

        out = {"tokens": tokens}
        stride = model_inputs.pop("stride", None)
        if stride is not None:
            out["stride"] = stride

        extra = model_inputs
        maybe_padded_ret = {"is_last": is_last, **out, **extra}

        # Remove inputs and outputs associated with padded inputs.
        if not isinstance(is_last, list):
            is_last = [is_last]

        first_padding_idx = tokens.shape[0]
        for idx, last in enumerate(is_last):
            if last is None:
                first_padding_idx = idx
                break

        if first_padding_idx == tokens.shape[0]:
            return maybe_padded_ret

        padded_keys = ["is_last", "tokens"]
        if stride is not None:
            padded_keys.append("stride")
        for padded_key in padded_keys:
            maybe_padded_ret[padded_key] = maybe_padded_ret[padded_key][:first_padding_idx]
        return maybe_padded_ret
