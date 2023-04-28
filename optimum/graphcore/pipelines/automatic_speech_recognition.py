# Copyright 2021 The HuggingFace Team. All rights reserved.
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
import os

import torch

from optimum.graphcore.models.whisper import PipelinedWhisperForConditionalGeneration

# IPU: IPU-specific imports
from poptorch import PoplarExecutor
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(self.model.model, PipelinedWhisperForConditionalGeneration):
            self.type = "seq2seq"
        self.fp16 = kwargs.get("fp16", True)

    def _sanitize_parameters(self, **kwargs):
        preprocess_params, _, postprocess_params = super()._sanitize_parameters(**kwargs)
        forward_params = {}

        params = ["num_beams", "use_cache", "do_sample", "max_new_tokens"]
        for p in params:
            if p in kwargs:
                forward_params[p] = kwargs[p]

        return preprocess_params, forward_params, postprocess_params

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

        # IPU: the last batch may contain fewer than batch_size elements. In that case we must pad it
        def batch_padding(items, batch_size):
            if batch_size == 1:
                return items
            actual_batch_size = len(items["is_last"])
            if actual_batch_size < batch_size:
                n_to_pad = batch_size - actual_batch_size
                is_last = items["is_last"]
                stride = items["stride"]
                input_features = items["input_features"]
                is_last = is_last + [False] * n_to_pad
                stride = stride + [stride[-1]] * n_to_pad
                new_input_features = torch.zeros(batch_size, *(input_features.shape[1:]), dtype=input_features.dtype)
                new_input_features[:actual_batch_size] = input_features
                items = dict(is_last=is_last, stride=stride, input_features=new_input_features)
            return items

        batch_padder = PipelineIterator(dataloader, batch_padding, dict(batch_size=batch_size))
        model_iterator = PipelinePackIterator(batch_padder, self.forward, forward_params, loader_batch_size=batch_size)

        final_iterator = PipelineIterator(model_iterator, self.postprocess, postprocess_params)
        return final_iterator

    def _forward(self, model_inputs, **forward_params):
        is_last = model_inputs.pop("is_last")
        if self.type == "seq2seq":
            encoder = self.model.get_encoder()
            # Consume values so we can let extra information flow freely through
            # the pipeline (important for `partial` in microphone)
            if "input_features" in model_inputs:
                inputs = model_inputs.pop("input_features")
            elif "input_values" in model_inputs:
                inputs = model_inputs.pop("input_values")
            else:
                raise ValueError(
                    "Seq2Seq speech recognition model requires either a "
                    f"`input_features` or `input_values` key, but only has {model_inputs.keys()}"
                )

            # we need to pass `processed.get("attention_mask")` here since audio encoder
            # attention mask  length is different from expected text decoder `encoder_attention_mask` length
            # `generate` magic to create the mask automatically won't work, we basically need to help
            # it here.
            attention_mask = model_inputs.pop("attention_mask", None)
            # IPU: For Whisper we call generate() without encoder_output nor attention_mask
            if not isinstance(self.model, PoplarExecutor):
                if isinstance(self.model, PipelinedWhisperForConditionalGeneration):
                    print("Calling generate()")
                    tokens = self.model.generate(inputs=inputs, attention_mask=attention_mask, **forward_params)
                else:
                    tokens = self.model.generate(
                        encoder_outputs=encoder(inputs, attention_mask=attention_mask),
                        attention_mask=attention_mask,
                    )
            elif isinstance(self.model.model, PipelinedWhisperForConditionalGeneration):
                tokens = self.model.generate(inputs, attention_mask, **forward_params)
            else:
                raise ValueError("Unsupported model")

            out = {"tokens": tokens}

        else:
            stride = model_inputs.pop("stride", None)
            input_values = model_inputs.pop("input_values")
            attention_mask = model_inputs.pop("attention_mask", None)
            outputs = self.model(input_values=input_values, attention_mask=attention_mask)
            logits = outputs.logits

            if self.type == "ctc_with_lm":
                out = {"logits": logits}
            else:
                out = {"tokens": logits.argmax(dim=-1)}
            if stride is not None:
                # Send stride to `postprocess`.
                # it needs to be handled there where
                # the pieces are to be concatenated.
                ratio = 1 / self.model.config.inputs_to_logits_ratio
                if isinstance(stride, tuple):
                    out["stride"] = rescale_stride([stride], ratio)[0]
                else:
                    out["stride"] = rescale_stride(stride, ratio)
        # Leftover
        extra = model_inputs
        return {"is_last": is_last, **out, **extra}
