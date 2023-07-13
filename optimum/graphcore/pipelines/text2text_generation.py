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

from transformers import SummarizationPipeline, Text2TextGenerationPipeline, TranslationPipeline
from transformers.pipelines.text2text_generation import TruncationStrategy


class IPUText2TextGenerationPipeline(Text2TextGenerationPipeline):
    def _sanitize_parameters(
        self,
        return_tensors=None,
        return_text=None,
        return_type=None,
        clean_up_tokenization_spaces=None,
        truncation=None,
        stop_sequence=None,
        max_input_length=None,
        **generate_kwargs,
    ):
        preprocess_params, forward_params, postprocess_params = super()._sanitize_parameters(
            return_tensors,
            return_text,
            return_type,
            clean_up_tokenization_spaces,
            truncation,
            stop_sequence,
            **generate_kwargs,
        )
        if max_input_length is not None:
            preprocess_params["max_input_length"] = max_input_length
        return preprocess_params, forward_params, postprocess_params

    def _parse_and_tokenize(self, *args, truncation, **kwargs):
        prefix = self.model.config.prefix if self.model.config.prefix is not None else ""
        if isinstance(args[0], list):
            if self.tokenizer.pad_token_id is None:
                raise ValueError("Please make sure that the tokenizer has a pad_token_id when using a batch input")
            args = ([prefix + arg for arg in args[0]],)
            padding = True

        elif isinstance(args[0], str):
            args = (prefix + args[0],)
            padding = False
        else:
            raise ValueError(
                f" `args[0]`: {args[0]} have the wrong format. The should be either of type `str` or type `list`"
            )
        padding = "max_length"
        inputs = self.tokenizer(
            *args,
            padding=padding,
            max_length=kwargs.get("max_input_length"),
            truncation=truncation,
            return_tensors=self.framework,
        )
        # This is produced by tokenizers but is an invalid generate kwargs
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]
        return inputs


class IPUSummarizationPipeline(SummarizationPipeline, IPUText2TextGenerationPipeline):
    pass


class IPUTranslationPipeline(TranslationPipeline, IPUText2TextGenerationPipeline):
    def preprocess(
        self, *args, truncation=TruncationStrategy.DO_NOT_TRUNCATE, src_lang=None, tgt_lang=None, max_input_length=None
    ):
        if getattr(self.tokenizer, "_build_translation_inputs", None):
            return self.tokenizer._build_translation_inputs(
                *args,
                return_tensors=self.framework,
                max_length=max_input_length,
                padding="max_length",
                truncation=truncation,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
            )
        else:
            return super()._parse_and_tokenize(*args, truncation=truncation, max_input_length=max_input_length)
