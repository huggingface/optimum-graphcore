from typing import List, Union

import numpy as np

from transformers import ZeroShotClassificationPipeline
from transformers.pipelines.base import PIPELINE_INIT_ARGS, ArgumentHandler, ChunkPipeline
from transformers.tokenization_utils import TruncationStrategy
from transformers.utils import add_end_docstrings, logging


logger = logging.get_logger(__name__)


@add_end_docstrings(PIPELINE_INIT_ARGS)
class IPUZeroShotClassificationPipeline(ZeroShotClassificationPipeline):
    def _parse_and_tokenize(
        self, sequence_pairs, padding=True, add_special_tokens=True, truncation=TruncationStrategy.ONLY_FIRST, **kwargs
    ):
        """
        Parse arguments and tokenize only_first so that hypothesis (label) is not truncated
        """
        return_tensors = self.framework
        if self.tokenizer.pad_token is None:
            # Override for tokenizers not supporting padding
            logger.error(
                "Tokenizer was not supporting padding necessary for zero-shot, attempting to use "
                " `pad_token=eos_token`"
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
        try:
            inputs = self.tokenizer(
                sequence_pairs,
                add_special_tokens=add_special_tokens,
                return_tensors=return_tensors,
                padding=padding,
                truncation=truncation,
                **kwargs,
            )
        except Exception as e:
            if "too short" in str(e):
                # tokenizers might yell that we want to truncate
                # to a value that is not even reached by the input.
                # In that case we don't want to truncate.
                # It seems there's not a really better way to catch that
                # exception.

                inputs = self.tokenizer(
                    sequence_pairs,
                    add_special_tokens=add_special_tokens,
                    return_tensors=return_tensors,
                    padding=padding,
                    truncation=TruncationStrategy.DO_NOT_TRUNCATE,
                )
            else:
                raise e

        return inputs

    def _sanitize_parameters(self, **kwargs):
        if kwargs.get("multi_class", None) is not None:
            kwargs["multi_label"] = kwargs["multi_class"]
            logger.warning(
                "The `multi_class` argument has been deprecated and renamed to `multi_label`. "
                "`multi_class` will be removed in a future version of Transformers."
            )
        preprocess_params = {}
        if "candidate_labels" in kwargs:
            preprocess_params["candidate_labels"] = self._args_parser._parse_labels(kwargs["candidate_labels"])
        if "hypothesis_template" in kwargs:
            preprocess_params["hypothesis_template"] = kwargs["hypothesis_template"]
        if "padding" in kwargs:
            preprocess_params["padding"] = kwargs["padding"]
        if "max_length" in kwargs:
            preprocess_params["max_length"] = kwargs["max_length"]

        postprocess_params = {}
        if "multi_label" in kwargs:
            postprocess_params["multi_label"] = kwargs["multi_label"]
        return preprocess_params, {}, postprocess_params

    def preprocess(self, inputs, candidate_labels=None, hypothesis_template="This example is {}.", **tokenizer_kwargs):
        sequence_pairs, sequences = self._args_parser(inputs, candidate_labels, hypothesis_template)

        for i, (candidate_label, sequence_pair) in enumerate(zip(candidate_labels, sequence_pairs)):
            model_input = self._parse_and_tokenize([sequence_pair], **tokenizer_kwargs)

            yield {
                "candidate_label": candidate_label,
                "sequence": sequences[0],
                "is_last": i == len(candidate_labels) - 1,
                **model_input,
            }
