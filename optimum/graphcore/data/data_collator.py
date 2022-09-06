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

from functools import wraps
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import torch
from torch import Tensor

from transformers.data import DataCollatorForLanguageModeling
from transformers.data.data_collator import DataCollator
from transformers.tokenization_utils_base import BatchEncoding


def pad_on_batch_axis(batch_size: int) -> Callable[[DataCollator], DataCollator]:
    """
    Creates a DataCollator wrapper that pads the batches generated by the DataCollator on the batch axis to generate
    fixed size batches. It implements the padding by repeating elements of the batch to reach the padded sized.
    """

    def pad_tensor(x: Tensor) -> Tensor:
        if batch_size != x.size(0):
            repeat_dims = torch.ones(x.ndim, dtype=int, requires_grad=False)
            num_repeats = batch_size // x.size(0) + 1
            repeat_dims[0] = num_repeats
            return x.repeat(*repeat_dims.tolist())[:batch_size]
        else:
            return x

    def decorator(data_collator: DataCollator) -> DataCollator:
        @wraps(data_collator)
        def wrapper(*args, **kwargs):
            batch = data_collator(*args, **kwargs)
            for k, v in batch.items():
                batch[k] = pad_tensor(v)
            return batch

        return wrapper

    return decorator


class DataCollatorForLanguageModelingWithMaxTokensMasked(DataCollatorForLanguageModeling):
    def __init__(self, max_seq_length, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_seq_length = max_seq_length
        self.max_num_of_masked_tokens = self._calculate_max_num_of_masked_tokens(max_seq_length)

    def _calculate_max_num_of_masked_tokens(self, max_seq_length):
        """
        Get the max number of masked tokens. The number of masked tokens follows a binomial distribution. We approximate
        the binomial distribution with an Gaussian distribution and cap the maximum number of masked tokens to 2 standard
        deviations above the mean.
        """
        import math

        mean = max_seq_length * self.mlm_probability
        var = max_seq_length * self.mlm_probability * (1 - self.mlm_probability)
        std = math.sqrt(var)
        max_num_of_masked_tokens = mean + 2 * std
        # Round up to a multiple of 16
        max_num_of_masked_tokens = math.ceil(max_num_of_masked_tokens / 16) * 16
        # Cap to max_seq_length
        max_num_of_masked_tokens = min(max_num_of_masked_tokens, max_seq_length)
        return max_num_of_masked_tokens

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)
        # Necessary for poptorch.DataLoaderMode.AsyncRebatched which can handle dictionaries but not BatchEncoding.
        if isinstance(batch, BatchEncoding):
            batch = dict(batch)
        return batch

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix)

        # Making sure there are at most max_num_of_masked_tokens that are masked for each example.
        # torch_mask_tokens is called after padding so labels should be of fixed shape.
        # Adding a small noise to -masked_indices to make the torch.topk selection of the ones to delete stochastic.
        small_noise = torch.rand(masked_indices.size())
        _, indices = torch.topk(
            -masked_indices + small_noise, k=self.max_seq_length - self.max_num_of_masked_tokens, dim=1
        )
        masked_indices.scatter_(1, indices, 0)

        masked_indices = masked_indices.bool()

        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
