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

import torch

import poptorch
from transformers.generation.utils import (
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    ForceTokensLogitsProcessor,
    MinLengthLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,
)


VERY_LARGE_NEGATIVE_CONST = -1e18


class IPUForcedBOSTokenLogitsProcessor(ForcedBOSTokenLogitsProcessor):
    @classmethod
    def from_model(cls, inst, vocab_size: int):
        self = inst
        self.bos_scores = VERY_LARGE_NEGATIVE_CONST * torch.ones((1, vocab_size), dtype=torch.int32)
        self.bos_scores[:, self.bos_token_id] = 0
        self.__class__ = cls
        return self

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, absolute_step: torch.IntTensor
    ) -> torch.FloatTensor:
        cond = (absolute_step > 1).int()
        return cond * scores + (1 - cond) * self.bos_scores.to(device=scores.device)


class IPUForcedEOSTokenLogitsProcessor(ForcedEOSTokenLogitsProcessor):
    @classmethod
    def from_model(cls, inst, vocab_size: int):
        self = inst
        self.eos_scores = VERY_LARGE_NEGATIVE_CONST * torch.ones((1, vocab_size), dtype=torch.int32)
        self.eos_scores[:, self.eos_token_id] = 0
        self.__class__ = cls
        return self

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, absolute_step: torch.IntTensor
    ) -> torch.FloatTensor:
        cond = (absolute_step < self.max_length).int()
        return cond * scores + (1 - cond) * self.eos_scores.to(device=scores.device)


class IPUMinLengthLogitsProcessor(MinLengthLogitsProcessor):
    @classmethod
    def from_model(cls, inst, vocab_size: int):
        self = inst
        self.__class__ = cls

        self.mask = torch.ones((1, vocab_size), dtype=torch.int32)
        self.mask[:, self.eos_token_id] = 0
        return self

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, absolute_step: torch.IntTensor
    ) -> torch.FloatTensor:
        mask = self.mask.to(scores.device)
        cond = absolute_step >= self.min_length
        mask = mask | cond
        return mask * scores + (1 - mask) * VERY_LARGE_NEGATIVE_CONST


class IPUNoRepeatNGramLogitsProcessor(NoRepeatNGramLogitsProcessor):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, absolute_step: torch.IntTensor
    ) -> torch.FloatTensor:
        # mask out values above cur_len
        cur_len = absolute_step
        cur_len_mask = torch.arange(0, input_ids.shape[-1], device=input_ids.device).unsqueeze(0) < cur_len
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        input_ids = input_ids * cur_len_mask

        start_idx = torch.maximum(cur_len + 1 - self.ngram_size, torch.tensor(self.ngram_size))
        ngrams = input_ids.unfold(-1, self.ngram_size, 1)
        last_tokens = poptorch.dynamic_slice(input_ids, 1, start_idx, self.ngram_size - 1, 1).unsqueeze(1)
        last_tokens = (start_idx > self.ngram_size) * last_tokens

        mask = torch.all(ngrams[..., : self.ngram_size - 1] == last_tokens, -1)

        # If absolute_step + 1 < ngram_size set indices all to zero
        mask = ~(cur_len + 1 < self.ngram_size) * mask
        idx = torch.where(mask, ngrams[..., -1], -100)

        val = (idx != -100) * torch.ones_like(idx) * VERY_LARGE_NEGATIVE_CONST
        scores.scatter_add_(1, idx, val)
        return scores


class IPUSuppressTokensLogitsProcessor(SuppressTokensLogitsProcessor):
    @classmethod
    def from_model(cls, inst, vocab_size: int):
        self = inst
        self.__class__ = cls

        self.mask = torch.ones((1, vocab_size), dtype=torch.int32)
        self.mask[:, self.suppress_tokens] = 0
        return self

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, absolute_step: torch.IntTensor
    ) -> torch.FloatTensor:
        mask = self.mask.to(scores.device)
        return mask * scores + (1 - mask) * VERY_LARGE_NEGATIVE_CONST


class IPUSuppressTokensAtBeginLogitsProcessor(SuppressTokensAtBeginLogitsProcessor):
    @classmethod
    def from_model(cls, inst, vocab_size: int):
        self = inst
        self.__class__ = cls

        self.mask = torch.ones((1, vocab_size), dtype=torch.int32)
        self.mask[:, self.begin_suppress_tokens] = 0
        return self

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, absolute_step: torch.IntTensor
    ) -> torch.FloatTensor:
        mask = self.mask.to(scores.device)
        cond = absolute_step != self.begin_index
        mask = mask | cond
        return mask * scores + (1 - mask) * VERY_LARGE_NEGATIVE_CONST


class IPUForceTokensLogitsProcessor(ForceTokensLogitsProcessor):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, absolute_step: torch.IntTensor
    ) -> torch.FloatTensor:
        force_token_map_keys = torch.tensor(list(self.force_token_map.keys()))
        force_token_map_values = torch.tensor(list(self.force_token_map.values()))
        mask = absolute_step == force_token_map_keys
        selected_value = torch.amax(mask * force_token_map_values)
        cond = torch.any(mask).int()
        scores = cond * torch.ones_like(scores) * VERY_LARGE_NEGATIVE_CONST + (1 - cond) * scores
        idx = torch.ones((scores.shape[0], 1), dtype=torch.long) * selected_value
        val = cond * torch.ones_like(idx) * -VERY_LARGE_NEGATIVE_CONST
        scores.scatter_add_(1, idx, val)
        return scores


IPULogitsProcessors = {
    ForcedBOSTokenLogitsProcessor: IPUForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor: IPUForcedEOSTokenLogitsProcessor,
    ForceTokensLogitsProcessor: IPUForceTokensLogitsProcessor,
    MinLengthLogitsProcessor: IPUMinLengthLogitsProcessor,
    NoRepeatNGramLogitsProcessor: IPUNoRepeatNGramLogitsProcessor,
    SuppressTokensLogitsProcessor: IPUSuppressTokensLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor: IPUSuppressTokensAtBeginLogitsProcessor,
}
