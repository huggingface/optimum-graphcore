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

import poptorch
import torch
from transformers.generation.logits_process import (
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    ForceTokensLogitsProcessor,
    MinLengthLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,
    WhisperTimeStampLogitsProcessor,
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
    @classmethod
    def from_model(cls, inst, vocab_size: int):
        self = inst
        self.__class__ = cls

        self.force_token_map_keys = torch.tensor(list(self.force_token_map.keys()))
        self.force_token_map_values = torch.tensor(list(self.force_token_map.values()))
        return self

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, absolute_step: torch.IntTensor
    ) -> torch.FloatTensor:
        mask = absolute_step == self.force_token_map_keys.to(scores.device)
        selected_value = torch.amax(mask * self.force_token_map_values.to(scores.device))
        value_mask = torch.arange(scores.shape[1], dtype=torch.long) == selected_value
        cond = torch.any(mask).int()
        scores = cond * torch.ones_like(scores) * VERY_LARGE_NEGATIVE_CONST + (1 - cond) * scores
        scores -= cond * value_mask.unsqueeze(0) * VERY_LARGE_NEGATIVE_CONST
        return scores


class IPUWhisperTimeStampLogitsProcessor(WhisperTimeStampLogitsProcessor):
    @classmethod
    def from_model(cls, inst, vocab_size: int):
        self = inst
        self.__class__ = cls

        self.no_timestamps_mask = torch.ones((1, vocab_size), dtype=torch.int32)
        self.no_timestamps_mask[:, self.no_timestamps_token_id] = 0

        self.after_timestamp_begin_mask = torch.ones((1, vocab_size), dtype=torch.int32)
        self.after_timestamp_begin_mask[:, self.timestamp_begin :] = 0

        self.before_eos_mask = torch.ones((1, vocab_size), dtype=torch.int32)
        self.before_eos_mask[:, : self.eos_token_id] = 0

        self.last_allowed_mask = torch.ones((1, vocab_size), dtype=torch.int32)
        if self.max_initial_timestamp_index is not None:
            self.last_allowed_mask[:, self.timestamp_begin + self.max_initial_timestamp_index + 1 :] = 0

        self.timestamp_begin_scores = VERY_LARGE_NEGATIVE_CONST * torch.ones((1, vocab_size), dtype=torch.int32)
        self.timestamp_begin_scores[:, self.timestamp_begin] = 0

        self.pre_timestamp_begin_mask = torch.ones((1, vocab_size), dtype=torch.int32)
        self.pre_timestamp_begin_mask[:, : self.timestamp_begin] = 0

        return self

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, absolute_step: torch.IntTensor
    ) -> torch.FloatTensor:
        input_ids = input_ids.view(-1, input_ids.shape[-1])

        no_timestamps_mask = self.no_timestamps_mask.to(scores.device)
        scores = no_timestamps_mask * scores + (1 - no_timestamps_mask) * VERY_LARGE_NEGATIVE_CONST

        cur_len = absolute_step

        cond = cur_len == self.begin_index - 1
        scores = ~cond * scores + cond * self.timestamp_begin_scores.to(device=scores.device)
        timestamp_begin_not_forced = ~cond

        last_was_timestamp = torch.index_select(input_ids, 1, cur_len - 1) >= self.timestamp_begin
        last_was_timestamp &= (cur_len - self.begin_index) >= 1
        penultimate_was_timestamp = (
            torch.index_select(input_ids, 1, torch.where(cur_len > 1, cur_len - 2, cur_len - 1))
            >= self.timestamp_begin
        )
        penultimate_was_timestamp |= (cur_len - self.begin_index) < 2

        after_timestamp_begin_mask = self.after_timestamp_begin_mask.to(scores.device)
        before_eos_mask = self.before_eos_mask.to(scores.device)
        after_timestamp_begin_mask = after_timestamp_begin_mask | ~(
            timestamp_begin_not_forced & last_was_timestamp & penultimate_was_timestamp
        )
        before_eos_mask = before_eos_mask | ~(
            timestamp_begin_not_forced & last_was_timestamp & ~penultimate_was_timestamp
        )
        scores = after_timestamp_begin_mask * scores + (1 - after_timestamp_begin_mask) * VERY_LARGE_NEGATIVE_CONST
        scores = before_eos_mask * scores + (1 - before_eos_mask) * VERY_LARGE_NEGATIVE_CONST

        last_allowed_mask = self.last_allowed_mask.to(scores.device)
        apply_max_initial_timestamp = cur_len == self.begin_index
        last_allowed_mask = last_allowed_mask | ~(timestamp_begin_not_forced & apply_max_initial_timestamp)
        scores = last_allowed_mask * scores + (1 - last_allowed_mask) * VERY_LARGE_NEGATIVE_CONST

        log_probs = torch.nn.functional.log_softmax(scores, dim=-1)
        timestamp_logprob = torch.logsumexp(log_probs[:, self.timestamp_begin :], dim=-1, keepdim=True)
        max_text_token_logprob = torch.amax(log_probs[:, : self.timestamp_begin], dim=-1, keepdim=True)
        pre_timestamp_begin_mask = self.pre_timestamp_begin_mask.to(scores.device)
        pre_timestamp_begin_mask = pre_timestamp_begin_mask | ~(
            timestamp_begin_not_forced & (timestamp_logprob > max_text_token_logprob)
        )
        scores = pre_timestamp_begin_mask * scores + (1 - pre_timestamp_begin_mask) * VERY_LARGE_NEGATIVE_CONST

        return scores


IPULogitsProcessors = {
    ForcedBOSTokenLogitsProcessor: IPUForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor: IPUForcedEOSTokenLogitsProcessor,
    ForceTokensLogitsProcessor: IPUForceTokensLogitsProcessor,
    MinLengthLogitsProcessor: IPUMinLengthLogitsProcessor,
    NoRepeatNGramLogitsProcessor: IPUNoRepeatNGramLogitsProcessor,
    SuppressTokensLogitsProcessor: IPUSuppressTokensLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor: IPUSuppressTokensAtBeginLogitsProcessor,
    WhisperTimeStampLogitsProcessor: IPUWhisperTimeStampLogitsProcessor,
}
