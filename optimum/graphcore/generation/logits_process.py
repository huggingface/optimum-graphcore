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

from transformers.generation.utils import (
    ForceTokensLogitsProcessor,
    MinLengthLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,
)


VERY_LARGE_NEGATIVE_CONST = -1e18


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
        mask |= cond
        return mask * scores + (1 - mask) * VERY_LARGE_NEGATIVE_CONST


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
        mask |= cond
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
    ForceTokensLogitsProcessor: IPUForceTokensLogitsProcessor,
    MinLengthLogitsProcessor: IPUMinLengthLogitsProcessor,
    SuppressTokensLogitsProcessor: IPUSuppressTokensLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor: IPUSuppressTokensAtBeginLogitsProcessor,
}
