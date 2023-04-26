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


LARGE_NEGATIVE_CONST = -1e9


class IPUMinLengthLogitsProcessor(MinLengthLogitsProcessor):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, absolute_step: torch.IntTensor
    ) -> torch.FloatTensor:
        # upstream checks `if input_ids.shape[-1] < self.min_length:`
        idx = torch.ones((scores.shape[0], 1), dtype=torch.long, device=scores.device) * self.eos_token_id
        cond = absolute_step + 1 < self.min_length
        val = cond * torch.ones_like(idx) * LARGE_NEGATIVE_CONST
        scores.scatter_add_(1, idx, val)
        return scores


class IPUSuppressTokensLogitsProcessor(SuppressTokensLogitsProcessor):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, absolute_step: torch.IntTensor
    ) -> torch.FloatTensor:
        idx = torch.ones(
            (scores.shape[0], len(self.suppress_tokens)), dtype=torch.long, device=scores.device
        ) * torch.tensor(self.suppress_tokens)
        val = torch.ones_like(idx) * LARGE_NEGATIVE_CONST
        scores.scatter_add_(1, idx, val)
        return scores


class IPUSuppressTokensAtBeginLogitsProcessor(SuppressTokensAtBeginLogitsProcessor):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, absolute_step: torch.IntTensor
    ) -> torch.FloatTensor:
        idx = torch.ones(
            (scores.shape[0], len(self.begin_suppress_tokens)), dtype=torch.long, device=scores.device
        ) * torch.tensor(self.begin_suppress_tokens)
        cond = absolute_step + 1 == self.begin_index
        val = cond * torch.ones_like(idx) * LARGE_NEGATIVE_CONST
        scores.scatter_add_(1, idx, val)
        return scores


class IPUForceTokensLogitsProcessor(ForceTokensLogitsProcessor):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, absolute_step: torch.IntTensor
    ) -> torch.FloatTensor:
        force_token_map_keys = torch.tensor(list(self.force_token_map.keys()))
        force_token_map_values = torch.tensor(list(self.force_token_map.values()))
        mask = absolute_step + 1 == force_token_map_keys
        selected_value = torch.amax(mask * force_token_map_values)
        cond = torch.any(mask).int()
        scores = cond * torch.ones_like(scores) * LARGE_NEGATIVE_CONST + (1 - cond) * scores
        idx = torch.ones((scores.shape[0], 1), dtype=torch.long, device=scores.device) * selected_value
        val = cond * torch.ones_like(idx) * -LARGE_NEGATIVE_CONST
        scores.scatter_add_(1, idx, val)
        return scores


IPULogitsProcessors = {
    ForceTokensLogitsProcessor: IPUForceTokensLogitsProcessor,
    MinLengthLogitsProcessor: IPUMinLengthLogitsProcessor,
    SuppressTokensLogitsProcessor: IPUSuppressTokensLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor: IPUSuppressTokensAtBeginLogitsProcessor,
}
