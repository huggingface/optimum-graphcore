# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
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

"""
IPU implementation of the Gumbel vector quantizer.
Compared to the original implementation, the main differences are:
- The temperature is passed as an input to the module's forward() method
  instead of being a class attribute, in order to allow different values
  of the temperature at runtime on IPU;
- Use of an IPU-customised Gumbel softmax;
- Replace a large element-wise multiplication followed by a sum reduction
  with a matrix multiplication (einsum), to allow for a more efficient memory
  usage on IPU.
"""

import warnings

import torch

from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2GumbelVectorQuantizer


def _ipu_gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.")

    gumbels = -(
        torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_() + 1e-4
    ).log()  # ~Gumbel(0,1)

    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        index = y_soft.max(dim, keepdim=True)[1]

        update_values = torch.ones_like(index, dtype=logits.dtype)
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(
            dim, index, update_values
        )

        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft

    return ret


class IPUWav2Vec2GumbelVectorQuantizer(Wav2Vec2GumbelVectorQuantizer):
    @staticmethod
    def _compute_perplexity(probs, mask=None):
        if mask is not None:
            probs *= mask.flatten()[:, None, None].float()
            num = mask.sum()
            marginal_probs = probs.sum(dim=0) / num
        else:
            num = probs.shape[0]
            marginal_probs = probs.sum(dim=0) / num

        log_marginal_probs = torch.log(marginal_probs + 1e-7)
        perplexity = torch.exp(-torch.sum(marginal_probs * log_marginal_probs, dim=-1)).sum()
        return perplexity

    def forward(self, hidden_states, gumbel_temperature=2.0, mask_time_indices=None):
        batch_size, sequence_length, hidden_size = hidden_states.shape

        # project to codevector dim
        hidden_states = self.weight_proj(hidden_states)
        hidden_states = hidden_states.view(batch_size * sequence_length * self.num_groups, -1)

        codevector_idx = hidden_states.argmax(dim=-1)
        hard_probs = torch.nn.functional.one_hot(codevector_idx.long(), num_classes=self.num_vars).view(
            batch_size * sequence_length, self.num_groups, -1
        )
        code_perplexity = self._compute_perplexity(hard_probs.float(), mask_time_indices)

        soft_probs = torch.softmax(
            hidden_states.view(batch_size * sequence_length, self.num_groups, -1).float(),
            dim=-1,
        )
        prob_perplexity = self._compute_perplexity(soft_probs, mask_time_indices)

        if self.training:
            # sample code vector probs via gumbel in differentiateable way
            codevector_probs = _ipu_gumbel_softmax(hidden_states.float(), tau=gumbel_temperature, hard=True).type_as(
                hidden_states
            )
        else:
            codevector_probs = hard_probs.type_as(hidden_states)

        codevector_probs = codevector_probs.view(batch_size * sequence_length, self.num_groups, -1)
        codebook = self.codevectors[0, :, :]
        codebook = codebook.view(self.num_groups, self.num_vars, -1)
        codevectors = torch.bmm(codevector_probs.permute(1, 0, 2), codebook).permute(1, 0, 2)
        codevectors = codevectors.reshape(batch_size, sequence_length, -1)

        codevectors = codevectors.reshape(batch_size, sequence_length, -1)

        return codevectors, code_perplexity, prob_perplexity
