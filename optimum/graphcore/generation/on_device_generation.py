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

from typing import Optional, Union

import torch

from optimum.utils import logging


logger = logging.get_logger(__name__)


class OnDeviceGreedySearch(torch.nn.Module):
    def __init__(self, logits_processor):
        super().__init__()
        self.logits_processor = logits_processor

    def forward(self, input_ids, logits, absolute_step):
        next_tokens_scores = self.logits_processor(input_ids, logits, absolute_step=absolute_step)
        return torch.argmax(next_tokens_scores, dim=-1, keepdim=True)


class OnDeviceBeamSearch(torch.nn.Module):
    def forward(self, input_ids, logits, absolute_step):
        raise NotImplementedError()


class OnDeviceGenerationModel(torch.nn.Module):
    """
    A wrapper around a user generation model that effectively runs the entire generation loop
    on device without returning to host after each generated token. Instead, each generated token is stored
    in a torch buffer, and appended to the input at the next time step.
    Suppose the input tensor is of shape [B, C]; B = batch size, C = context length. To generate up to
    a max sequence length of S, device iterations should be set to <= S - C.  Further, this is only compatible
    with poptorch.ShardedExecution.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        generation_strategy: Union[OnDeviceGreedySearch, OnDeviceBeamSearch],
        batch_size: int,
        max_sequence_length: int,
        pad_token_id: int,
        eos_token_id: int,
        context_length: Optional[int] = 1,
        num_beams: Optional[int] = 1,
        use_cache: Optional[bool] = True,
    ):
        super().__init__()

        self.model = model
        self.generation_strategy = generation_strategy
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.context_length = context_length
        self.eos_token_id = eos_token_id
        self.num_beams = num_beams
        self.use_cache = use_cache

        self.max_generation_step = max_sequence_length - context_length

        self.buffer_reset_value = pad_token_id

        # Poptorch buffers become constant if kept as int, so set them as float.
        self.register_buffer(
            "generated_tokens",
            torch.zeros((batch_size * num_beams, self.max_generation_step), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer("_generation_step", torch.tensor([0], dtype=torch.float32), persistent=False)

    def forward(self, **kwargs):
        input_ids_key = "decoder_input_ids"
        input_ids = kwargs.pop(input_ids_key, None)
        if input_ids is None:
            input_ids_key = "input_ids"
            input_ids = kwargs.pop(input_ids_key, None)
            if input_ids is None:
                raise ValueError(
                    f"On device generation model was called with kwargs that are missing both `decoder_input_ids` "
                    "and `input_ids`. Please provide one of these as inputs (default is `decoder_input_ids`)."
                )

        assert input_ids.shape[0] == self.generated_tokens.shape[0], input_ids.shape
        assert input_ids.ndim == self.generated_tokens.ndim == 2
        assert input_ids.shape[1] == self.context_length

        if generation_step := kwargs.pop("generation_step", None) is not None:
            self._generation_step.copy_(generation_step)

        # Make sure generation_step does not go out of bounds.
        self._generation_step.copy_((self._generation_step < self.max_generation_step) * self._generation_step)

        # Reset on-device state buffers when starting generation anew.
        begin_new_generation = (self._generation_step == 0).int()
        buffer_ = (1 - begin_new_generation) * self.generated_tokens + begin_new_generation * torch.ones_like(
            self.generated_tokens
        ) * self.buffer_reset_value
        generation_step = (1 - begin_new_generation) * self._generation_step
        self.generated_tokens.copy_(buffer_)
        self._generation_step.copy_(generation_step)

        # input_ids provides tokens for timesteps < context_length, buffer for timesteps >= context_length
        all_tokens = torch.cat([input_ids, self.generated_tokens.to(input_ids.dtype)], dim=1)
        assert all_tokens.shape[1] == self.max_sequence_length

        model_input_ids = all_tokens
        absolute_step = self.context_length + self._generation_step.to(torch.long) - 1
        if self.use_cache:
            model_input_ids = torch.index_select(all_tokens, 1, absolute_step)

        kwargs[input_ids_key] = model_input_ids
        logits = self.model(**kwargs)
        if hasattr(logits, "logits"):
            logits = logits.logits
        if logits.shape[1] > 1:
            logits = torch.index_select(logits, 1, absolute_step)
        logits = logits.squeeze(1).float()

        tokens = self.generation_strategy(model_input_ids, logits, absolute_step)
        assert tokens.shape == (self.generated_tokens.shape[0], 1)
        assert tokens.ndim == 2

        # If sentence has finished - previous token was EOS - set new tokens to EOS.
        prev_tokens = torch.index_select(all_tokens, 1, absolute_step)
        sentence_eos = (prev_tokens == self.eos_token_id).int()
        tokens = sentence_eos * self.eos_token_id + (1 - sentence_eos) * tokens

        # Update on-device state buffers.
        # Workaround for generic slice assignment self.generated_tokens[:, self.timestep] = tokens
        generation_step = torch.ones(
            (self.generated_tokens.shape[0], 1), dtype=torch.long, device=self.generated_tokens.device
        )
        generation_step *= self._generation_step.to(torch.long)
        self.generated_tokens.scatter_add_(
            1, generation_step, (tokens - self.buffer_reset_value).to(self.generated_tokens.dtype)
        )

        self._generation_step.copy_(self._generation_step + 1)

        return tokens
