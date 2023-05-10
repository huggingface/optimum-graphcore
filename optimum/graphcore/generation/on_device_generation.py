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

from dataclasses import dataclass
from typing import Optional, Union

import torch

import poptorch
from optimum.utils import logging
from transformers.generation.utils import LogitsProcessorList
from transformers.modeling_outputs import ModelOutput


logger = logging.get_logger(__name__)


LARGE_NEGATIVE_CONST = -1e9


@dataclass
class OnDeviceGenerationModelOutput(ModelOutput):
    generated_tokens: torch.Tensor = None
    done: torch.Tensor = None


class OnDeviceGreedySearch(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        batch_size: int,
        max_length: int,
        eos_token_id: int,
        pad_token_id: int,
        logits_processor: LogitsProcessorList,
    ):
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.max_length = max_length
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.logits_processor = logits_processor

        self.input_ids_mask = torch.zeros((batch_size, max_length), dtype=torch.int32)
        self.input_ids_mask[:, 0] = 1

        # Poptorch buffers become constant if kept as int, so set them as float.
        self.generated_tokens_reset_value = torch.ones(batch_size, max_length, dtype=torch.float32) * pad_token_id
        self.register_buffer(
            "generated_tokens",
            self.generated_tokens_reset_value.clone(),
            persistent=False,
        )

    def reset_state(self, begin_new_generation: torch.Tensor) -> None:
        self.generated_tokens.copy_(
            (1 - begin_new_generation) * self.generated_tokens
            + begin_new_generation * self.generated_tokens_reset_value.to(self.generated_tokens.device)
        )

    def forward(self, input_ids: torch.Tensor, absolute_step: torch.Tensor, **kwargs) -> torch.Tensor:
        # Workaround for generic slice assignment self.generated_tokens[:, self.absolute_step] = tokens
        assert input_ids.shape[-1] == 1
        input_ids_mask = self.input_ids_mask.to(input_ids.device)
        padded_input_ids = torch.nn.functional.pad(input_ids, (0, self.max_length - 1)).int()
        generated_tokens = input_ids_mask * padded_input_ids + (1 - input_ids_mask) * self.generated_tokens.int()

        model_input_ids = torch.index_select(generated_tokens, 1, absolute_step - 1)
        logits = self.model(decoder_input_ids=model_input_ids, **kwargs)
        if hasattr(logits, "logits"):
            logits = logits.logits
        logits = logits.squeeze(1).float()
        next_tokens_scores = self.logits_processor(generated_tokens, logits, absolute_step=absolute_step)

        next_tokens = torch.argmax(next_tokens_scores, dim=-1, keepdim=True).int()
        # If sentence has finished - previous token was EOS - set new tokens to EOS.
        sentence_eos = model_input_ids == self.eos_token_id
        sentence_eos &= (absolute_step - 1) != 0
        sentence_eos = sentence_eos.int()
        next_tokens = sentence_eos * self.eos_token_id + (1 - sentence_eos) * next_tokens
        all_eos = torch.all(next_tokens == self.eos_token_id)

        self.generated_tokens.copy_(
            poptorch.dynamic_update(generated_tokens, next_tokens, 1, absolute_step, 1).float()
        )

        return OnDeviceGenerationModelOutput(generated_tokens=next_tokens, done=all_eos)


class OnDeviceBeamSearch(torch.nn.Module):
    """Based on https://github.com/huggingface/transformers/blob/main/src/transformers/generation/flax_utils.py::_beam_search."""

    def __init__(
        self,
        model: torch.nn.Module,
        batch_size: int,
        num_beams: int,
        max_length: int,
        eos_token_id: int,
        pad_token_id: int,
        logits_processor: LogitsProcessorList,
        length_penalty: Optional[float] = 1.0,
        early_stopping: Optional[Union[bool, str]] = None,
    ):
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.num_beams = num_beams
        self.max_length = max_length
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.logits_processor = logits_processor
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping

        self.input_ids_mask = torch.zeros((batch_size, num_beams, max_length), dtype=torch.int32)
        self.input_ids_mask[:, :, 0] = 1

        # Poptorch buffers become constant if kept as int, so set them as float.
        self.sequences_reset_value = torch.ones(batch_size, num_beams, max_length, dtype=torch.float32) * pad_token_id
        self.running_sequences_reset_value = (
            torch.ones(batch_size, num_beams, max_length, dtype=torch.float32) * pad_token_id
        )
        self.log_probs_reset_value = torch.ones(batch_size, num_beams) * LARGE_NEGATIVE_CONST
        self.running_log_probs_reset_value = torch.tensor([[0.0] + [LARGE_NEGATIVE_CONST] * (num_beams - 1)]).repeat(
            batch_size, 1
        )
        self.is_finished_reset_value = torch.zeros(batch_size, num_beams, dtype=torch.float32)
        self._cached_beam_idx_reset_value = torch.arange(batch_size * num_beams, dtype=torch.float32)
        self.register_buffer("sequences", self.sequences_reset_value.clone(), persistent=False)
        self.register_buffer(
            "running_sequences",
            self.running_sequences_reset_value.clone(),
            persistent=False,
        )
        self.register_buffer("log_probs", self.log_probs_reset_value.clone(), persistent=False)
        self.register_buffer(
            "running_log_probs",
            self.running_log_probs_reset_value.clone(),
            persistent=False,
        )
        self.register_buffer("is_finished", self.is_finished_reset_value.clone(), persistent=False)
        # Holds the beam indices that will be used to permute the KV caches at next time step.
        self.register_buffer(
            "_cached_beam_idx",
            self._cached_beam_idx_reset_value.clone(),
            persistent=False,
        )

    def reset_state(self, begin_new_generation: torch.Tensor) -> None:
        self.sequences.copy_(
            (1 - begin_new_generation) * self.sequences
            + begin_new_generation * self.sequences_reset_value.to(self.sequences.device)
        )
        self.running_sequences.copy_(
            (1 - begin_new_generation) * self.running_sequences
            + begin_new_generation * self.running_sequences_reset_value.to(self.running_sequences.device)
        )
        self.log_probs.copy_(
            (1 - begin_new_generation) * self.log_probs
            + begin_new_generation * self.log_probs_reset_value.to(self.log_probs.device)
        )
        self.running_log_probs.copy_(
            (1 - begin_new_generation) * self.running_log_probs
            + begin_new_generation * self.running_log_probs_reset_value.to(self.running_log_probs.device)
        )
        self.is_finished.copy_(
            (1 - begin_new_generation) * self.is_finished
            + begin_new_generation * self.is_finished_reset_value.to(self.is_finished.device)
        )
        self._cached_beam_idx.copy_(
            (1 - begin_new_generation) * self._cached_beam_idx
            + begin_new_generation * self._cached_beam_idx_reset_value.to(self._cached_beam_idx.device)
        )

    def _flatten_beam_dim(self, tensor, num_beams):
        return tensor.view(self.batch_size * num_beams, *tensor.shape[2:])

    def _unflatten_beam_dim(self, tensor, num_beams):
        return tensor.view(self.batch_size, num_beams, *tensor.shape[1:])

    def _gather_beams(self, tensor, beam_indices, batch_size, old_num_beams, new_num_beams):
        # Ones constraint we have is that index_select must use 1D indices, hence the additional reshaping.
        if tensor.shape[0] != self.batch_size * old_num_beams:
            tensor = self._flatten_beam_dim(tensor, old_num_beams)
        flat_batch_indices = torch.arange(batch_size * new_num_beams) // new_num_beams
        flat_batch_indices = flat_batch_indices * old_num_beams
        flat_beam_indices = beam_indices.view(-1)
        flat_beam_indices = flat_batch_indices + flat_beam_indices
        gathered_tensor = torch.index_select(tensor, 0, flat_beam_indices)
        return self._unflatten_beam_dim(gathered_tensor, new_num_beams)

    def forward(self, input_ids: torch.Tensor, absolute_step: torch.Tensor, **kwargs) -> torch.Tensor:
        # Workaround for generic slice assignment self.running_sequences[:, self.absolute_step] = tokens
        assert input_ids.shape[-1] == 1
        input_ids_mask = self.input_ids_mask.to(input_ids.device)
        padded_input_ids = torch.nn.functional.pad(input_ids, (0, self.max_length - 1))
        padded_input_ids = self._unflatten_beam_dim(padded_input_ids, self.num_beams).int()

        # Since we are constrained to keeping buffers in float, including ones holding tokens which are ints,
        # we do most of the processing as int and cast just before writing.
        sequences = self.sequences.int()
        running_sequences = input_ids_mask * padded_input_ids + (1 - input_ids_mask) * self.running_sequences.int()
        is_finished = self.is_finished.int()

        model_input_ids = torch.index_select(running_sequences, 2, absolute_step - 1)
        model_input_ids = self._flatten_beam_dim(model_input_ids, self.num_beams)

        logits = self.model(decoder_input_ids=model_input_ids, **kwargs)
        if hasattr(logits, "logits"):
            logits = logits.logits
        logits = logits.squeeze(1).float()

        # 2. Compute log probs
        vocab_size = logits.shape[-1]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        log_probs = self.logits_processor(running_sequences, log_probs, absolute_step=absolute_step)
        log_probs = self._unflatten_beam_dim(log_probs, self.num_beams)
        log_probs = log_probs + self.running_log_probs.unsqueeze(-1)
        log_probs = log_probs.view(self.batch_size, self.num_beams * vocab_size)

        # 3. Retrieve top-2*K
        beams_to_keep = 2 * self.num_beams
        topk_log_probs, topk_indices = torch.topk(log_probs, k=beams_to_keep)
        topk_beam_indices = torch.div(topk_indices, vocab_size).int()
        topk_running_sequences = self._gather_beams(
            running_sequences, topk_beam_indices, self.batch_size, self.num_beams, beams_to_keep
        )
        topk_ids = topk_indices % vocab_size
        topk_ids = topk_ids.unsqueeze(-1).int()

        topk_sequences = poptorch.dynamic_update(topk_running_sequences, topk_ids, 2, absolute_step, 1)

        # 4. Check which sequences have ended
        did_topk_just_finish = torch.index_select(topk_sequences, 2, absolute_step).squeeze(2) == self.eos_token_id
        running_topk_log_probs = topk_log_probs + did_topk_just_finish * LARGE_NEGATIVE_CONST

        # 5. Get running sequences scores for next
        next_topk_indices = torch.topk(running_topk_log_probs, k=self.num_beams)[1]

        next_running_sequences = self._gather_beams(
            topk_sequences, next_topk_indices, self.batch_size, beams_to_keep, self.num_beams
        )
        next_running_log_probs = self._gather_beams(
            running_topk_log_probs, next_topk_indices, self.batch_size, beams_to_keep, self.num_beams
        )

        # 6. Process topk logits
        topk_log_probs = topk_log_probs / (absolute_step**self.length_penalty)
        beams_in_batch_are_full = self.is_finished.all(axis=-1, keepdims=True).repeat(
            1, did_topk_just_finish.shape[-1]
        ) & (self.early_stopping is True)
        add_penalty = ~did_topk_just_finish | beams_in_batch_are_full
        topk_log_probs += add_penalty * LARGE_NEGATIVE_CONST

        # 7. Get scores, sequences, is sentence finished for next.
        merged_sequences = torch.cat([sequences, topk_sequences], axis=1)
        merged_log_probs = torch.cat([self.log_probs, topk_log_probs], axis=1)
        merged_is_finished = torch.cat([is_finished, did_topk_just_finish], axis=1)
        topk_merged_indices = torch.topk(merged_log_probs, k=self.num_beams)[1]
        next_sequences = self._gather_beams(
            merged_sequences, topk_merged_indices, self.batch_size, 3 * self.num_beams, self.num_beams
        )
        next_log_probs = self._gather_beams(
            merged_log_probs, topk_merged_indices, self.batch_size, 3 * self.num_beams, self.num_beams
        )
        next_is_finished = self._gather_beams(
            merged_is_finished, topk_merged_indices, self.batch_size, 3 * self.num_beams, self.num_beams
        )

        # 8. Determine the top k beam indices from the original set of all beams.
        next_running_indices = self._gather_beams(
            topk_beam_indices, next_topk_indices, self.batch_size, 2 * self.num_beams, self.num_beams
        )

        # 9. Check for termination.
        not_max_length_yet = absolute_step < self.max_length

        worst_finished_score = torch.where(
            torch.any(next_is_finished, axis=1),
            torch.amin(self.log_probs, axis=1),
            torch.ones(self.batch_size) * LARGE_NEGATIVE_CONST,
        )
        if self.early_stopping == "never" and self.length_penalty > 0.0:
            best_running_score = self.running_log_probs[:, 0] / (self.max_length**self.length_penalty)
        else:
            best_running_score = self.running_log_probs[:, 0] / (absolute_step**self.length_penalty)
        improvement_still_possible = torch.any(best_running_score > worst_finished_score)

        still_open_beam = ~(torch.all(next_is_finished) & (self.early_stopping is True))

        continue_search = not_max_length_yet & still_open_beam & improvement_still_possible

        # 10. Return best beam for each batch and beam indices.
        # Account for the edge-case where there are no finished sequences for a
        # particular batch item. If so, return running sequences for that batch item.
        none_finished = torch.any(next_is_finished, axis=1)
        sequences = torch.where(none_finished[:, None, None], next_sequences, next_running_sequences)
        return_sequences = sequences[:, 0]

        flat_batch_indices = torch.arange(self.batch_size * self.num_beams) // self.num_beams
        flat_batch_indices = flat_batch_indices * self.num_beams
        beam_indices = self._flatten_beam_dim(next_running_indices, self.num_beams)
        beam_indices = beam_indices + flat_batch_indices

        self.sequences.copy_(next_sequences.float())
        self.running_sequences.copy_(next_running_sequences.float())
        self.log_probs.copy_(next_log_probs)
        self.running_log_probs.copy_(next_running_log_probs)
        self.is_finished.copy_(next_is_finished.float())
        self._cached_beam_idx.copy_(beam_indices.float())

        return OnDeviceGenerationModelOutput(
            generated_tokens=return_sequences,
            done=~continue_search,
        )


class OnDeviceGenerationModel(torch.nn.Module):
    """
    A wrapper around a user generation model that effectively runs the entire generation loop
    on device without returning to host after each generated token. Instead, each generated token is stored
    in a torch buffer, and appended to the input at the next time step.
    Suppose the input tensor is of shape [B, C]; B = batch size, C = context length. C is currently
    restricted to C=1. To generate up to a max sequence length of S, device iterations should be set to <= S - C.
    Further, this is only compatible with poptorch.ShardedExecution.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        batch_size: int,
        max_length: int,
        eos_token_id: int,
        pad_token_id: int,
        logits_processor: LogitsProcessorList,
        num_beams: Optional[int] = 1,
        use_cache: Optional[bool] = True,
        length_penalty: Optional[float] = 1.0,
        early_stopping: Optional[bool] = False,
    ):
        super().__init__()

        if not use_cache:
            raise NotImplementedError("On device generation assumes `use_cache=True`.")

        self.max_length = max_length
        self.context_length = 1

        if num_beams == 1:
            self.generation_strategy = OnDeviceGreedySearch(
                model,
                batch_size=batch_size,
                max_length=max_length,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                logits_processor=logits_processor,
            )
        else:
            self.generation_strategy = OnDeviceBeamSearch(
                model,
                batch_size=batch_size,
                num_beams=num_beams,
                max_length=max_length,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                logits_processor=logits_processor,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
            )

        # Poptorch buffers become constant if kept as int, so set them as float.
        self.register_buffer("_generation_step", torch.tensor([0], dtype=torch.int32), persistent=False)

    def _reset_generation_step(self, begin_new_generation: torch.Tensor) -> None:
        generation_step = (1 - begin_new_generation) * self._generation_step
        self._generation_step.copy_(generation_step)

    def forward(self, **kwargs):
        input_ids_key = "decoder_input_ids"
        input_ids = kwargs.pop(input_ids_key, None)
        if input_ids is None:
            input_ids_key = "input_ids"
            input_ids = kwargs.pop(input_ids_key, None)
            if input_ids is None:
                raise ValueError(
                    f"The on device generation model was called with kwargs that are missing both `decoder_input_ids` "
                    "and `input_ids`. Please provide one of these as inputs (default is `decoder_input_ids`)."
                )
        if input_ids.shape[-1] > 1:
            raise ValueError("Context length (input_ids.shape[-1]) > 1 is not supported yet.")

        if generation_step := kwargs.pop("generation_step", None) is not None:
            self._generation_step.copy_(generation_step)

        absolute_step = self._generation_step + self.context_length

        # Make sure generation_step does not go out of bounds.
        self._generation_step.copy_(self._generation_step % self.max_length)

        # Reset on-device state buffers when starting generation anew.
        begin_new_generation = (self._generation_step == 0).int()
        self.generation_strategy.reset_state(begin_new_generation)
        self._reset_generation_step(begin_new_generation)

        outputs = self.generation_strategy(input_ids, absolute_step, **kwargs)
        if not isinstance(outputs, OnDeviceGenerationModelOutput):
            raise TypeError(
                f"Unexpected type {type(outputs)} returned from {self.generation_strategy.__class__.__name__}."
            )

        self._generation_step.copy_(self._generation_step + 1)

        return outputs
