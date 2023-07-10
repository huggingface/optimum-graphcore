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

import copy
from typing import Callable, Optional, Tuple

import poptorch
import torch

from .utils import assert_poptorch_supports_cond


FLOAT16_LIMIT = 1e4


class IPUAttentionMixin:
    """
    The aim of this class is to provide common, model-agnostic functionality such as KV caching and attention
    serialization to transformer attention layers.

    The intended usage is best demonstrated with an existing example, Whisper. There are roughly two steps:
    1. subclass the parent attention layer to inject this mixin, for example, `class IPUWhisperAttention(WhisperAttention, IPUAttentionMixin)`
    and use the `add_to_kv_cache` and `update_attention_mask` methods to add the KV values at the current time
    step to the cache, or `serialized_attention` to serialize attention across the batch or sequence dimensions.

    2. replace the existing attention layers with above via the provided class method `from_model`, e.g.
    `decoder_layer.self_attn = IPUWhisperAttention.from_model(decoder_layer.self_attn, use_cache=True, **kwargs)`.
    """

    _kv_cache_initialized: bool = False
    _cross_kv_cache_initialized: bool = False
    _num_beams: int = 1
    _batch_serialization_factor: int = 1
    _sequence_serialization_factor: int = 1

    @property
    def kv_cache_initialized(self) -> bool:
        return self._kv_cache_initialized

    @property
    def cross_kv_cache_initialized(self) -> bool:
        return self._cross_kv_cache_initialized

    def _create_kv_cache(self, cache_shape: Tuple[int], dtype: torch.dtype, num_beams=1):
        self.register_buffer("_generation_step", torch.tensor([0], dtype=torch.int32), persistent=False)
        self.register_buffer("_k_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False)
        self.register_buffer("_v_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False)
        if num_beams > 1:
            self.register_buffer("_beam_idx", torch.arange(cache_shape[0], dtype=torch.int32), persistent=False)
        self._num_beams = num_beams
        self._kv_cache_initialized = True

    def _delete_kv_cache(self):
        if not self._kv_cache_initialized:
            return

        del self._generation_step
        del self._k_cache
        del self._v_cache
        if hasattr(self, "_beam_idx"):
            del self._beam_idx
        del self._num_beams
        del self._kv_cache_initialized

    def _create_cross_kv_cache(self, cache_shape: Tuple[int], dtype: torch.dtype, num_beams=1):
        if not hasattr(self, "_generation_step"):
            self.register_buffer("_generation_step", torch.tensor([0], dtype=torch.int32), persistent=False)
        self.register_buffer("_cross_k_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False)
        self.register_buffer("_cross_v_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False)
        if num_beams > 1 and not hasattr(self, "_beam_idx"):
            self.register_buffer("_beam_idx", torch.arange(cache_shape[0], dtype=torch.int32), persistent=False)
        self._cross_kv_cache_initialized = True

    def _delete_cross_kv_cache(self):
        if not self._cross_kv_cache_initialized:
            return

        if hasattr(self, "_generation_step"):
            del self._generation_step
        del self._cross_k_cache
        del self._cross_v_cache
        if hasattr(self, "_beam_idx"):
            del self._beam_idx
        del self._cross_kv_cache_initialized

    @classmethod
    def from_model(
        cls,
        attention_layer: torch.nn.Module,
        use_cache: bool = False,
        batch_size: int = 1,
        max_length: int = 128,
        num_beams: int = 1,
        num_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        dtype: torch.dtype = torch.float16,
        batch_serialization_factor: int = 1,
        sequence_serialization_factor: int = 1,
        use_cross_cache: bool = False,
        encoder_max_length: int = 128,
    ):
        """
        Returns an instance of the provided `attention_layer` with functionality provided by `IPUAttentionMixin`.

        If `use_cache=True`, instantiates the self-attention KV caches, each of shape
        `(batch_size * num_beams, num_heads, max_length, head_dim)`.

        If `batch_serialization_factor > 1` or `sequence_serialization_factor > 1`, attention will be serialized
        along the batch or sequence dimension respectively.
        """
        clone = copy.deepcopy(attention_layer)
        clone.__class__ = cls

        def infer_attribute_from_layer(attr: str):
            err_msg = (
                f"Attempting to replace attention class `{attention_layer.__class__.__name__}` with `{cls.__name__}`."
                f" However unable to infer `{{0}}` from `{attention_layer.__class__.__name__}`."
                " Provide the `{0}` argument to `IPUAttentionMixin.from_model`."
            )
            try:
                value = getattr(clone, attr)
                return value
            except AttributeError as e:
                raise AttributeError(err_msg.format(attr)) from e

        if use_cache or use_cross_cache:
            num_heads = infer_attribute_from_layer("num_heads") if num_heads is None else num_heads
            head_dim = infer_attribute_from_layer("head_dim") if head_dim is None else head_dim

        if use_cache:
            clone._create_kv_cache(
                (batch_size * num_beams, num_heads, max_length, head_dim),
                dtype=dtype,
                num_beams=num_beams,
            )

        if use_cross_cache:
            assert_poptorch_supports_cond(
                context="Cross-attention KV caching has been enabled with `use_cross_cache=True`."
            )
            clone._create_cross_kv_cache(
                (batch_size * num_beams, num_heads, encoder_max_length, head_dim),
                dtype=dtype,
                num_beams=num_beams,
            )

        if batch_serialization_factor < 1 or sequence_serialization_factor < 1:
            raise ValueError(
                "`batch_serialization_factor` and `sequence_serialization_factor` must be > 0 if provided."
            )
        elif batch_serialization_factor > 1 and sequence_serialization_factor > 1:
            raise ValueError(
                "If serializing attention, only one of `batch_serialization_factor` "
                "and `sequence_serialization_factor` should be greater than 1, not both."
            )
        elif batch_serialization_factor > 1 or sequence_serialization_factor > 1:
            if use_cache:
                raise ValueError("Attention serialization is redundant when KV caching is enabled.")

            clone._batch_serialization_factor = batch_serialization_factor
            clone._sequence_serialization_factor = sequence_serialization_factor

        return clone

    def to_model(self, cls) -> torch.nn.Module:
        """
        Returns an instance of the `attention_layer` provided to `from_model` with functionality provided by `IPUAttentionMixin` removed.
        """
        self._delete_kv_cache()
        self._delete_cross_kv_cache()
        self._delete_serialization_factors()

        original = copy.deepcopy(self)
        original.__class__ = cls
        return original

    def add_to_kv_cache(self, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Copies the key-value pair into their corresponding key-value caches.

        Args:
            key (`torch.FloatTensor`): key tensor of shape `(batch_size * num_beams, num_heads, 1, head_dim)`.
            value (`torch.FloatTensor`): value tensor of shape `(batch_size * num_beams, num_heads, 1, head_dim)`.
        """
        if not self.kv_cache_initialized:
            raise ValueError(
                f"{self.__class__.__name__} assumes that self-attention has KV caching enabled. "
                f"Please instantiate using `{self.__class__.__name__}.from_model()` so the KV "
                "cache can be created."
            )

        if self.training:
            raise RuntimeError("KV caching is currently only supported for inference.")

        expected_key_shape, expected_value_shape = list(self._k_cache.shape), list(self._v_cache.shape)
        expected_key_shape[-2] = 1
        expected_value_shape[-2] = 1
        if list(key.shape) != expected_key_shape:
            raise ValueError(f"Expected key shape {expected_key_shape}, received {list(key.shape)}.")
        if list(value.shape) != expected_value_shape:
            raise ValueError(f"Expected value shape {expected_value_shape}, received {list(value.shape)}.")

        # For now assume that generation will always start from step 0.
        reset_kv_cache = self._generation_step == 0
        self._k_cache *= 1 - reset_kv_cache.to(self._k_cache.dtype)
        self._v_cache *= 1 - reset_kv_cache.to(self._v_cache.dtype)

        if hasattr(self, "_beam_idx"):
            # For beam search, permute the cache since inputs are permuted on host.
            _k_cache = torch.index_select(self._k_cache, 0, self._beam_idx)
            _v_cache = torch.index_select(self._v_cache, 0, self._beam_idx)
            self._k_cache.copy_(_k_cache)
            self._v_cache.copy_(_v_cache)

        # Dynamic update leads to uneven tile placement, and scatter leads to large re-arrangements,
        # so use a brute force matmul approach which empirically seems best for now.
        bsz, heads, src_len, head_dim = self._k_cache.shape
        mm_mask = (torch.arange(src_len) == self._generation_step).view(src_len, 1)
        _key = torch.matmul(mm_mask.to(key.dtype), key.view(bsz * heads, 1, head_dim))
        _value = torch.matmul(mm_mask.to(value.dtype), value.view(bsz * heads, 1, head_dim))
        self._k_cache += _key.view(self._k_cache.shape)
        self._v_cache += _value.view(self._v_cache.shape)

        return self._k_cache, self._v_cache

    def update_attention_mask(self, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Creates a default attention mask intended for use with KV caches. It masks up to and including the current generation step,
        marking the point up to which the caches have been populated.
        """
        bsz, _, src_len, _ = self._k_cache.shape
        mask = torch.full((1, src_len), -FLOAT16_LIMIT)
        mask_cond = torch.arange(src_len).view(1, src_len)
        mask.masked_fill_(mask_cond < self._generation_step + 1, 0)
        mask = mask.to(self._k_cache.dtype)
        mask = mask.expand(bsz, 1, 1, src_len)

        if attention_mask is not None:
            if attention_mask.size() != mask.size():
                raise ValueError(
                    f"Attention mask does not match expected KV cache mask dimensions. "
                    f"Received: {attention_mask.size()}, expected {mask.size()}."
                )
            mask = mask + attention_mask

        return mask

    def add_to_cross_kv_cache(
        self,
        cross_input: torch.Tensor,
        key_fn: Callable[[torch.Tensor], torch.Tensor],
        value_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.cross_kv_cache_initialized:
            raise ValueError(
                f"{self.__class__.__name__} assumes that cross-attention has cross KV caching enabled. "
                f"Please instantiate using `{self.__class__.__name__}.from_model()` so the cross KV "
                "cache can be created."
            )

        if self.training:
            raise RuntimeError("Cross KV caching is currently only supported for inference.")

        assert_poptorch_supports_cond(
            context="Cross-attention KV caching has been enabled with `use_cross_cache=True`."
        )

        # For now assume that generation will always start from step 0.
        reset_kv_cache = self._generation_step == 0
        self._cross_k_cache *= 1 - reset_kv_cache.to(self._cross_k_cache.dtype)
        self._cross_v_cache *= 1 - reset_kv_cache.to(self._cross_v_cache.dtype)

        if hasattr(self, "_beam_idx"):
            # For beam search, permute the cache since inputs are permuted on host.
            _cross_k_cache = torch.index_select(self._cross_k_cache, 0, self._beam_idx)
            _cross_v_cache = torch.index_select(self._cross_v_cache, 0, self._beam_idx)
            self._cross_k_cache.copy_(_cross_k_cache)
            self._cross_v_cache.copy_(_cross_v_cache)

        def then_k_body(x):
            return key_fn(x)

        def else_k_body(_):
            return self._cross_k_cache

        def then_v_body(x):
            return value_fn(x)

        def else_v_body(_):
            return self._cross_v_cache

        self._cross_k_cache.copy_(
            poptorch.cond(reset_kv_cache, then_k_body, [cross_input], else_k_body, [cross_input])[0]
        )
        self._cross_v_cache.copy_(
            poptorch.cond(reset_kv_cache, then_v_body, [cross_input], else_v_body, [cross_input])[0]
        )

        return self._cross_k_cache, self._cross_v_cache

    @property
    def is_attention_serialized(self) -> bool:
        return self._batch_serialization_factor > 1 or self._sequence_serialization_factor > 1

    def _delete_serialization_factors(self):
        if not self.is_attention_serialized:
            return

        del self._batch_serialization_factor
        del self._sequence_serialization_factor

    def serialized_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: Optional[float] = 1.0,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Serializes the attention operation either across the batch (if `batch_serialization_factor > 1`)
        or the sequence (if `sequence_serialization_factor > 1`) dimensions to reduce peak memory usage.
        NB: if serializing across the batch, this will include `num_heads` wherein we expect the leading
        dimension to be of size `batch_size * num_heads`.
        """
        if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
            raise ValueError(
                "Expected query, key, value all to be 3D, which we will interpret "
                "as (batch_size * num_heads, sequence_length, head_dim). Received "
                f"{query.shape}, {key.shape}, {value.shape}."
            )

        if self._batch_serialization_factor > 1:
            return self._batch_serialized_attention(
                query, key, value, scale, attention_mask, self._batch_serialization_factor
            )
        elif self._sequence_serialization_factor > 1:
            return self._sequence_serialized_attention(
                query, key, value, scale, attention_mask, self._sequence_serialization_factor
            )
        else:
            raise ValueError(
                "Attempting to serialize attention but neither serialization factor is >1. "
                "To serialize attention, please provide either a `batch_serialization_factor` or "
                "`sequence_serialization_factor` kwarg to `IPUWhisperAttention.from_model` with "
                "values greater than 1."
            )

    def _batch_serialized_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: Optional[float] = 1.0,
        attention_mask: Optional[torch.Tensor] = None,
        serialization_factor: Optional[int] = 1,
    ) -> torch.Tensor:
        if query.shape[0] % serialization_factor != 0:
            raise ValueError(
                f"Cannot evenly divide query batch dim: {query.shape[0]} by `serialization_factor`: {serialization_factor}."
            )
        slice_size = query.shape[0] // serialization_factor

        hidden_states = []
        key = key.transpose(1, 2)
        for i in range(serialization_factor):
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size

            attn_slice = torch.matmul(query[start_idx:end_idx], key[start_idx:end_idx]) * scale
            if attention_mask is not None:
                attn_slice = attn_slice + attention_mask[start_idx:end_idx]
            attn_slice = attn_slice.softmax(dim=-1)
            attn_slice = torch.matmul(attn_slice, value[start_idx:end_idx])

            hidden_states.append(attn_slice)

        hidden_states = torch.cat(hidden_states, dim=0)
        return hidden_states

    def _sequence_serialized_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: Optional[float] = 1.0,
        attention_mask: Optional[torch.Tensor] = None,
        serialization_factor: Optional[int] = 1,
    ) -> torch.Tensor:
        if query.shape[1] % serialization_factor != 0:
            raise ValueError(
                f"Cannot evenly divide query sequence dim: {query.shape[1]} by `serialization_factor`: {serialization_factor}."
            )
        slice_size = query.shape[1] // serialization_factor

        hidden_states = []
        key = key.transpose(1, 2)
        for i in range(serialization_factor):
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size

            attn_slice = torch.matmul(query[:, start_idx:end_idx], key) * scale
            if attention_mask is not None:
                attn_slice = attn_slice + attention_mask[:, start_idx:end_idx]
            attn_slice = attn_slice.softmax(dim=-1)
            attn_slice = torch.matmul(attn_slice, value)

            hidden_states.append(attn_slice)

        hidden_states = torch.cat(hidden_states, dim=1)
        return hidden_states
