# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from pathlib import Path
from typing import Tuple

import numpy as np
import poptorch
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import nn


def _int4_npy_decoder(data: NDArray) -> NDArray:
    """Int4 NumPy decoder.

    Args:
        data: Numpy uint16 array to decode.
    Returns:
        Float16 decoded array.
    """
    assert len(data.shape) == 1
    assert len(data) % 4 == 0
    assert data.dtype == np.int16
    N = len(data)
    # Reshape (-1, 4) as the C++ codelet is working on vectors of 4 elements.
    data = data.view(np.uint16).reshape((-1, 4))
    out = np.zeros((N, 4), dtype=np.float16)

    mask0 = np.uint16(0x000F)
    mask1 = np.uint16(0x00F0)

    # Decoding per "block"
    v0 = np.bitwise_and(data, mask0).view(np.float16)
    out[0::4] = v0.astype(np.float32) * (32768 * 512)
    v0 = np.bitwise_and(data, mask1).view(np.float16)
    out[1::4] = v0.astype(np.float32) * (2048 * 512)

    # Shift, and redo the same!
    data = np.right_shift(data, 8)

    v0 = np.bitwise_and(data, mask0).view(np.float16)
    out[2::4] = v0.astype(np.float32) * (32768 * 512)
    v0 = np.bitwise_and(data, mask1).view(np.float16)
    out[3::4] = v0.astype(np.float32) * (2048 * 512)

    out = np.ravel(out)
    return out


def _int4_npy_encoder(data: NDArray) -> NDArray:
    """Int4 encoder.

    Note: this function requires uint16 values already normalized in
    the interval [0, 16), hence leaving to the user to define the
    preferred normalization and rounding mode.

    Args:
        data: Numpy array uint16 in [0, 16)
    Returns:
        Int4 encoded array.
    """
    assert len(data.shape) == 1
    # Int4 blocks + 64bits vectorized instructions!
    assert len(data) % 16 == 0
    assert data.dtype == np.uint16
    assert np.max(data) <= 15
    N = len(data) // 4

    # Reshaping in blocks of 4 to account for C++ vectorization.
    out = np.zeros((N,), dtype=np.uint16)
    out = out.reshape((-1, 4))
    data = data.reshape((-1, 4))

    # Decoding per "block"
    out = np.bitwise_or(out, data[0::4])
    out = np.bitwise_or(out, np.left_shift(data[1::4], 4))
    out = np.bitwise_or(out, np.left_shift(data[2::4], 8))
    out = np.bitwise_or(out, np.left_shift(data[3::4], 12))

    out = np.ravel(out).view(np.int16)
    return out


def group_quantize_compress(t: torch.Tensor, group_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Implementation of 4-bit group-quantized floating point tensors compression
    as described in "FlexGen: High-throughput Generative Inference of Large Language Models
    with a Single GPU" https://arxiv.org/abs/2303.06865
    Input tensors are compressed by grouping along a particular axis, scaling according to the
    min and max value within the group, then rounding to the nearest 4-bit value [0, 16) (or
    0-f if you prefer). These 4-bit values are been packed into a standard datatype
    (uint16 in this case).
    Args:
        t (torch.Tensor): Tensor to compress (float16)
                      (shape=(num_rows, num_groups * group_size * 4), dtype=float16)
        group_size (int): number of elements to compute floating point stats
    Returns:
        (torch.Tensor): 4-bit compressed, packed Tensor
                      (shape=(num_rows, num_groups, group_size), dtype=uint16)
        (torch.Tensor): scaling factor of unpacked uint16 inputs
                      (shape=(num_rows, num_groups, 1), dtype=float16)
        (torch.Tensor): bias term for unpacked uint16 inputs
                      (shape=(num_rows, num_groups, 1), dtype=float16)
    """

    torch_dtype = t.dtype
    t = t.numpy()

    n_rows, n_cols = t.shape
    n_groups = n_cols // group_size

    if not np.isfinite(t).all():
        raise ValueError("Cannot compress t which contains non-finite values")

    t_grouped = t.reshape(n_rows, n_groups, group_size)
    t_max = t_grouped.max(-1, keepdims=True).astype(np.float16)
    t_min = t_grouped.min(-1, keepdims=True).astype(np.float16)

    t_scale = (t_max - t_min) / (2**4 - 1)
    t_bias = t_min

    t_quantized = np.round((t_grouped - t_bias) / t_scale).astype(np.uint16)
    t_packed = _int4_npy_encoder(t_quantized.flatten())
    t_packed = t_packed.reshape(n_rows, n_groups, group_size // 4)
    return (
        torch.tensor(t_packed.view(np.int16)),
        torch.tensor(t_scale, dtype=torch_dtype),
        torch.tensor(t_bias, dtype=torch_dtype),
    )


def group_quantize_decompress(
    t_packed: torch.Tensor, group_scale: torch.Tensor, group_bias: torch.Tensor, dtype: torch.Type = torch.float16
) -> torch.Tensor:
    """
    IPU PopTorch implementation of 4-bit group-quantized floating point tensors decompression
    as described in "FlexGen: High-throughput Generative Inference of Large Language Models
    with a Single GPU" https://arxiv.org/abs/2303.06865
    Assumes that input tensors have been compressed by grouping along a particular axis,
    scaling according to the min and max value within the group, then rounding to the
    nearest 4-bit value [0, 16) (or 0-f if you prefer). These 4-bit values have then been
    packed into a standard datatype (uint16 in this case).
    To decompress, the packed tensor must be unpacked and rescaled according to the min
    and max value. For efficiency, the min and max values are passed here as scale and
    bias terms.
    Args:
        t_packed    (torch.Tensor): 4-bit compressed, packed Tensor
                          (shape=(num_rows, num_groups, num_group_ids), dtype=uint16)
        group_scale (torch.Tensor): scaling factor of unpacked uint16 inputs
                          (shape=(num_rows, num_groups, 1), dtype=float16)
        group_bias (torch.Tensor): bias term for unpacked uint16 inputs
                          (shape=(num_rows, num_groups, 1), dtype=float16)
    Returns:
        torch.Tensor: Decompressed result (float16)
                      (shape=(num_rows, num_groups * num_group_ids * 4), dtype=float16)
    """

    if poptorch.isRunningOnIpu():
        n_rows, n_groups, n_ids = t_packed.shape
        example_output = torch.empty(n_rows, n_groups * n_ids * 4, dtype=dtype)
        root_path = str(Path(__file__).parent.parent / "custom_ops" / "group_quantize_decompress")
        o = poptorch.custom_op(
            [t_packed, group_scale, group_bias],
            "GroupQuantizeDecompress",
            "optimum.custom_ops",
            1,
            example_outputs=[example_output],
            attributes={"root_path": root_path},
        )
        return o[0]

    else:
        n_rows, n_groups, n_group_ids = t_packed.shape
        t_packed = t_packed.numpy().flatten()
        t_unpacked = torch.tensor(_int4_npy_decoder(t_packed).reshape(n_rows, n_groups, n_group_ids * 4))
        t_scaled = t_unpacked * group_scale + group_bias  # rescale
        return t_scaled.reshape(n_rows, n_groups * n_group_ids * 4)


class GroupQuantLinear(nn.Module):
    def __init__(self, in_features, out_features, w_packed, w_scale, w_bias, bias=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w_packed = nn.Parameter(w_packed, requires_grad=False)
        self.w_scale = nn.Parameter(w_scale, requires_grad=False)
        self.w_bias = nn.Parameter(w_bias, requires_grad=False)
        self.bias = bias  # Bias is uncompressed

    @classmethod
    def from_model(cls, linear: nn.Linear, num_groups: int):
        w = linear.weight.data
        bias = linear.bias
        w_packed, w_scale, w_bias = group_quantize_compress(w, num_groups)
        return cls(linear.in_features, linear.out_features, w_packed, w_scale, w_bias, bias)

    def forward(self, input):
        weight = group_quantize_decompress(self.w_packed.data, self.w_scale.data, self.w_bias.data, dtype=input.dtype)
        return F.linear(input, weight, self.bias)
