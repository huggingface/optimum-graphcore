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
# RANDOM CHANGE

import math
from typing import Tuple

import torch
import torch.nn as nn

from optimum.utils import logging


logger = logging.get_logger(__name__)


class GroupBertConvolution(nn.Module):
    """
    GroupBERT convolution module. Includes a GLU, group convolution, Swish, LayerNorm and
    output projection.
    """

    def __init__(self, config):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.conv_group_size = config.conv_group_size
        self.conv_kernel_size = config.conv_kernel_size

        self.prenorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.glu = nn.Linear(self.hidden_size, 2 * self.hidden_size)
        self.sigmoid = nn.Sigmoid()

        self.groupconv = nn.Conv1d(
            self.hidden_size,
            self.hidden_size,
            self.conv_kernel_size,
            padding=int((self.conv_kernel_size - 1) / 2),
            groups=int(self.hidden_size / self.conv_group_size),
        )
        self.conv_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.swish = nn.SiLU()
        self.output_projection = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def get_input_mask_from_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        attention_mask = 1.0 - (attention_mask / (-10000.0))
        input_mask = attention_mask[:, 0, :]
        return input_mask

    def forward(
        self, input_tensor: torch.Tensor, attention_mask: torch.Tensor, training: bool = False
    ) -> torch.Tensor:
        # Prenorm
        hidden_states = self.prenorm(input_tensor)

        # Gated Linear Unit
        hidden_states = self.glu(hidden_states)
        gates = hidden_states[..., : self.hidden_size]
        values = hidden_states[..., self.hidden_size :]
        gates = self.sigmoid(gates)
        hidden_states = torch.mul(values, gates)

        # Grouped convolution
        input_mask = self.get_input_mask_from_attention_mask(attention_mask)
        mask = torch.transpose(input_mask, 1, 2)
        conv_input = torch.mul(hidden_states, mask)
        conv_input = torch.transpose(conv_input, -1, -2)
        conv_output = self.groupconv(conv_input)
        conv_output = torch.transpose(conv_output, -1, -2)
        hidden_states = torch.squeeze(conv_output, 2)

        # Norm and activation functiin
        hidden_states = self.conv_norm(hidden_states)
        hidden_states = self.swish(hidden_states)

        # Output projection
        hidden_states = self.output_projection(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return input_tensor + hidden_states
