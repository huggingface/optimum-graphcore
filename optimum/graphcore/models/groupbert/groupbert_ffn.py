# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
from transformers.activations import ACT2FN


logger = logging.get_logger(__name__)


class GroupBertIntermediate(nn.Module):
    """
    GroupBERT FFN intermediate layer is similar to original BERT, but includes
    prenorm.
    """

    def __init__(self, config):
        super().__init__()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class GroupBertOutput(nn.Module):
    """
    GroupBERT FFN output layer is uses grouped matmul to reduce the parameter cound and compensates
    input locality with an output projection layer, similar to attention module.
    """

    def __init__(self, config):
        super().__init__()
        self.ffn_groups = config.ffn_groups

        self.grouped_matmul = nn.Conv1d(
            config.intermediate_size, config.hidden_size, 1, padding=0, groups=self.ffn_groups
        )
        self.dense_output_projection = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # Grouped matmul using conv1d
        hidden_states = torch.transpose(hidden_states, -1, -2)
        hidden_states = self.grouped_matmul(hidden_states)
        hidden_states = torch.transpose(hidden_states, -1, -2)
        # Output projection
        hidden_states = self.dense_output_projection(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states + input_tensor
