# coding=utf-8
# Copyright 2021 The Fairseq Authors and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
These are the same blocks as in the original implementation in transformers,
but with a traceable implementation of LayerDrop.
"""

import torch
from torch.nn import functional as F

from transformers.modeling_outputs import BaseModelOutput
from transformers.models.hubert.modeling_hubert import HubertEncoder, HubertEncoderStableLayerNorm


class IPUHubertEncoder(HubertEncoder):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            # make sure padded tokens output 0
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attention_mask] = 0

            # extend attention_mask
            attention_mask = (1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer(hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            # Modify LayerDrop so it can be statically compiled without eager mode
            if self.config.layerdrop > 0.0:
                dropout_probability = torch.rand(tuple(), device=hidden_states.device)
                skip_the_layer = (
                    torch.tensor(self.training, device=hidden_states.device)
                    & (dropout_probability < self.config.layerdrop)
                ).to(dtype=hidden_states.dtype)
                hidden_states = hidden_states * skip_the_layer + layer_outputs[0] * (1 - skip_the_layer)
            else:
                hidden_states = layer_outputs[0]

            if output_attentions:
                if self.config.layerdrop > 0.0:
                    all_self_attentions = all_self_attentions + ((1 - skip_the_layer) * layer_outputs[1],)
                else:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class IPUHubertEncoderStableLayerNorm(HubertEncoderStableLayerNorm):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            # make sure padded tokens are not attended to
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attention_mask] = 0

            # extend attention_mask
            attention_mask = (1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.dropout(hidden_states)

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer(hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            # Modify LayerDrop so it can be statically compiled without eager mode
            if self.config.layerdrop > 0.0:
                dropout_probability = torch.rand(tuple(), device=hidden_states.device)
                skip_the_layer = (
                    torch.tensor(self.training, device=hidden_states.device)
                    & (dropout_probability < self.config.layerdrop)
                ).to(dtype=hidden_states.dtype)
                hidden_states = hidden_states * skip_the_layer + layer_outputs[0] * (1 - skip_the_layer)
            else:
                hidden_states = layer_outputs[0]

            if output_attentions:
                if self.config.layerdrop > 0.0:
                    all_self_attentions = all_self_attentions + ((1 - skip_the_layer) * layer_outputs[1],)
                else:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
