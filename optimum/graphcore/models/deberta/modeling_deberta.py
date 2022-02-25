# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
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

import math

import torch
import torch.nn as nn

import poptorch
from optimum.utils import logging
from transformers import DebertaForQuestionAnswering, DebertaForSequenceClassification, DebertaForTokenClassification
from transformers.models.deberta.modeling_deberta import (
    DebertaEncoder,
    DebertaLayerNorm,
    DisentangledSelfAttention,
    StableDropout,
    build_relative_position,
)

from ...modeling_utils import PipelineMixin, _get_layer_ipu, outline_attribute, recomputation_checkpoint, register


logger = logging.get_logger(__name__)


class XSoftmax(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, input, mask):
        """ """
        rmask = ~(mask.bool())
        output = self.masked_fill_approx(input, rmask, -100000)
        output = torch.softmax(output, self.dim)
        output = self.masked_fill_approx(output, rmask, 0)
        return output

    def masked_fill_approx(self, input, mask, value):
        mask_int = mask.to(torch.int)
        mask_ = value * mask_int
        output = input + mask_
        return output


def _get_rel_embedding(self):
    return self.rel_embeddings.weight + 0.0 if self.relative_attention else None


class IPUDisentangledSelfAttention(DisentangledSelfAttention):
    """
    Disentangled self-attention module

    Parameters:
        config (`str`):
            A model config class instance with the configuration to build a new model. The schema is similar to
            *BertConfig*, for more details, please refer [`DebertaConfig`]

    """

    def __init__(self, config):
        super().__init__(config)
        self.xsoftmax = XSoftmax(-1)

    def index_select_gather(self, t, pos):
        """Uses `index_select` function to gather indices on the last two dimension of the attention tensor
        shaped [bs, num_attn_heads, seq_len, 2*seq_len]
        """
        size = t.size()
        seq_len = size[-2]
        indices = ((torch.arange(0, seq_len) * 2 * seq_len).unsqueeze(1) + pos).reshape(-1)
        tf = t.reshape(size[0], size[1], -1)

        out = []
        chunk_size = indices.shape[-1] // 4
        for i in range(0, indices.shape[-1], chunk_size):
            out.append(torch.index_select(tf, -1, indices[i : i + chunk_size]))
        out = torch.cat(out, dim=-1)
        return out.reshape(size[0], size[1], seq_len, seq_len)

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
    ):
        """
        Call the module

        Args:
            hidden_states (`torch.FloatTensor`):
                Input states to the module usually the output from previous layer, it will be the Q,K and V in
                *Attention(Q,K,V)*

            attention_mask (`torch.ByteTensor`):
                An attention mask matrix of shape [*B*, *N*, *N*] where *B* is the batch size, *N* is the maximum
                sequence length in which element [i,j] = *1* means the *i* th token in the input can attend to the *j*
                th token.

            output_attentions (`bool`, optional):
                Whether return the attention matrix.

            query_states (`torch.FloatTensor`, optional):
                The *Q* state in *Attention(Q,K,V)*.

            relative_pos (`torch.LongTensor`):
                The relative position encoding between the tokens in the sequence. It's of shape [*B*, *N*, *N*] with
                values ranging in [*-max_relative_positions*, *max_relative_positions*].

            rel_embeddings (`torch.FloatTensor`):
                The embedding of relative distances. It's a tensor of shape [\\(2 \\times
                \\text{max_relative_positions}\\), *hidden_size*].


        """
        if query_states is None:
            qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
            query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
        else:

            def linear(w, b, x):
                if b is not None:
                    return torch.matmul(x, w.t()) + b.t()
                else:
                    return torch.matmul(x, w.t())  # + b.t()

            ws = self.in_proj.weight.chunk(self.num_attention_heads * 3, dim=0)
            qkvw = [torch.cat([ws[i * 3 + k] for i in range(self.num_attention_heads)], dim=0) for k in range(3)]
            qkvb = [None] * 3

            q = linear(qkvw[0], qkvb[0], query_states)
            k, v = [linear(qkvw[i], qkvb[i], hidden_states) for i in range(1, 3)]
            query_layer, key_layer, value_layer = [self.transpose_for_scores(x) for x in [q, k, v]]

        query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
        value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])

        rel_att = None
        # Take the dot product between "query" and "key" to get the raw attention scores.
        scale_factor = 1 + len(self.pos_att_type)
        scale = math.sqrt(query_layer.size(-1) * scale_factor)
        query_layer = query_layer / scale
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        if self.relative_attention:
            rel_embeddings = self.pos_dropout(rel_embeddings)
            rel_att = self.disentangled_att_bias(query_layer, key_layer, relative_pos, rel_embeddings, scale_factor)

        if rel_att is not None:
            attention_scores = attention_scores + rel_att

        # bxhxlxd
        if self.talking_head:
            attention_scores = self.head_logits_proj(attention_scores.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        attention_probs = self.xsoftmax(attention_scores, attention_mask)
        attention_probs = self.dropout(attention_probs)
        if self.talking_head:
            attention_probs = self.head_weights_proj(attention_probs.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (-1,)
        context_layer = context_layer.view(*new_context_layer_shape)
        if output_attentions:
            return (context_layer, attention_probs)
        else:
            return context_layer

    def disentangled_att_bias(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor):
        if relative_pos is None:
            q = query_layer.size(-2)
            relative_pos = build_relative_position(q, key_layer.size(-2), query_layer.device)
        if relative_pos.dim() == 2:
            relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
        elif relative_pos.dim() == 3:
            relative_pos = relative_pos.unsqueeze(1)
        # bxhxqxk
        elif relative_pos.dim() != 4:
            raise ValueError(f"Relative position ids must be of dim 2 or 3 or 4. {relative_pos.dim()}")

        att_span = min(max(query_layer.size(-2), key_layer.size(-2)), self.max_relative_positions)
        relative_pos = relative_pos.long().to(query_layer.device)
        rel_embeddings = rel_embeddings[
            self.max_relative_positions - att_span : self.max_relative_positions + att_span, :
        ].unsqueeze(0)

        score = 0

        # content->position
        if "c2p" in self.pos_att_type:
            pos_key_layer = self.pos_proj(rel_embeddings)
            pos_key_layer = self.transpose_for_scores(pos_key_layer)
            c2p_att = torch.matmul(query_layer, pos_key_layer.transpose(-1, -2))
            c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span * 2 - 1)
            # index = c2p_dynamic_expand(c2p_pos, query_layer, relative_pos)
            # c2p_att = self.c2p_gather(c2p_att)
            # c2p_att = self.p2c_gather(c2p_att)
            c2p_att = self.index_select_gather(c2p_att, c2p_pos)
            # c2p_att = torch.gather(c2p_att, dim=-1, index=index)
            # assert (c2p_att == c2p_att2).all(), "c2p not equal"
            score += c2p_att

        # position->content
        if "p2c" in self.pos_att_type:
            pos_query_layer = self.pos_q_proj(rel_embeddings)
            pos_query_layer = self.transpose_for_scores(pos_query_layer)
            pos_query_layer /= math.sqrt(pos_query_layer.size(-1) * scale_factor)
            if query_layer.size(-2) != key_layer.size(-2):
                r_pos = build_relative_position(key_layer.size(-2), key_layer.size(-2), query_layer.device)
            else:
                r_pos = relative_pos
            p2c_pos = torch.clamp(-r_pos + att_span, 0, att_span * 2 - 1)
            p2c_att = torch.matmul(key_layer, pos_query_layer.transpose(-1, -2))
            # index = p2c_dynamic_expand(p2c_pos, query_layer, key_layer)
            # import ipdb; ipdb.set_trace()
            # p2c_att = self.p2c_gather(p2c_att).transpose(-1, -2)
            p2c_att = self.index_select_gather(p2c_att, p2c_pos).transpose(-1, -2)
            # p2c_att = torch.gather(
            #     p2c_att, dim=-1, index=p2c_dynamic_expand(p2c_pos, query_layer, key_layer)
            # ).transpose(-1, -2)
            # assert (p2c_att == p2c_att2).all(), f"p2c not equal {p2c_pos}"

            # TODO When does path this occur and what is the value of index in this case?
            if query_layer.size(-2) != key_layer.size(-2):
                pos_index = relative_pos[:, :, :, 0].unsqueeze(-1)
                # index = pos_dynamic_expand(pos_index, p2c_att, key_layer)
                p2c_att = self.index_select_gather(p2c_att, pos_index)
            score += p2c_att

        return score


class DebertaPipelineMixin(PipelineMixin):
    def change_modules_for_ipu(self, restore: bool):
        for mod in self.modules():
            if isinstance(mod, DisentangledSelfAttention):
                mod.__class__ = DisentangledSelfAttention if restore else IPUDisentangledSelfAttention
                if restore:
                    del mod.xsoftmax
                else:
                    mod.xsoftmax = XSoftmax(-1)
            if isinstance(mod, StableDropout):
                mod.__class__ = nn.Dropout
                mod.p = mod.drop_prob
                mod.inplace = False
            if isinstance(mod, DebertaLayerNorm):
                mod.forward = DebertaLayerNorm.forward if restore else poptorch.autocast(enabled=True)(mod.forward)

            if isinstance(mod, DebertaEncoder):
                func = DebertaEncoder.get_rel_embedding if restore else _get_rel_embedding
                mod.get_rel_embedding = func.__get__(mod, DebertaEncoder)

    def parallelize(self):
        """
        Transform the model to run in an IPU pipeline.
        - Adds pipeline stages to the model
        - (If enabled) Replaces the word embedding with a SerializedEmbedding
        - Replaces several modules with IPU compatible counterparts
        - Adds recomputation checkpoints
        """
        self._hooks = []
        layer_ipu = _get_layer_ipu(self.config.layers_per_ipu)

        logger.info("-------------------- Device Allocation --------------------")
        logger.info("Embedding  --> IPU 0")
        # if self.config.embedding_serialization_factor > 1:
        #     self.deberta.embeddings.word_embeddings = SerializedEmbedding(
        #         self.deberta.embeddings.word_embeddings, self.config.embedding_serialization_factor
        #     )

        self.change_modules_for_ipu(False)

        self.deberta.embeddings = poptorch.BeginBlock(self.deberta.embeddings, "Embedding", ipu_id=0)
        hs = outline_attribute(self.deberta.embeddings.LayerNorm, "embedding")
        self._hooks.extend(hs)

        self.deberta.encoder = poptorch.BeginBlock(self.deberta.encoder, ipu_id=0)
        self.deberta.encoder.rel_embeddings = poptorch.BeginBlock(self.deberta.encoder.rel_embeddings, ipu_id=0)

        for index, layer in enumerate(self.deberta.encoder.layer):
            ipu = layer_ipu[index]
            if self.config.recompute_checkpoint_every_layer and index != self.config.num_hidden_layers - 1:
                h = recomputation_checkpoint(layer)
                self._hooks.append(h)
            self.deberta.encoder.layer[index] = poptorch.BeginBlock(layer, f"Encoder{index}", ipu_id=ipu)
            logger.info(f"Encoder {index:<2} --> IPU {ipu}")
        return self

    def deparallelize(self):
        """
        Undo the changes to the model done by `parallelize`.
        You should call this before doing `save_pretrained` so that the `model.state_dict` is
        compatible with the original model.
        """
        super().deparallelize()
        self.change_modules_for_ipu(True)
        # Deserialize the serialized word embedding
        if self.config.embedding_serialization_factor > 1:
            self.deberta.embeddings.word_embeddings = self.deberta.embeddings.word_embeddings.deserialize()
        return self


@register(DebertaForSequenceClassification)
class PipelinedDebertaForSequenceClassification(DebertaEncoder, DebertaPipelineMixin):
    """
    DebertaForSequenceClassification transformed to run in an IPU pipeline.

    Recommended usage:
    ```
    model = PipelinedDebertaForSequenceClassification(config).parallelize().half()
    ```
    """

    def parallelize(self):
        super().parallelize()
        last_ipu = self.config.ipus_per_replica - 1
        logger.info(f"Classifier Output --> IPU {last_ipu}")
        self.classifier = poptorch.BeginBlock(self.classifier, "Classifier Output", ipu_id=last_ipu)
        logger.info("-----------------------------------------------------------")
        return self

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            return_dict=False,
        )


@register(DebertaForTokenClassification)
class PipelinedDebertaForTokenClassification(DebertaForTokenClassification, DebertaPipelineMixin):
    """
    DebertaForTokenClassification transformed to run in an IPU pipeline.

    Recommended usage:
    ```
    model = PipelinedDebertaForTokenClassification(config).parallelize().half()
    ```
    """

    def parallelize(self):
        super().parallelize()
        last_ipu = self.config.ipus_per_replica - 1
        logger.info(f"Classifier Output --> IPU {last_ipu}")
        self.classifier = poptorch.BeginBlock(self.classifier, "Classifier Output", ipu_id=last_ipu)
        logger.info("-----------------------------------------------------------")
        return self

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            return_dict=False,
        )


@register(DebertaForQuestionAnswering)
class PipelinedDebertaForQuestionAnswering(DebertaForQuestionAnswering, DebertaPipelineMixin):
    # def __init__(self, config):
    #     super().__init__(config)

    #     # Replace the DisentangledSelfAttention with IPU version
    #     # for layer in self.deberta.encoder.layer:
    #     #     self_attn = IPUDisentangledSelfAttention(self.config)
    #     #     self_attn.load_state_dict(layer.attention.self.state_dict())
    #     #     layer.attention.self = self_attn

    #     # for mod in self.modules():
    #     #     # if isinstance(mod, StableDropout):
    #     #     #     mod.__class__ = nn.Dropout
    #     #     #     mod.p = mod.drop_prob
    #     #     #     mod.inplace = False
    #     #     if isinstance(mod, DebertaLayerNorm):
    #     #         mod.__class__ = nn.LayerNorm
    #     #         mod.normalized_shape = mod.weight.shape
    #     #         mod.eps = mod.variance_epsilon

    def parallelize(self):
        super().parallelize()
        last_ipu = self.config.ipus_per_replica - 1
        logger.info(f"QA Outputs --> IPU {last_ipu}")
        self.qa_outputs = poptorch.BeginBlock(self.qa_outputs, "QA Outputs", ipu_id=last_ipu)
        logger.info("-----------------------------------------------------------")
        return self

    def forward(self, input_ids, attention_mask, token_type_ids, start_positions=None, end_positions=None):
        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            start_positions=start_positions,
            end_positions=end_positions,
            return_dict=False,
        )
        if start_positions is not None and end_positions is not None:
            output = (poptorch.identity_loss(output[0], reduction="none"),) + output[1:]
        return output
