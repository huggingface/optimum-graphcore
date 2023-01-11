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
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import poptorch
from optimum.utils import logging
from transformers import (
    DebertaForMaskedLM,
    DebertaForQuestionAnswering,
    DebertaForSequenceClassification,
    DebertaForTokenClassification,
)
from transformers.modeling_outputs import MaskedLMOutput, QuestionAnsweringModelOutput
from transformers.models.deberta.modeling_deberta import (
    DebertaEncoder,
    DisentangledSelfAttention,
    StableDropout,
    build_relative_position,
)

from ...modeling_utils import (
    OnehotGather,
    PipelineMixin,
    SerializedEmbedding,
    SerializedLinear,
    get_layer_ipu,
    outline_attribute,
    recomputation_checkpoint,
    register,
)


logger = logging.get_logger(__name__)


class FastGatherLastDim(nn.Module):
    """
    Custom Op that does a faster specialised version of `gather`
    on the last dimension of a tensor.
    """

    def __init__(self):
        super().__init__()

    def forward(self, data, idx, target=None):
        if poptorch.isRunningOnIpu():
            if target is None:
                target = torch.zeros_like(idx).to(data.dtype)
            else:
                target = target.type_as(data)

            target.requires_grad_()
            o = poptorch.custom_op(
                [data, idx],
                "FastGatherLastDim",
                "poptorch.custom_ops",
                1,
                example_outputs=[target],
                attributes={"axis": -1},
            )
            return o[0]
        else:
            return torch.gather(data, -1, idx)


gather_last_dim = FastGatherLastDim()


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
            index = c2p_pos.expand(
                [query_layer.size(0), query_layer.size(1), query_layer.size(2), relative_pos.size(-1)]
            )
            c2p_att = gather_last_dim(c2p_att, index)
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
            index = p2c_pos.expand([query_layer.size(0), query_layer.size(1), key_layer.size(-2), key_layer.size(-2)])
            p2c_att = torch.matmul(key_layer, pos_query_layer.transpose(-1, -2))
            p2c_att = gather_last_dim(p2c_att, index).transpose(-1, -2)

            if query_layer.size(-2) != key_layer.size(-2):
                pos_index = relative_pos[:, :, :, 0].unsqueeze(-1)
                index = pos_index.expand(pos_index, p2c_att, key_layer)
                p2c_att = gather_last_dim(p2c_att, index)
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
            if restore:
                if isinstance(mod, nn.Dropout):
                    mod.__class__ = StableDropout
                    mod.drop_prob = mod.p
                    mod.count = 0
                    mod.context_stack = None
            else:
                if isinstance(mod, StableDropout):
                    mod.__class__ = nn.Dropout
                    mod.p = mod.drop_prob
                    mod.inplace = False
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

        logger.info("-------------------- Device Allocation --------------------")
        logger.info("Embedding  --> IPU 0")
        if self.ipu_config.embedding_serialization_factor > 1:
            if isinstance(self, PipelinedDebertaForMaskedLM):
                serialized_decoder = SerializedLinear(
                    self.config.hidden_size,
                    self.config.vocab_size,
                    self.ipu_config.embedding_serialization_factor,
                    bias=True,
                    mode=poptorch.MatMulSerializationMode.OutputChannels,
                )
                serialized_decoder.load_state_dict(self.cls.predictions.decoder.state_dict())
                self.cls.predictions.decoder = serialized_decoder
                self.tie_weights()
            else:
                self.deberta.embeddings.word_embeddings = SerializedEmbedding(
                    self.deberta.embeddings.word_embeddings, self.ipu_config.embedding_serialization_factor
                )

        self.change_modules_for_ipu(False)

        self.deberta.embeddings = poptorch.BeginBlock(self.deberta.embeddings, "Embedding", ipu_id=0)
        hs = outline_attribute(self.deberta.embeddings.LayerNorm, "embedding")
        self._hooks.extend(hs)

        self.deberta.encoder = poptorch.BeginBlock(self.deberta.encoder, ipu_id=0)
        if self.deberta.encoder.relative_attention:
            self.deberta.encoder.rel_embeddings = poptorch.BeginBlock(self.deberta.encoder.rel_embeddings, ipu_id=0)

        layer_ipu = get_layer_ipu(self.ipu_config.layers_per_ipu, self.deberta.encoder.layer)
        for index, layer in enumerate(self.deberta.encoder.layer):
            ipu = layer_ipu[index]
            if self.ipu_config.recompute_checkpoint_every_layer and index != self.config.num_hidden_layers - 1:
                h = recomputation_checkpoint(layer)
                self._hooks.append(h)
            self.deberta.encoder.layer[index] = poptorch.BeginBlock(layer, f"Encoder{index}", ipu_id=ipu)
            logger.info(f"Encoder {index:<2} --> IPU {ipu}")

        if isinstance(self, PipelinedDebertaForMaskedLM):
            logger.info(f"Projection {index:<2} --> IPU {0}")
            self.cls.predictions.decoder = poptorch.BeginBlock(self.cls.predictions.decoder, "Projection", ipu_id=0)
        return self

    def deparallelize(self):
        """
        Undo the changes to the model done by `parallelize`.
        You should call this before doing `save_pretrained` so that the `model.state_dict` is
        compatible with the original model.
        """
        super().deparallelize()
        self.change_modules_for_ipu(True)
        if self.ipu_config.embedding_serialization_factor > 1:
            if isinstance(self, PipelinedDebertaForMaskedLM):
                decoder = nn.Linear(
                    self.config.hidden_size,
                    self.config.vocab_size,
                    bias=True,
                )
                decoder.load_state_dict(self.cls.predictions.decoder.state_dict())
                self.cls.predictions.decoder = decoder
                self.tie_weights()
            else:
                # Deserialize the serialized word embedding
                self.deberta.embeddings.word_embeddings = self.deberta.embeddings.word_embeddings.deserialize()
        return self


@register(DebertaForMaskedLM)
class PipelinedDebertaForMaskedLM(DebertaForMaskedLM, DebertaPipelineMixin):
    """
    DebertaForMaskedLM transformed to run in an IPU pipeline.

    Recommended usage:
    ```
    model = PipelinedDebertaForMaskedLM(config).parallelize().half()
    ```
    """

    def __init__(self, config):
        super().__init__(config)
        self.gather_indices = OnehotGather()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        if labels is not None:
            # Select only the masked tokens for the classifier
            max_number_of_masked_tokens = int(labels.size(1) * 0.25)
            masked_lm_labels, masked_lm_positions = torch.topk(labels, k=max_number_of_masked_tokens, dim=1)
            masked_output = self.gather_indices(sequence_output, masked_lm_positions)
        else:
            # This case should never happen during training
            masked_output = sequence_output

        prediction_scores = self.cls(masked_output)

        masked_lm_loss = None
        if labels is not None:
            masked_lm_loss = F.cross_entropy(
                prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1)
            ).float()

        if not return_dict:
            output = (prediction_scores,) + outputs[1:]
            return ((masked_lm_loss,)) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores if masked_lm_loss is None else None,
            hidden_states=outputs.hidden_states if masked_lm_loss is None else None,
            attentions=outputs.attentions if masked_lm_loss is None else None,
        )


@register(DebertaForSequenceClassification)
class PipelinedDebertaForSequenceClassification(DebertaForSequenceClassification, DebertaPipelineMixin):
    """
    DebertaForSequenceClassification transformed to run in an IPU pipeline.

    Recommended usage:
    ```
    model = PipelinedDebertaForSequenceClassification(config).parallelize().half()
    ```
    """

    def parallelize(self):
        super().parallelize()
        last_ipu = self.ipu_config.ipus_per_replica - 1
        logger.info(f"Classifier Output --> IPU {last_ipu}")
        self.classifier = poptorch.BeginBlock(self.classifier, "Classifier Output", ipu_id=last_ipu)
        logger.info("-----------------------------------------------------------")
        return self


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
        last_ipu = self.ipu_config.ipus_per_replica - 1
        logger.info(f"Classifier Output --> IPU {last_ipu}")
        self.classifier = poptorch.BeginBlock(self.classifier, "Classifier Output", ipu_id=last_ipu)
        logger.info("-----------------------------------------------------------")
        return self

    def deparallelize(self):
        super().deparallelize()
        # Last dropout isn't a StableDropout so undo its replacement
        # made by change_modules_for_ipu
        mod = self.dropout
        if isinstance(mod, StableDropout):
            mod.__class__ = nn.Dropout
            mod.p = mod.drop_prob
            mod.inplace = False


@register(DebertaForQuestionAnswering)
class PipelinedDebertaForQuestionAnswering(DebertaForQuestionAnswering, DebertaPipelineMixin):
    """
    DebertaForQuestionAnswering transformed to run in an IPU pipeline.

    Recommended usage:
    ```
    model = PipelinedDebertaForQuestionAnswering(config).parallelize().half()
    ```
    """

    def parallelize(self):
        super().parallelize()
        last_ipu = self.ipu_config.ipus_per_replica - 1
        logger.info(f"QA Outputs --> IPU {last_ipu}")
        self.qa_outputs = poptorch.BeginBlock(self.qa_outputs, "QA Outputs", ipu_id=last_ipu)
        logger.info("-----------------------------------------------------------")
        return self

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # return_dict = False

        output = super().forward(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            start_positions=start_positions,
            end_positions=end_positions,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if start_positions is not None and end_positions is not None:
            output = (poptorch.identity_loss(output[0], reduction="none"),) + output[1:]
        return output
