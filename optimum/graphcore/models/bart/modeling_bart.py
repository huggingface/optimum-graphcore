#  Copyright 2021 The HuggingFace Team. All rights reserved.
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
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

import transformers
from optimum.utils import logging
from transformers import BartForConditionalGeneration, BartForSequenceClassification
from transformers.models.bart.modeling_bart import BartAttention
from transformers.modeling_outputs import Seq2SeqLMOutput, Seq2SeqSequenceClassifierOutput

from ....fx.optimization import ChangeTrueDivToMulByInverse, MergeLinears, ReversibleTransformation, compose
from ...fx.transformations import (
    AddPoptorchBlock,
    AddPoptorchBlocksInSeries,
    ClipValues,
    ClipValuesSymmetric,
    LinearToSerializedLinear,
    RecomputationCheckpoint,
    ShareEmbeddingComputation,
    TieWeights,
    TupleOutput,
    VocabEmbeddingToSerializedEmbedding,
)
from ...fx.utils import symbolic_trace_pipelined_model
from ...generation_utils import IPUGenerationMixin
from ...modeling_utils import GenerationMethodsMixin, PipelineMixin, get_layer_ipu, register


logger = logging.get_logger(__name__)

FLOAT16_LIMIT = 1e4

_OPTIMIZATION_TRANSFORMATIONS = [
    ChangeTrueDivToMulByInverse(),
    MergeLinears(),
    #    FuseBiasInLinear(),
]

_NON_REVERSIBLE_TRANSFORMATIONS = [
    ClipValuesSymmetric(1e4, exclude_targets=["view"]),
    ClipValues(1e-4, float("inf"), include_targets=[torch.nn.LayerNorm]),
    TupleOutput(),
]


def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
    """Makes causal mask used for bi-directional self-attention.
    This differs from the original implementation by:
        - Making the mask creation simpler in terms of operations used
        - Changing the value for tokens to mask to something compatible with fp16
        - Not expanding the final mask to [bsz, 1, tgt_len, src_len]
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), -FLOAT16_LIMIT, dtype=dtype)
    mask = torch.triu(mask, diagonal=1).to(dtype=dtype)
    return mask[None, None, :, :]


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, 1, src_seq_len]`.
    This differs from the original implementation by:
        - Changing the value for tokens to mask to something compatible with fp16
        - Not expanding the final mask to [bsz, 1, tgt_len, src_len]
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :]
    inverted_mask = 1.0 - expanded_mask

    inverted_mask = -float("inf") * inverted_mask
    return inverted_mask


class _BartAttentionWithoutException(BartAttention):
    """The same as BartAttention without the attention mask shape check.

    This is needed because the original BartAttention checks that the attention mask shape is [bs, 1, tgt_len, src_len]
    but the pipelined implementation does not expand the mask, it just inserts dimensions, the shape is then
    [bs, 1, 1, src_len], and broadcasting does the rest.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        # if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
        #     raise ValueError(
        #         f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
        #     )

        if attention_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = torch.nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        # if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
        #     raise ValueError(
        #         f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
        #     )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


@register(BartForConditionalGeneration)
class PipelinedBartForConditionalGeneration(
    GenerationMethodsMixin, BartForConditionalGeneration, PipelineMixin, IPUGenerationMixin
):
    def get_transformations(self):
        log_insertions = self.ipu_config.log_insertions
        layer_ipu = get_layer_ipu(self.ipu_config.layers_per_ipu)
        transformations = [
            AddPoptorchBlock("Embedding", 0, "model.shared", log_insertions=log_insertions),
            # AddPoptorchBlock("Embedding", 0, "model.encoder.embed_positions"),
            # AddPoptorchBlock("Embedding", 0, "model.encoder.layernorm_embedding"),
            # AddPoptorchBlock("Embedding", 0, "model.decoder.embed_positions"),
            # AddPoptorchBlock("Embedding", 0, "model.decoder.layernorm_embedding"),
            AddPoptorchBlocksInSeries(
                "Encoder",
                layer_ipu[: self.config.encoder_layers],
                r"model.encoder.layers.[0-9]+",
                log_insertions=log_insertions,
            ),
            AddPoptorchBlocksInSeries(
                "Decoder",
                layer_ipu[self.config.encoder_layers :],
                r"model.decoder.layers.[0-9]+",
                log_insertions=log_insertions,
            ),
            AddPoptorchBlock("LM Head Output", 0, "lm_head", log_insertions=log_insertions),
        ]
        if self.ipu_config.recompute_checkpoint_every_layer:
            transformations += [
                RecomputationCheckpoint(
                    "model.encoder.layers.[0-9]+", to_exclude=f"model.encoder.layers.{self.config.encoder_layers - 1}"
                ),
                RecomputationCheckpoint(
                    "model.decoder.layers.[0-9]+", to_exclude=f"model.decoder.layers.{self.config.decoder_layers - 1}"
                ),
            ]

        if not isinstance(self, torch.fx.GraphModule):
            if self.ipu_config.embedding_serialization_factor > 1:
                transformations += [
                    LinearToSerializedLinear("lm_head"),
                    TieWeights("model.shared", "lm_head"),
                ]
            transformations += [ShareEmbeddingComputation()]
        return transformations

    def parallelize(self):
        """
        Transform the model to run in an IPU pipeline.
        - Adds pipeline stages to the model
        - (If enabled) Replaces the shared embedding with a SerializedEmbedding
        - Adds recomputation checkpoints

        Recommended usage:
        ```
        model = PipelinedBartForConditionalGeneration(config).parallelize().half()
        ```
        """
        super().parallelize()
        orig_make_causal_mask = transformers.models.bart.modeling_bart._make_causal_mask
        orig_expand_mask = transformers.models.bart.modeling_bart._expand_mask
        transformers.models.bart.modeling_bart._make_causal_mask = _make_causal_mask
        transformers.models.bart.modeling_bart._expand_mask = _expand_mask
        for mod in self.modules():
            if isinstance(mod, BartAttention):
                mod.__class__ = _BartAttentionWithoutException
        traced = symbolic_trace_pipelined_model(self)
        transformers.models.bart.modeling_bart._make_causal_mask = orig_make_causal_mask
        transformers.models.bart.modeling_bart._expand_mask = orig_expand_mask
        transformations = self.get_transformations()
        transformations += _OPTIMIZATION_TRANSFORMATIONS
        composition = compose(*transformations)
        non_reversible_composition = compose(*_NON_REVERSIBLE_TRANSFORMATIONS)
        traced = composition(traced)
        traced = non_reversible_composition(traced)
        return traced

    def deparallelize(self):
        """
        Undo the changes to the model done by `parallelize`.
        You should call this before doing `save_pretrained` so that the `model.state_dict` is
        fully compatible with `transformers.BartForConditionalGeneration`.
        """
        super().deparallelize()
        transformations = self.get_transformations()
        transformations += _OPTIMIZATION_TRANSFORMATIONS
        transformations = [t for t in transformations if isinstance(t, ReversibleTransformation)]
        composition = compose(*transformations)
        self = composition(self, reverse=True)
        return self

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """
        outputs = super().forward(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if self.training:
            # Only returning the loss to make the communication between the host and the device faster.
            if not return_dict:
                return outputs[0:1]
            else:
                return Seq2SeqLMOutput(loss=outputs.loss)
        else:
            return outputs


@register(BartForSequenceClassification)
class PipelinedBartForSequenceClassification(BartForSequenceClassification, PipelineMixin):
    def get_transformations(self):
        log_insertions = self.ipu_config.log_insertions
        layer_ipu = get_layer_ipu(self.ipu_config.layers_per_ipu)
        transformations = [
            AddPoptorchBlock("Embedding", 0, "model.shared", log_insertions=log_insertions),
            # AddPoptorchBlock("Embedding", 0, "model.encoder.embed_positions"),
            # AddPoptorchBlock("Embedding", 0, "model.encoder.layernorm_embedding"),
            # AddPoptorchBlock("Embedding", 0, "model.decoder.embed_positions"),
            # AddPoptorchBlock("Embedding", 0, "model.decoder.layernorm_embedding"),
            AddPoptorchBlocksInSeries(
                "Encoder",
                layer_ipu[: self.config.encoder_layers],
                r"model.encoder.layers.[0-9]+",
                log_insertions=log_insertions,
            ),
            AddPoptorchBlocksInSeries(
                "Decoder",
                layer_ipu[self.config.encoder_layers :],
                r"model.decoder.layers.[0-9]+",
                log_insertions=log_insertions,
            ),
            AddPoptorchBlock(
                "Classification Head Output", layer_ipu[-1], "classification_head", log_insertions=log_insertions
            ),
        ]
        if self.ipu_config.recompute_checkpoint_every_layer:
            transformations += [
                RecomputationCheckpoint(
                    "model.encoder.layers.[0-9]+", to_exclude=f"model.encoder.layers.{self.config.encoder_layers - 1}"
                ),
                RecomputationCheckpoint(
                    "model.decoder.layers.[0-9]+", to_exclude=f"model.decoder.layers.{self.config.decoder_layers - 1}"
                ),
            ]

        if not isinstance(self, torch.fx.GraphModule):
            if self.ipu_config.embedding_serialization_factor > 1:
                transformations.append(VocabEmbeddingToSerializedEmbedding())
            transformations += [ShareEmbeddingComputation()]
        return transformations

    def parallelize(self):
        """
        Transform the model to run in an IPU pipeline.
        - Adds pipeline stages to the model
        - (If enabled) Replaces the shared embedding with a SerializedEmbedding
        - Adds recomputation checkpoints

        Recommended usage:
        ```
        model = PipelinedBartForConditionalGeneration(config).parallelize().half()
        ```
        """
        super().parallelize()
        orig_make_causal_mask = transformers.models.bart.modeling_bart._make_causal_mask
        orig_expand_mask = transformers.models.bart.modeling_bart._expand_mask
        transformers.models.bart.modeling_bart._make_causal_mask = _make_causal_mask
        transformers.models.bart.modeling_bart._expand_mask = _expand_mask
        for mod in self.modules():
            if isinstance(mod, BartAttention):
                mod.__class__ = _BartAttentionWithoutException
        traced = symbolic_trace_pipelined_model(self)
        transformers.models.bart.modeling_bart._make_causal_mask = orig_make_causal_mask
        transformers.models.bart.modeling_bart._expand_mask = orig_expand_mask
        transformations = self.get_transformations()
        transformations += _OPTIMIZATION_TRANSFORMATIONS
        composition = compose(*transformations)
        non_reversible_composition = compose(*_NON_REVERSIBLE_TRANSFORMATIONS)
        traced = composition(traced)
        traced = non_reversible_composition(traced)
        return traced

    def deparallelize(self):
        """
        Undo the changes to the model done by `parallelize`.
        You should call this before doing `save_pretrained` so that the `model.state_dict` is
        fully compatible with `transformers.BartForConditionalGeneration`.
        """
        super().deparallelize()
        transformations = self.get_transformations()
        transformations += _OPTIMIZATION_TRANSFORMATIONS
        transformations = [t for t in transformations if isinstance(t, ReversibleTransformation)]
        composition = compose(*transformations)
        self = composition(self, reverse=True)
        return self

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqSequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]  # last hidden state
        B, L, E = hidden_states.shape

        eos_mask = torch.eq(input_ids, self.config.eos_token_id)
        # Static tensor shape version of hidden_states[eos_mask, :]
        eos_indices = eos_mask * torch.arange(L).unsqueeze(0)
        last_eos_index, _ = torch.max(eos_indices, dim=1)
        # torch.index_select requires a 1D tensor of indices
        last_eos_index += torch.arange(B) * L
        hidden_states = hidden_states.view(B * L, E)
        sentence_representation = torch.index_select(hidden_states, 0, last_eos_index)

        logits = self.classification_head(sentence_representation)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.config.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.config.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.config.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
