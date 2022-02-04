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

from typing import Optional, Tuple

import torch
import torch.nn as nn

import poptorch
import transformers
from optimum.utils import logging
from transformers import BartForConditionalGeneration, BartModel
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqModelOutput
from transformers.models.bart.modeling_bart import _expand_mask as original_expand_mask
from transformers.models.bart.modeling_bart import _make_causal_mask as original_make_causal_mask
from transformers.models.bart.modeling_bart import shift_tokens_right

from ...generation_utils import IPUGenerationMixin
from ...modeling_utils import (
    PipelineMixin,
    SerializedLinear,
    _get_layer_ipu,
    recomputation_checkpoint,
    register,
)


logger = logging.get_logger(__name__)

FLOAT16_LIMIT = 1e4


class SharedEmbedding(torch.nn.Module):
    """Wrapper around the shared embedding between the encoder and the decoder stacks.

    Attributes:
        shared: The shared embedding layer.
    """

    def __init__(self, shared: nn.Embedding):
        super().__init__()
        self.shared = shared

    # def _combine_inputs(self, input_ids, decoder_input_ids):
    #     encoder_seq_length = input_ids.size(1)
    #     decoder_seq_length = decoder_input_ids.size(1)
    #     if encoder_seq_length > decoder_seq_length:
    #         decoder_input_ids = torch.nn.functional.pad(
    #             decoder_input_ids,
    #             (0, encoder_seq_length - decoder_seq_length),
    #             value=self.pad_token_id
    #         )
    #     elif encoder_seq_length < decoder_seq_length:
    #         input_ids = torch.nn.functional.pad(
    #             input_ids,
    #             (0, decoder_seq_length - encoder_seq_length),
    #             value=self.pad_token_id
    #         )
    #     combined = torch.stack([input_ids, decoder_input_ids], dim=0)
    #     return combined, encoder_seq_length, decoder_seq_length

    # def _separate_inputs(self, embeds, encoder_seq_length, decoder_seq_length):
    #     encoder_embeds, decoder_embeds = embeds
    #     return encoder_embeds[:, :encoder_seq_length, :], decoder_embeds[:, :decoder_seq_length, :]

    def _combine_inputs(self, input_ids: torch.Tensor, decoder_input_ids: torch.Tensor) -> Tuple[int, torch.Tensor]:
        idx = input_ids.size(1)
        return idx, torch.cat([input_ids, decoder_input_ids], dim=1)

    def _separate_inputs(self, idx: int, embeds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return embeds[:, :idx, :], embeds[:, idx:, :]

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_embed_scale: float,
        decoder_input_ids: torch.Tensor,
        decoder_embed_scale: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: use this once the TiedGather pattern issue is solved.
        # encoder_inputs_embeds, decoder_inputs_embeds = None, None
        # if input_ids is not None and encoder_embed_scale is not None:
        #     encoder_inputs_embeds = self.shared(input_ids) * encoder_embed_scale
        # if decoder_input_ids is not None and decoder_embed_scale is not None:
        #     decoder_inputs_embeds = self.shared(decoder_input_ids) * decoder_embed_scale
        # combined, n1, n2 = self._combine_inputs(input_ids, decoder_input_ids)
        # encoder_inputs_embeds, decoder_inputs_embeds = self._separate_inputs(self.shared(combined), n1, n2)
        idx, combined = self._combine_inputs(input_ids, decoder_input_ids)
        encoder_inputs_embeds, decoder_inputs_embeds = self._separate_inputs(idx, self.shared(combined))

        encoder_inputs_embeds = encoder_inputs_embeds * encoder_embed_scale
        decoder_inputs_embeds = decoder_inputs_embeds * decoder_embed_scale

        return encoder_inputs_embeds, decoder_inputs_embeds


class BartModelWithSharedEmbedding(BartModel):
    @property
    def is_encoder_and_decoder_embeddings_computation_shared(self):
        return isinstance(self.shared, SharedEmbedding)

    def encoder_and_decoder_embeddings_computation(self, use_shared_embedding: bool):
        """Sets the BartModel shared embedding layer to SharedEmbedding that combines the computation under one layer.

        Args:
            use_shared_embedding: whether to use SharedEmbedding or not.
        """

        if use_shared_embedding:
            if isinstance(self.shared, SharedEmbedding):
                logger.warning("encoder and decoder embeddings computation is already shared")
            else:
                self.shared = SharedEmbedding(self.shared)
                self.encoder.embed_tokens = None
                self.decoder.embed_tokens = None
        else:
            if isinstance(self.shared, nn.Embedding):
                logger.warning("encoder and decoder embeddings computation is not shared")
            else:
                self.shared = self.shared.shared

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.is_encoder_and_decoder_embeddings_computation_shared:
            encoder_inputs_embeds, decoder_inputs_embeds = self.shared(
                input_ids=input_ids,
                encoder_embed_scale=self.encoder.embed_scale,
                decoder_input_ids=decoder_input_ids,
                decoder_embed_scale=self.decoder.embed_scale,
            )
            if encoder_inputs_embeds is not None:
                input_ids = None
            if decoder_inputs_embeds is not None:
                decoder_input_ids = None

        if attention_mask is not None:
            attention_mask = _expand_mask(attention_mask, encoder_inputs_embeds.dtype)

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=encoder_inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), -FLOAT16_LIMIT)
    mask = torch.triu(mask, diagonal=1)
    return mask[None, None, :, :]


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :]
    inverted_mask = 1.0 - expanded_mask

    # Using FLOAT16_LIMIT instead of -float("inf") to avoid NaNs on the IPUs.
    inverted_mask = -FLOAT16_LIMIT * inverted_mask
    return inverted_mask


@register(BartForConditionalGeneration)
class PipelinedBartForConditionalGeneration(IPUGenerationMixin, BartForConditionalGeneration, PipelineMixin):
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
        # Use faster fused-qkv self-attention
        # TODO: should we use this for BART?
        # for layer in self.bert.encoder.layer:
        #     fused = BertFusedSelfAttention(self.config)
        #     fused.load_state_dict(layer.attention.self.state_dict())
        #     layer.attention.self = fused

        layer_ipu = _get_layer_ipu(self.config.layers_per_ipu)

        logger.info("-------------------- Device Allocation --------------------")
        logger.info("Embedding  --> IPU 0")

        if self.config.embedding_serialization_factor > 1:
            serialized_lm_head = SerializedLinear(
                self.config.d_model,
                self.model.shared.num_embeddings,
                self.config.embedding_serialization_factor,
                bias=False,
                mode=poptorch.MatMulSerializationMode.OutputChannels,
            )
            serialized_lm_head.load_state_dict(self.lm_head.state_dict())
            self.lm_head = serialized_lm_head
            self.lm_head.weight
            self.tie_weights()

        self.model.__class__ = BartModelWithSharedEmbedding
        transformers.models.bart.modeling_bart._make_causal_mask = _make_causal_mask
        transformers.models.bart.modeling_bart._expand_mask = _expand_mask
        self.model.encoder_and_decoder_embeddings_computation(True)

        self.model.shared = poptorch.BeginBlock(self.model.shared, "Embedding", ipu_id=0)
        self.model.encoder.embed_positions = poptorch.BeginBlock(self.model.encoder.embed_positions, "Embedding", ipu_id=0)
        self.model.encoder.layernorm_embedding = poptorch.BeginBlock(self.model.encoder.layernorm_embedding, "Embedding", ipu_id=0)
        # outline_attribute(self.model.encoder.layernorm_embedding, "Embedding")

        for index, layer in enumerate(self.model.encoder.layers):
            ipu = layer_ipu[index]
            if self.config.recompute_checkpoint_every_layer and index != self.config.num_hidden_layers - 1:
                recomputation_checkpoint(layer)
            self.model.encoder.layers[index] = poptorch.BeginBlock(layer, f"Encoder{index}", ipu_id=ipu)
            logger.info(f"Encoder {index:<2} --> IPU {ipu}")

        shift = len(self.model.encoder.layers)
        # self.model.decoder.embed_positions = poptorch.BeginBlock(
        #     self.model.decoder.embed_positions, ipu_id=layer_ipu[shift]
        # )
        # self.model.decoder.layernorm_embedding = poptorch.BeginBlock(
        #     self.model.decoder.layernorm_embedding, ipu_id=layer_ipu[shift]
        # )
        self.model.decoder.embed_positions = poptorch.BeginBlock(
            self.model.decoder.embed_positions, "Embedding", ipu_id=0
        )
        self.model.decoder.layernorm_embedding = poptorch.BeginBlock(
            self.model.decoder.layernorm_embedding, "Embedding", ipu_id=0
        )
        for index, layer in enumerate(self.model.decoder.layers):
            ipu = layer_ipu[index + shift]
            if self.config.recompute_checkpoint_every_layer and index != self.config.num_hidden_layers - 1:
                recomputation_checkpoint(layer)
            self.model.decoder.layers[index] = poptorch.BeginBlock(layer, f"Decoder{index}", ipu_id=ipu)
            logger.info(f"Decoder {index:<2} --> IPU {ipu}")

        logger.info("LM Head Output --> IPU 0")
        self.lm_head = poptorch.BeginBlock(self.lm_head, "LM Head Output", ipu_id=0)
        logger.info("-----------------------------------------------------------")
        return self

    def deparallelize(self):
        """
        Undo the changes to the model done by `parallelize`.
        You should call this before doing `save_pretrained` so that the `model.state_dict` is
        fully compatible with `transformers.BartForConditionalGeneration`.
        """
        super().deparallelize()
        self.model.encoder_and_decoder_embeddings_computation(False)
        self.model.__class__ = BartModel
        transformers.models.bart.modeling_bart._make_causal_mask = original_make_causal_mask
        transformers.models.bart.modeling_bart._expand_mask = original_expand_mask
        return self

    def train(self, mode: bool = True) -> "PipelinedBartForConditionalGeneration":
        mod = super(BartForConditionalGeneration, self).train(mode=mode)
        mod.forward = mod._forward_for_train if mode else mod._forward_for_generate
        return mod

    def _forward_for_train(self, input_ids, attention_mask, decoder_input_ids, labels=None):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            return_dict=False,
        )
        # Only returning the loss to make the communication between the host and the device faster.
        return outputs[0]

    def _forward_for_generate(self, encoder_outputs, decoder_input_ids, attention_mask):
        return super().forward(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            return_dict=False,
            # TODO: actually find a way to use cache for decoding.
            # use_cache=False,
        )

    forward = _forward_for_train
