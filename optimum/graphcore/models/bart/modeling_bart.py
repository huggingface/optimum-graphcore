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
from typing import Optional

import torch
import torch.nn as nn

import poptorch
import transformers
from optimum.utils import logging
from transformers import BartForConditionalGeneration, BartModel
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqModelOutput
from transformers.models.bart.modeling_bart import shift_tokens_right

from ...modeling_utils import PipelineMixin, register
from ...generation_utils import IPUGenerationMixin


logger = logging.get_logger(__name__)

FLOAT16_LIMIT = 1e4


def _get_layer_ipu(layers_per_ipu):
    # List of the IPU Id for each encoder layer
    layer_ipu = []
    for ipu, n_layers in enumerate(layers_per_ipu):
        layer_ipu += [ipu] * n_layers
    return layer_ipu


def recomputation_checkpoint(module: nn.Module):
    """Annotates the output of a module to be checkpointed instead of
    recomputed"""

    def recompute_outputs(module, inputs, outputs):
        return tuple(poptorch.recomputationCheckpoint(y) for y in outputs)

    module.register_forward_hook(recompute_outputs)


def outline_attribute(module: nn.Module, value: str):
    """Adds an attribute to a module. This attribute will be used
    when comparing operation equivalence in outlining. For example:

    layer1 = nn.Linear(...)
    layer2 = nn.Linear(...)
    layer3 = nn.Linear(...)
    layer4 = nn.Linear(...)
    outline_attribute(layer1, "A")
    outline_attribute(layer2, "A")
    outline_attribute(layer3, "B")

    The code for layer1 can be reused for layer2.
    But it can't be used for layer3 or layer4.
    """
    context = poptorch.Attribute(__outline={"layer": value})

    def enable(*args):
        context.__enter__()

    def disable(*args):
        context.__exit__(None, None, None)

    module.register_forward_pre_hook(enable)
    module.register_forward_hook(disable)


class SerializedEmbedding(nn.Module):
    """
    Wrapper for `nn.Embedding` layer that performs the embedding look-up into
    smaller serialized steps in order to reduce memory in the embedding gradient
    calculation.

    Args:
        embedding: A `nn.Embedding` to wrap
        serialization_factor: The number of serialized embedding look-ups
    """

    def __init__(self, embedding: nn.Embedding, serialization_factor: int):
        super().__init__()
        self.serialization_factor = serialization_factor
        self.num_embeddings = embedding.num_embeddings

        # Num embeddings should be divisible by the serialization factor
        assert self.num_embeddings % self.serialization_factor == 0
        self.split_size = self.num_embeddings // self.serialization_factor
        self.split_embeddings = nn.ModuleList(
            [
                nn.Embedding.from_pretrained(
                    embedding.weight[i * self.split_size : (i + 1) * self.split_size, :].detach(),
                    freeze=False,
                    padding_idx=embedding.padding_idx if i == 0 else None,
                )
                for i in range(self.serialization_factor)
            ]
        )

    def deserialize(self):
        """
        Deserialize the internal wrapped embedding layer and return it as a
        `nn.Embedding` object.

        Returns:
            `nn.Embedding` layer
        """
        return nn.Embedding.from_pretrained(torch.vstack([l.weight for l in self.split_embeddings]), padding_idx=0)

    def forward(self, indices):
        # iterate through the splits
        x_sum = None
        for i in range(self.serialization_factor):
            # mask out the indices not in this split
            split_indices = indices - i * self.split_size
            mask = (split_indices >= 0) * (split_indices < self.split_size)
            mask = mask.detach()
            split_indices *= mask

            # do the embedding lookup
            x = self.split_embeddings[i](split_indices)

            # multiply the output by mask
            x *= mask.unsqueeze(-1)

            # add to partial
            if x_sum is not None:
                x_sum += x
            else:
                x_sum = x
        return x_sum


class SharedEmbedding(torch.nn.Module):
    def __init__(self, shared):
        super().__init__()
        # self.shared = shared
        # Doing this is necessary to make it work, otherwise an error is thrown:
        # "In poptorch/python/poptorch.cpp:1319: 'popart_internal_exception': tensor model.model.shared.shared.weight/d"
        self.shared = nn.Embedding.from_pretrained(shared.weight, freeze=False, padding_idx=shared.padding_idx)

    def forward(self, input_ids, encoder_embed_scale, decoder_input_ids, decoder_embed_scale):
        encoder_inputs_embeds, decoder_inputs_embeds = None, None
        if input_ids is not None and encoder_embed_scale is not None:
            encoder_inputs_embeds = self.shared(input_ids) * encoder_embed_scale
        if decoder_input_ids is not None and decoder_embed_scale is not None:
            decoder_inputs_embeds = self.shared(decoder_input_ids) * decoder_embed_scale
        return encoder_inputs_embeds, decoder_inputs_embeds

class SerializedLinear(nn.Linear):
    """
    Exactly equivalent to `nn.Linear` layer, but with the matrix multiplication replaced with
    a serialized matrix multiplication: `poptorch.serializedMatMul`.
    The matrix multiplication is split into separate smaller multiplications, calculated one after the other,
    to reduce the memory requirements of the multiplication and its gradient calculation.

    Args:
        in_features: Size of each input sample
        out_features: Size of each output sample
        factor: Number of serialized multiplications. Must be a factor of
            the dimension to serialize on.
        bias: If set to False, the layer will not learn an additive bias.
            Default: True
        mode: Which dimension of the matmul to serialize on:
            for matrix A (m by n) multiplied by matrix B (n by p).
            * InputChannels: Split across the input channels (dimension m).
            * ReducingDim: Split across the reducing dimension (n).
            * OutputChannels: Split across the output channels (dimension p).
            * Disabled: Same as an ordinary matrix multiplication.
    """

    def __init__(
        self, in_features, out_features, factor, bias=False, mode=poptorch.MatMulSerializationMode.OutputChannels
    ):
        super().__init__(in_features, out_features, bias)
        self.mode = mode
        self.factor = factor

    def forward(self, x):
        output = poptorch.serializedMatMul(x, self.weight.t(), self.mode, self.factor)
        if self.bias is not None:
            output += self.bias
        return output


class BartModelWithInputEmbeds(BartModel):
    @property
    def is_encoder_and_decoder_embeddings_computation_shared(self):
        return isinstance(self.shared, SharedEmbedding)

    def encoder_and_decoder_embeddings_computation(self, use_shared_embedding):
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

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                # input_ids=input_ids,
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
            # input_ids=decoder_input_ids,
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
    # Using FLOAT16_LIMIT instead of -float("inf") to avoid NaNs on the IPUs.
    mask = torch.full((tgt_len, tgt_len), -FLOAT16_LIMIT)
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)
    # return torch.broadcast_to(mask[None, None, :, :], (bsz, 1, tgt_len, tgt_len + past_key_values_length))


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    # expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    expanded_mask = torch.zeros((bsz, 1, tgt_len, src_len), ).to(dtype)
    # expanded_mask = torch.broadcast_to(mask[:, None, None, :], (bsz, 1, tgt_len, src_len)).to(dtype)
    # expanded_mask = mask[:, None, None, :].repeat(bsz, 1, tgt_len, 1).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    # Using FLOAT16_LIMIT instead of -float("inf") to avoid NaNs on the IPUs.
    return inverted_mask.masked_fill(inverted_mask.bool(), -FLOAT16_LIMIT)


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
            self.tie_weights()

            # self.model.shared = SerializedEmbedding(self.model.shared, self.config.embedding_serialization_factor)

        self.model.__class__ = BartModelWithInputEmbeds
        # TODO: make sure to restore those functions back in deparallelize
        transformers.models.bart.modeling_bart._make_causal_mask = _make_causal_mask
        transformers.models.bart.modeling_bart._expand_mask = _expand_mask
        self.model.encoder_and_decoder_embeddings_computation(True)
        # self.model.shared.shared.weight = self.lm_head.weight
        self.lm_head.weight = self.model.shared.shared.weight
        self.model.shared = poptorch.BeginBlock(self.model.shared, "Embedding", ipu_id=0)

        self.model.encoder.embed_positions = poptorch.BeginBlock(self.model.encoder.embed_positions, ipu_id=0)
        # TODO: no LayerNorm in Bart after the embeddings?
        # outline_attribute(self.bert.embeddings.LayerNorm, "embedding")

        self.model.encoder.layernorm_embedding = poptorch.BeginBlock(self.model.encoder.layernorm_embedding, ipu_id=0)
        for index, layer in enumerate(self.model.encoder.layers):
            ipu = layer_ipu[index]
            if self.config.recompute_checkpoint_every_layer and index != self.config.num_hidden_layers - 1:
                recomputation_checkpoint(layer)
            self.model.encoder.layers[index] = poptorch.BeginBlock(layer, f"Encoder{index}", ipu_id=ipu)
            logger.info(f"Encoder {index:<2} --> IPU {ipu}")

        shift = len(self.model.encoder.layers)
        self.model.decoder.embed_positions = poptorch.BeginBlock(
            self.model.decoder.embed_positions, ipu_id=layer_ipu[shift]
        )
        self.model.decoder.layernorm_embedding = poptorch.BeginBlock(
            self.model.decoder.layernorm_embedding, ipu_id=layer_ipu[shift]
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
        # Deserialize the serialized word embedding
        if self.config.embedding_serialization_factor > 1:
            self.model.shared = self.model.shared.deserialize()

        return self

    def train(self, mode: bool = True) -> 'PipelinedBartForConditionalGeneration':
        mod = super(BartForConditionalGeneration, self).train(mode=mode)
        mod.forward = mod._forward_for_train if mode else mod._forward_for_generate
        return mod

    def _forward_for_train(self, input_ids, attention_mask, decoder_input_ids, labels=None):
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            # decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            return_dict=False,
            # TODO: actually find a way to use cache for decoding.
            # use_cache=False,
        )

    def _forward_for_generate(self, encoder_outputs, decoder_input_ids, attention_mask):
        return super().forward(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            # decoder_attention_mask=decoder_attention_mask,
            return_dict=False,
            # TODO: actually find a way to use cache for decoding.
            # use_cache=False,
        )

    forward = _forward_for_train
