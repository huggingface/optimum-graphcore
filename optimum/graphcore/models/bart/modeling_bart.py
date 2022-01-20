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

import torch
import torch.nn as nn
import torch.nn.functional as F

import poptorch
from optimum.utils import logging
from scipy.stats import truncnorm
from transformers import (
    BartModel,
    BartForConditionalGeneration,
)
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqModelOutput
from transformers.models.bart.modeling_bart import shift_tokens_right

from ...modeling_utils import PipelineMixin, register


logger = logging.get_logger(__name__)


class OnehotGather(nn.Module):
    """
    Gathers selected indices from a tensor by transforming the list of indices
    into a one-hot matrix and then multiplying the tensor by that matrix.
    """

    def __init__(self):
        super().__init__()
        self._is_half = False

    def half(self):
        super().half()
        # Tracing is always executed in float as there are missing
        # implementations of operations in half on the CPU.
        # So we cannot query the inputs to know if we are running
        # with a model that has had .half() called on it.
        # To work around it nn.Module::half is overridden
        self._is_half = True

    def forward(self, sequence, positions):
        """
        Gather the vectors at the specific positions over a batch.
        """
        num_classes = int(sequence.shape[1])
        one_hot_positions = F.one_hot(positions, num_classes)
        if self._is_half:
            one_hot_positions = one_hot_positions.half()
        else:
            one_hot_positions = one_hot_positions.float()
        return torch.matmul(one_hot_positions.detach(), sequence)


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


def accuracy(out, targ):
    return (out.argmax(dim=-1) == targ).float().mean()


def accuracy_masked(out, targ, mask_val):
    mask = (targ != mask_val).float()
    num_unmasked = mask.sum(1).unsqueeze(1)
    return (out.argmax(dim=-1) == targ).float().mul(mask).div(num_unmasked).sum(1).mean()


# class PipelinedBertForPreTraining(BertForPreTraining, PipelineMixin):
#     def __init__(self, config):
#         super().__init__(config)
#         self.gather_indices = OnehotGather()
#
#     def parallelize(self):
#         """
#         Transform the model to run in an IPU pipeline.
#         - Adds pipeline stages to the model
#         - Replaces self-attention layers with fused-qkv self-attention layers
#         - (If enabled) Replaces the word embedding projection with a SerializedLinear layer
#         - Adds recomputation checkpoints
#
#         Recommended usage:
#         ```
#         model = PipelinedBertForPretraining(config).parallelize().half().train()
#         ```
#         """
#         # Use faster fused-qkv self-attention
#         for layer in self.bert.encoder.layer:
#             fused = BertFusedSelfAttention(self.config)
#             fused.load_state_dict(layer.attention.self.state_dict())
#             layer.attention.self = fused
#
#         if self.config.embedding_serialization_factor > 1:
#             serialized_decoder = SerializedLinear(
#                 self.config.hidden_size,
#                 self.config.vocab_size,
#                 self.config.embedding_serialization_factor,
#                 bias=True,
#                 mode=poptorch.MatMulSerializationMode.OutputChannels,
#             )
#             serialized_decoder.load_state_dict(self.cls.predictions.decoder.state_dict())
#             self.cls.predictions.decoder = serialized_decoder
#             self.tie_weights()
#
#         layer_ipu = _get_layer_ipu(self.config.layers_per_ipu)
#
#         logger.info("-------------------- Device Allocation --------------------")
#         logger.info("Embedding  --> IPU 0")
#         self.bert.embeddings = poptorch.BeginBlock(self.bert.embeddings, "Embedding", ipu_id=0)
#         # Preventing the embeddings.LayerNorm from being outlined with the encoder.layer.LayerNorm
#         # improves the tile mapping of the pipeline stashes
#         outline_attribute(self.bert.embeddings.LayerNorm, "embeddings")
#
#         for index, layer in enumerate(self.bert.encoder.layer):
#             ipu = layer_ipu[index]
#             if self.config.recompute_checkpoint_every_layer:
#                 recomputation_checkpoint(layer)
#             self.bert.encoder.layer[index] = poptorch.BeginBlock(layer, f"Encoder{index}", ipu_id=ipu)
#             logger.info(f"Encoder {index:<2} --> IPU {ipu}")
#
#         logger.info("Pooler     --> IPU 0")
#         self.bert.pooler = poptorch.BeginBlock(self.bert.pooler, "Pooler", ipu_id=0)
#
#         logger.info("Classifier --> IPU 0")
#         self.cls = poptorch.BeginBlock(self.cls, "Classifier", ipu_id=0)
#         logger.info("-----------------------------------------------------------")
#         return self
#
#     def _init_weights(self, module):
#         """Initialize the weights"""
#
#         def truncated_normal_(tensor, mean=0, std=1):
#             """
#             Truncated Normal distribution, truncated at 2 sigma
#             """
#             r = torch.tensor(truncnorm.rvs(-2, 2, loc=mean, scale=std, size=tensor.shape))
#             tensor.data.copy_(r)
#
#         if isinstance(module, nn.Linear):
#             truncated_normal_(module.weight, mean=0.0, std=self.config.initializer_range)
#             if module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.Embedding):
#             truncated_normal_(module.weight, mean=0.0, std=self.config.initializer_range)
#             if module.padding_idx is not None:
#                 module.weight.data[module.padding_idx].zero_()
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)
#
#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         labels=None,
#         next_sentence_label=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#     ):
#         output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
#         sequence_output, pooled_output = output[:2]
#
#         # Select only the masked tokens for the classifier
#         max_number_of_masked_tokens = int(labels.size(1) * 0.25)
#         masked_lm_labels, masked_lm_positions = torch.topk(labels, k=max_number_of_masked_tokens, dim=1)
#         masked_output = self.gather_indices(sequence_output, masked_lm_positions)
#
#         prediction_scores, sequential_relationship_score = self.cls(masked_output, pooled_output)
#         output = (
#             prediction_scores,
#             sequential_relationship_score,
#         ) + output[2:]
#
#         if labels is not None and next_sentence_label is not None:
#             masked_lm_loss = F.cross_entropy(
#                 prediction_scores.view(-1, self.config.vocab_size),
#                 masked_lm_labels.view(-1),
#                 ignore_index=-100,
#             ).float()
#             next_sentence_loss = F.cross_entropy(
#                 sequential_relationship_score.view(-1, 2), next_sentence_label.view(-1)
#             ).float()
#             total_loss = poptorch.identity_loss(masked_lm_loss + next_sentence_loss, reduction="none")
#             return (total_loss, masked_lm_loss, next_sentence_loss)
#
#         return output


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
        encoder_inputs_embeds = self.shared(input_ids) * encoder_embed_scale
        decoder_inputs_embeds = self.shared(decoder_input_ids) * decoder_embed_scale
        return encoder_inputs_embeds, decoder_inputs_embeds


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
            input_ids = None
            decoder_input_ids = None

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


@register(BartForConditionalGeneration)
class PipelinedBartForConditionalGeneration(BartForConditionalGeneration, PipelineMixin):
    def parallelize(self):
        """
        Transform the model to run in an IPU pipeline.
        - Adds pipeline stages to the model
        - Replaces self-attention layers with fused-qkv self-attention layers
        - (If enabled) Replaces the word embedding with a SerializedEmbedding
        - Adds recomputation checkpoints

        Recommended usage:
        ```
        model = PipelinedBertForSequenceClassification(config).parallelize().half()
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
            self.model.shared = SerializedEmbedding(
                self.model.shared, self.config.embedding_serialization_factor
            )

        self.model.__class__ = BartModelWithInputEmbeds
        self.model.encoder_and_decoder_embeddings_computation(True)
        self.model.shared = poptorch.BeginBlock(self.model.shared, "Embedding", ipu_id=0)

        self.model.encoder.embed_positions = poptorch.BeginBlock(
            self.model.encoder.embed_positions, ipu_id=0
        )
        # TODO: no LayerNorm in Bart after the embeddings?
        # outline_attribute(self.bert.embeddings.LayerNorm, "embedding")

        self.model.encoder.layernorm_embedding = poptorch.BeginBlock(
            self.model.encoder.layernorm_embedding, ipu_id=0
        )
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

        logger.info(f"LM Head Output --> IPU {ipu}")
        self.lm_head = poptorch.BeginBlock(self.lm_head, "LM Head Output", ipu_id=layer_ipu[-1])
        logger.info("-----------------------------------------------------------")
        return self

    def deparallelize(self):
        """
        Undo the changes to the model done by `parallelize`.
        You should call this before doing `save_pretrained` so that the `model.state_dict` is
        fully compatible with `transformers.BertForSequenceClassification`.
        """
        # TODO: implement deparallelize (need rebase)

        # Deserialize the serialized word embedding
        if self.config.embedding_serialization_factor > 1:
            self.model.shared = self.model.shared.deserialize()

        return self

    def forward(self, input_ids, attention_mask, decoder_input_ids, labels=None):
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            # decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            return_dict=False,
        )
