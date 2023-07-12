# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
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

from typing import Optional, Tuple, Union

import poptorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import truncnorm
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertForMultipleChoice,
    BertForPreTraining,
    BertForQuestionAnswering,
    BertForSequenceClassification,
    BertForTokenClassification,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    MaskedLMOutput,
    QuestionAnsweringModelOutput,
)
from transformers.modeling_utils import apply_chunking_to_forward
from transformers.models.bert.modeling_bert import BertForPreTrainingOutput, BertModel

from optimum.utils import logging

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
from .groupbert_attention import GroupBertAttention
from .groupbert_convolution import GroupBertConvolution
from .groupbert_ffn import GroupBertIntermediate, GroupBertOutput


logger = logging.get_logger(__name__)


class GroupBertConfig(BertConfig):
    r"""
    This is the configuration class to store the configuration of a [`GroupBertModel`]. It is used to
    instantiate a GroupBERT model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`BertConfig`] and can be used to control the model outputs. Read the
    documentation from [`BertConfig`] for more information.


    Args:
        ffn_groups (`int`, *optional*, defaults to 4):
            Number of groups on the down projection of FFN
        conv_group_size (`int`, *optional*, defaults to 16):
            Group size for the convolution operation in the dedicated convolution modele in GroupBERT
        conv_kernel_size (`int`, *optional*, defaults to 7):
            Kernel size for the convolution operation in the dedicated convolution modele in GroupBERT
    Examples:

    ```python
    >>> from optimum.graphcore import GroupBertModel, GroupBertConfig

    >>> # Initializing a GroupBERT configuration
    >>> configuration = GroupBertConfig()

    >>> # Initializing a model
    >>> model = GroupBertModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "groupbert"

    def __init__(self, ffn_groups=4, conv_group_size=16, conv_kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.ffn_groups = ffn_groups
        self.conv_group_size = conv_group_size
        self.conv_kernel_size = conv_kernel_size


class GroupBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.convolution = GroupBertConvolution(config)
        self.intermediate_first = GroupBertIntermediate(config)
        self.output_first = GroupBertOutput(config)
        self.attention = GroupBertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = GroupBertAttention(config, position_embedding_type="absolute")
        self.intermediate_second = GroupBertIntermediate(config)
        self.output_second = GroupBertOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        convolution_output = self.convolution(hidden_states, attention_mask)

        first_layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk_first, self.chunk_size_feed_forward, self.seq_len_dim, convolution_output
        )

        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            first_layer_output,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk_second, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk_first(self, conv_output):
        intermediate_output = self.intermediate_first(conv_output)
        layer_output = self.output_first(intermediate_output, conv_output)
        return layer_output

    def feed_forward_chunk_second(self, attention_output):
        intermediate_output = self.intermediate_second(attention_output)
        layer_output = self.output_second(intermediate_output, attention_output)
        return layer_output


class GroupBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([GroupBertLayer(config) for _ in range(config.num_hidden_layers)])
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        hidden_states = self.LayerNorm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class GroupBertModel(BertModel):
    config_class = GroupBertConfig

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)

        self.encoder = GroupBertEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def _init_weights(self, module):
        """Initialize the weights"""

        def truncated_normal_(tensor, mean=0, std=1):
            """
            Truncated Normal distribution, truncated at 2 sigma
            """
            r = torch.tensor(truncnorm.rvs(-2, 2, loc=mean, scale=std, size=tensor.shape))
            tensor.data.copy_(r)

        if isinstance(module, nn.Linear):
            truncated_normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            truncated_normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class GroupBertForPreTraining(BertForPreTraining):
    config_class = GroupBertConfig

    def __init__(self, config):
        super().__init__(config)

        self.bert = GroupBertModel(config)

        # Initialize weights and apply final processing
        self.post_init()


class GroupBertForMaskedLM(BertForMaskedLM):
    config_class = GroupBertConfig

    def __init__(self, config):
        super().__init__(config)

        self.bert = GroupBertModel(config)

        # Initialize weights and apply final processing
        self.post_init()


class GroupBertForSequenceClassification(BertForSequenceClassification):
    config_class = GroupBertConfig

    def __init__(self, config):
        super().__init__(config)

        self.bert = GroupBertModel(config)

        # Initialize weights and apply final processing
        self.post_init()


class GroupBertForMultipleChoice(BertForMultipleChoice):
    config_class = GroupBertConfig

    def __init__(self, config):
        super().__init__(config)

        self.bert = GroupBertModel(config)

        # Initialize weights and apply final processing
        self.post_init()


class GroupBertForTokenClassification(BertForTokenClassification):
    config_class = GroupBertConfig

    def __init__(self, config):
        super().__init__(config)

        self.bert = GroupBertModel(config)

        # Initialize weights and apply final processing
        self.post_init()


class GroupBertForQuestionAnswering(BertForQuestionAnswering):
    config_class = GroupBertConfig

    def __init__(self, config):
        super().__init__(config)

        self.bert = GroupBertModel(config)

        # Initialize weights and apply final processing
        self.post_init()


@register(GroupBertForPreTraining)
class PipelinedGroupBertForPreTraining(GroupBertForPreTraining, PipelineMixin):
    """
    GroupBertForPreTraining transformed to run in an IPU pipeline.

    Recommended usage:
    ```
    model = PipelinedGroupBertForPreTraining(config).parallelize().half().train()
    ```
    """

    def __init__(self, config):
        super().__init__(config)
        self.gather_indices = OnehotGather()

    def parallelize(self):
        """
        Transform the model to run in an IPU pipeline.
        - Adds pipeline stages to the model
        - (If enabled) Replaces the word embedding projection with a SerializedLinear layer
        - Adds recomputation checkpoints
        """
        super().parallelize()

        if self.ipu_config.embedding_serialization_factor > 1:
            self.cls.predictions.decoder = SerializedLinear.from_model(
                self.cls.predictions.decoder, self.ipu_config.embedding_serialization_factor
            )
            self.tie_weights()

        layer_ipu = get_layer_ipu(self.ipu_config, self.bert.encoder.layer)

        logger.info("-------------------- Device Allocation --------------------")
        logger.info("Embedding --> IPU 0")
        self.bert.embeddings = poptorch.BeginBlock(self.bert.embeddings, "Embedding", ipu_id=0)
        # Preventing the embeddings.LayerNorm from being outlined with the encoder.layer.LayerNorm
        # improves the tile mapping of the pipeline stashes
        hs = outline_attribute(self.bert.embeddings.LayerNorm, "embeddings")
        self._hooks.extend(hs)

        for index, layer in enumerate(self.bert.encoder.layer):
            ipu = layer_ipu[index]
            if self.ipu_config.recompute_checkpoint_every_layer:
                h = recomputation_checkpoint(layer)
                self._hooks.append(h)
            self.bert.encoder.layer[index] = poptorch.BeginBlock(layer, f"Encoder{index}", ipu_id=ipu)
            logger.info(f"Encoder {index:<2} --> IPU {ipu}")

        logger.info("Pooler --> IPU 0")
        self.bert.pooler = poptorch.BeginBlock(self.bert.pooler, "Pooler", ipu_id=0)

        logger.info("Classifier --> IPU 0")
        self.cls = poptorch.BeginBlock(self.cls, "Classifier", ipu_id=0)
        logger.info("-----------------------------------------------------------")
        return self

    def deparallelize(self):
        """
        Undo the changes to the model done by `parallelize`.
        You should call this before doing `save_pretrained` so that the `model.state_dict` is
        compatible with the original model.
        """
        super().deparallelize()

        if isinstance(self.cls.predictions.decoder, SerializedLinear):
            self.cls.predictions.decoder = self.cls.predictions.decoder.to_model()
            self.tie_weights()
        return self

    def _init_weights(self, module):
        """Initialize the weights"""

        def truncated_normal_(tensor, mean=0, std=1):
            """
            Truncated Normal distribution, truncated at 2 sigma
            """
            r = torch.tensor(truncnorm.rvs(-2, 2, loc=mean, scale=std, size=tensor.shape))
            tensor.data.copy_(r)

        if isinstance(module, nn.Linear):
            truncated_normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            truncated_normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        next_sentence_label: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BertForPreTrainingOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output, pooled_output = outputs[:2]

        if labels is not None:
            if hasattr(self.config, "max_num_masked_tokens"):
                # Select only the masked tokens for the classifier
                labels, positions = torch.topk(labels, k=self.config.max_num_masked_tokens, dim=1)
                sequence_output = self.gather_indices(sequence_output, positions)

        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        total_loss = None
        if labels is not None and next_sentence_label is not None:
            masked_lm_loss = F.cross_entropy(
                prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1),
            ).float()
            next_sentence_loss = F.cross_entropy(
                seq_relationship_score.view(-1, 2), next_sentence_label.view(-1)
            ).float()
            total_loss = poptorch.identity_loss(masked_lm_loss + next_sentence_loss, reduction="none")

        # If labels are provided (training mode) only output the loss
        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return (total_loss,) if total_loss is not None else output

        return BertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores if total_loss is None else None,
            seq_relationship_logits=seq_relationship_score if total_loss is None else None,
            hidden_states=outputs.hidden_states if total_loss is None else None,
            attentions=outputs.attentions if total_loss is None else None,
        )


@register(GroupBertForMaskedLM)
class PipelinedGroupBertForMaskedLM(GroupBertForMaskedLM, PipelineMixin):
    """
    GroupBertForMaskedLM transformed to run in an IPU pipeline.

    Recommended usage:
    ```
    model = PipelinedGroupBertForMaskedLM(config).parallelize().half().train()
    ```
    """

    def __init__(self, config):
        super().__init__(config)
        self.gather_indices = OnehotGather()

    def parallelize(self):
        """
        Transform the model to run in an IPU pipeline.
        - Adds pipeline stages to the model
        - (If enabled) Replaces the word embedding projection with a SerializedLinear layer
        - Adds recomputation checkpoints
        """
        super().parallelize()

        if self.ipu_config.embedding_serialization_factor > 1:
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

        layer_ipu = get_layer_ipu(self.ipu_config, self.bert.encoder.layer)

        logger.info("-------------------- Device Allocation --------------------")
        logger.info("Embedding  --> IPU 0")
        self.bert.embeddings = poptorch.BeginBlock(self.bert.embeddings, "Embedding", ipu_id=0)
        # Preventing the embeddings.LayerNorm from being outlined with the encoder.layer.LayerNorm
        # improves the tile mapping of the pipeline stashes
        hs = outline_attribute(self.bert.embeddings.LayerNorm, "embeddings")
        self._hooks.extend(hs)

        for index, layer in enumerate(self.bert.encoder.layer):
            ipu = layer_ipu[index]
            if self.ipu_config.recompute_checkpoint_every_layer:
                h = recomputation_checkpoint(layer)
                self._hooks.append(h)
            self.bert.encoder.layer[index] = poptorch.BeginBlock(layer, f"Encoder{index}", ipu_id=ipu)
            logger.info(f"Encoder {index:<2} --> IPU {ipu}")

        logger.info("Classifier --> IPU 0")
        self.cls = poptorch.BeginBlock(self.cls, "Classifier", ipu_id=0)
        logger.info("-----------------------------------------------------------")
        return self

    def deparallelize(self):
        """
        Undo the changes to the model done by `parallelize`.
        You should call this before doing `save_pretrained` so that the `model.state_dict` is
        compatible with the original model.
        """
        super().deparallelize()

        if self.ipu_config.embedding_serialization_factor > 1:
            decoder = nn.Linear(
                self.config.hidden_size,
                self.config.vocab_size,
                bias=True,
            )
            decoder.load_state_dict(self.cls.predictions.decoder.state_dict())
            self.cls.predictions.decoder = decoder
            self.tie_weights()
        return self

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.training:
            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output = outputs[0]

            if hasattr(self.config, "max_num_masked_tokens"):
                # Select only the masked tokens for the classifier
                labels, positions = torch.topk(labels, k=self.config.max_num_masked_tokens, dim=1)
                sequence_output = self.gather_indices(sequence_output, positions)

            prediction_scores = self.cls(sequence_output)
            outputs = (prediction_scores,) + outputs[2:]

            masked_lm_loss = F.cross_entropy(
                prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
            ).float()
            # When training only return the loss
            if return_dict:
                return MaskedLMOutput(loss=masked_lm_loss)
            else:
                return (masked_lm_loss,)

        else:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                labels=labels,
                return_dict=return_dict,
            )


class BertPipelineMixin(PipelineMixin):
    def parallelize(self):
        """
        Transform the model to run in an IPU pipeline.
        - Adds pipeline stages to the model
        - (If enabled) Replaces the word embedding with a SerializedEmbedding
        - Adds recomputation checkpoints
        """
        super().parallelize()

        layer_ipu = get_layer_ipu(self.ipu_config, self.bert.encoder.layer)

        logger.info("-------------------- Device Allocation --------------------")
        logger.info("Embedding --> IPU 0")
        if self.ipu_config.embedding_serialization_factor > 1:
            self.bert.embeddings.word_embeddings = SerializedEmbedding.from_model(
                self.bert.embeddings.word_embeddings, self.ipu_config.embedding_serialization_factor
            )
        self.bert.embeddings = poptorch.BeginBlock(self.bert.embeddings, "Embedding", ipu_id=0)
        hs = outline_attribute(self.bert.embeddings.LayerNorm, "embedding")
        self._hooks.extend(hs)

        for index, layer in enumerate(self.bert.encoder.layer):
            ipu = layer_ipu[index]
            if self.ipu_config.recompute_checkpoint_every_layer and index != self.config.num_hidden_layers - 1:
                h = recomputation_checkpoint(layer)
                self._hooks.append(h)
            self.bert.encoder.layer[index] = poptorch.BeginBlock(layer, f"Encoder{index}", ipu_id=ipu)
            logger.info(f"Encoder {index:<2} --> IPU {ipu}")
        return self

    def deparallelize(self):
        """
        Undo the changes to the model done by `parallelize`.
        You should call this before doing `save_pretrained` so that the `model.state_dict` is
        compatible with the original model.
        """
        super().deparallelize()

        # Deserialize the serialized word embedding
        if self.ipu_config.embedding_serialization_factor > 1:
            self.bert.embeddings.word_embeddings = self.bert.embeddings.word_embeddings.to_model()
        return self


@register(GroupBertForSequenceClassification)
class PipelinedGroupBertForSequenceClassification(GroupBertForSequenceClassification, BertPipelineMixin):
    """
    GroupBertForSequenceClassification transformed to run in an IPU pipeline.

    Recommended usage:
    ```
    model = PipelinedGroupBertForSequenceClassification(config).parallelize().half()
    ```
    """

    def parallelize(self):
        super().parallelize()
        last_ipu = self.ipu_config._ipus_per_replica - 1
        logger.info(f"Classifier Output --> IPU {last_ipu}")
        self.classifier = poptorch.BeginBlock(self.classifier, "Classifier Output", ipu_id=last_ipu)
        logger.info("-----------------------------------------------------------")
        return self


@register(GroupBertForMultipleChoice)
class PipelinedGroupBertForMultipleChoice(GroupBertForMultipleChoice, BertPipelineMixin):
    """
    GroupBertForMultipleChoice transformed to run in an IPU pipeline.

    Recommended usage:
    ```
    model = PipelinedGroupBertForMultipleChoice(config).parallelize().half()
    ```
    """

    def parallelize(self):
        super().parallelize()
        last_ipu = self.ipu_config._ipus_per_replica - 1
        logger.info(f"Classifier Output --> IPU {last_ipu}")
        self.classifier = poptorch.BeginBlock(self.classifier, "Classifier Output", ipu_id=last_ipu)
        logger.info("-----------------------------------------------------------")
        return self


@register(GroupBertForTokenClassification)
class PipelinedGroupBertForTokenClassification(GroupBertForTokenClassification, BertPipelineMixin):
    """
    GroupBertForTokenClassification transformed to run in an IPU pipeline.

    Recommended usage:
    ```
    model = PipelinedGroupBertForTokenClassification(config).parallelize().half()
    ```
    """

    def parallelize(self):
        super().parallelize()
        last_ipu = self.ipu_config._ipus_per_replica - 1
        logger.info(f"Classifier Output --> IPU {last_ipu}")
        self.classifier = poptorch.BeginBlock(self.classifier, "Classifier Output", ipu_id=last_ipu)
        logger.info("-----------------------------------------------------------")
        return self


@register(GroupBertForQuestionAnswering)
class PipelinedGroupBertForQuestionAnswering(GroupBertForQuestionAnswering, BertPipelineMixin):
    """
    GroupBertForQuestionAnswering transformed to run in an IPU pipeline.

    Recommended usage:
    ```
    model = PipelinedGroupBertForQuestionAnswering(config).parallelize().half()
    ```
    """

    def parallelize(self):
        super().parallelize()
        last_ipu = self.ipu_config._ipus_per_replica - 1
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
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
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

        output = super().forward(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
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
