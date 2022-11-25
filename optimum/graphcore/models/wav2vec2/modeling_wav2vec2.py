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

from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

import poptorch
from optimum.utils import logging
from transformers import Wav2Vec2ForPreTraining, Wav2Vec2Model
from transformers.modeling_outputs import CausalLMOutput
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Adapter,
    Wav2Vec2Encoder,
    Wav2Vec2EncoderStableLayerNorm,
    Wav2Vec2ForCTC,
    Wav2Vec2ForPreTrainingOutput,
    Wav2Vec2GumbelVectorQuantizer,
)

from ...modeling_utils import PipelineMixin, get_layer_ipu, recomputation_checkpoint, register
from .ipu_gumbel_vector_quantizer import IPUWav2Vec2GumbelVectorQuantizer
from .ipu_layer_drop import IPUWav2Vec2Adapter, IPUWav2Vec2Encoder, IPUWav2Vec2EncoderStableLayerNorm


logger = logging.get_logger(__name__)


class IPUWav2Vec2Model(Wav2Vec2Model):
    def _get_feature_vector_attention_mask(
        self,
        feature_vector_length: int,
        attention_mask: torch.LongTensor,
        add_adapter=None,
    ):
        # Effectively attention_mask.sum(-1), but not inplace to be able to run
        # on inference mode.
        # non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]
        # non_padded_lengths = attention_mask.cumsum(dim=-1)[:, 249999]
        non_padded_lengths = attention_mask.sum(dim=-1)

        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths, add_adapter=add_adapter)
        output_lengths = output_lengths.to(torch.long)

        batch_size = attention_mask.shape[0]

        attention_mask = torch.zeros(
            (batch_size, feature_vector_length),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        # these two operations makes sure that all values before the output lengths idxs are attended to
        attention_mask[
            (
                torch.arange(attention_mask.shape[0], device=attention_mask.device),
                output_lengths - 1,
            )
        ] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask


@register(Wav2Vec2ForPreTraining)
class PipelinedWav2Vec2ForPreTraining(Wav2Vec2ForPreTraining, PipelineMixin):
    def change_wav2vec2_encoder_class(self, restore: bool):
        """Changes the encoder class to update its forward pass so that it uses our custom version.

        Args:
            restore: whether to restore the encoder to its original version or not.
        """
        if self.config.do_stable_layer_norm:
            new_cls = Wav2Vec2EncoderStableLayerNorm if restore else IPUWav2Vec2EncoderStableLayerNorm
        else:
            new_cls = Wav2Vec2Encoder if restore else IPUWav2Vec2Encoder
        self.wav2vec2.encoder.__class__ = new_cls

    def change_wav2vec2_adapter_class(self, restore: bool):
        """Changes the adapter class to update its forward pass so that it uses our custom version.

        Args:
            restore: whether to restore the adapter to its original version or not.
        """
        if self.config.add_adapter:
            self.wav2vec2.adapter.__class__ = Wav2Vec2Adapter if restore else IPUWav2Vec2Adapter

    def change_quantizer_class(self, restore: bool):
        """Changes the quantizer class to update its forward pass so that it uses our custom version.

        Args:
            restore: whether to restore the quantizer to its original version or not.
        """
        self.quantizer.__class__ = Wav2Vec2GumbelVectorQuantizer if restore else IPUWav2Vec2GumbelVectorQuantizer

    def change_conv_eps(self, restore: bool):
        """Changes the epsilons in the layer norms of the conv layers to a value suitable for float16.

        Args:
            restore: whether to restore the epsilons to their original version or not.
        """
        if self.config.feat_extract_norm != "layer":
            # In this case there is no layer norm in the conv layers
            return
        if restore:
            for i, conv_layer in enumerate(self.wav2vec2.feature_extractor.conv_layers):
                # Restore the original values
                conv_layer.layer_norm.eps = self.original_eps[i]
        else:
            self.original_eps = []
            eps = 1e-4
            for conv_layer in self.wav2vec2.feature_extractor.conv_layers:
                # Save the original values, to restore later
                self.original_eps.append(conv_layer.layer_norm.eps)
                conv_layer.layer_norm.eps = eps

    def _add_begin_block(self, module, name, ipu_id):
        poptorch.BeginBlock(module, name, ipu_id)

    def parallelize(self):
        """
        Transform the model to run in an IPU pipeline.
        - Adds pipeline stages to the model
        - Replaces some layers with IPU-specialised ones
        - Set eps to a stable value in float16

        Recommended usage:
        ```
        model = PipelinedWav2Vec2ForPreTraining(config).parallelize().half()
        ```
        """
        super().parallelize()

        self.wav2vec2.__class__ = IPUWav2Vec2Model
        self.change_wav2vec2_encoder_class(False)
        self.change_wav2vec2_adapter_class(False)
        self.change_quantizer_class(False)
        self.change_conv_eps(False)

        logger.info("---------- Device Allocation -----------")
        layers = []
        # Conv layers
        for index, layer in enumerate(self.wav2vec2.feature_extractor.conv_layers):
            layers.append((f"Conv {index:<2}", layer))
        # Positional Embedding
        layers.append(("Positional Embedding", self.wav2vec2.encoder.pos_conv_embed))
        # Encoder layers
        for index, layer in enumerate(self.wav2vec2.encoder.layers):
            recomputation_checkpoint(layer)
            layers.append((f"Encoder {index:<2}", layer))
        # Project Hidden
        layers.append(("Project Hidden", self.project_hid))
        # Quantizer
        layers.append(("Quantizer", self.quantizer))
        # Project Quantizer
        layers.append(("Project Quantizer", self.project_q))

        layer_ipu = get_layer_ipu(self.ipu_config.layers_per_ipu, layers)

        for i, (name, layer) in enumerate(layers):
            logger.info(f"{name} --> IPU {layer_ipu[i]}")
            self._add_begin_block(layer, name, ipu_id=layer_ipu[i])

        logger.info("---------------------------------------")

    def deparallelize(self):
        """
        Undo the changes to the model done by `parallelize`.
        You should call this before doing `save_pretrained` so that the `model.state_dict` is
        fully compatible with `transformers.Wav2Vec2ForPreTraining`.
        """
        super().deparallelize()
        self.change_wav2vec2_encoder_class(True)
        self.change_wav2vec2_adapter_class(True)
        self.change_quantizer_class(True)
        self.change_conv_eps(True)
        self.wav2vec2.__class__ = Wav2Vec2Model
        return self

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        gumbel_temperature: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.BoolTensor] = None,
        sampled_negative_indices: Optional[torch.BoolTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        reduce_selector: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Wav2Vec2ForPreTrainingOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if mask_time_indices is not None:
            mask_time_indices = mask_time_indices.to(torch.bool)

        if gumbel_temperature is None:
            gumbel_temperature = torch.tensor(
                self.quantizer.temperature, device=input_values.device, dtype=input_values.dtype
            )

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            mask_time_indices=mask_time_indices,
            return_dict=return_dict,
        )

        transformer_features, extract_features = outputs[0], outputs[1]

        if attention_mask is not None:
            # compute reduced attention_mask correponding to feature vectors
            attention_mask = self.wav2vec2._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        # GC. remove a (static sized) portion of the output tensors at unmasked indices
        # unmasked indices do not contribute to loss. removing them now alleviates memory requirements
        if reduce_selector is not None:
            batch_size, sequence_length, feature_size = extract_features.shape
            cropped_length = reduce_selector.shape[1]

            if batch_size > 1:
                reduce_selector += torch.arange(batch_size, device=input_values.device).unsqueeze(1) * sequence_length
            mask_time_indices = mask_reduced.to(torch.bool)

            extract_features = extract_features.view(-1, feature_size)[reduce_selector.long().view(-1)]
            extract_features = extract_features.reshape(batch_size, cropped_length, feature_size)

            _, _, feature_size = transformer_features.shape
            transformer_features = transformer_features.view(-1, feature_size)[reduce_selector.long().view(-1)]
            transformer_features = transformer_features.reshape(batch_size, cropped_length, feature_size)

        # 1. project all transformed features (including masked) to final vq dim
        transformer_features = self.project_hid(transformer_features)

        # 2. quantize all (unmasked) extracted features and project to final vq dim
        extract_features = self.dropout_features(extract_features)

        if isinstance(self.quantizer, IPUWav2Vec2GumbelVectorQuantizer):
            quantized_features, code_perplexity, prob_perplexity = self.quantizer(
                extract_features,
                gumbel_temperature.mean(),
                mask_time_indices=mask_time_indices,
            )
        else:
            quantized_features, code_perplexity = self.quantizer(
                extract_features,
                mask_time_indices=mask_time_indices,
            )
            prob_perplexity = None

        quantized_features = self.project_q(quantized_features)

        loss = contrastive_loss = diversity_loss = None
        if sampled_negative_indices is not None:
            batch_size, sequence_length, hidden_size = quantized_features.shape

            # for training, we sample negatives
            # 3. sample K negatives (distractors) quantized states for contrastive loss
            # if attention_mask is passed, make sure that padded feature vectors cannot be sampled
            # sample negative quantized vectors BTC => (BxT)C
            # Moved the negative sampling batch offsetting into the model
            if batch_size > 1:
                sampled_negative_indices += (
                    torch.arange(batch_size, device=input_values.device)[:, None, None] * sequence_length
                )
            negative_quantized_features = quantized_features.view(-1, hidden_size)[
                sampled_negative_indices.long().view(-1)
            ]
            negative_quantized_features = negative_quantized_features.view(
                batch_size, sequence_length, -1, hidden_size
            ).permute(2, 0, 1, 3)

            # 4. compute logits, corresponding to `logs = sim(c_t, [q_t, \sim{q}_t]) / \kappa`
            # of equation (3) in https://arxiv.org/pdf/2006.11477.pdf
            logits = self.compute_contrastive_logits(
                quantized_features[None, :],
                negative_quantized_features,
                transformer_features,
                self.config.contrastive_logits_temperature,
            )

            # 5. if a negative vector is identical to the positive (i.e. when codebook utilization is low),
            # its cosine similarity will be masked
            neg_is_pos = (quantized_features == negative_quantized_features).all(-1)

            neg_is_pos = F.pad(neg_is_pos, (0, 0, 0, 0, 1, 0))
            logits = logits.masked_fill(neg_is_pos, -1e3)

            # 6. compute contrastive loss \mathbf{L}_m = cross_entropy(logs) =
            # -log(exp(sim(c_t, q_t)/\kappa) / \sum_{\sim{q}} exp(sim(c_t, \sim{q})/\kappa))
            logits = logits.permute(1, 2, 0).reshape(batch_size * sequence_length, -1)
            target = ((1 - mask_time_indices.long()) * -100).flatten()

            contrastive_loss = F.cross_entropy(logits.float(), target, reduction="sum")

            # 7. compute diversity loss: \mathbf{L}_d
            num_codevectors = self.config.num_codevectors_per_group * self.config.num_codevector_groups
            diversity_loss = ((num_codevectors - prob_perplexity) / num_codevectors) * mask_time_indices.sum()

            # 8. \mathbf{L} = \mathbf{L}_m + \alpha * \mathbf{L}_d
            loss = contrastive_loss + self.config.diversity_loss_weight * diversity_loss

        if not return_dict:
            if loss is not None:
                return (
                    loss,
                    transformer_features,
                    quantized_features,
                    code_perplexity,
                    prob_perplexity,
                ) + outputs[2:]
            return (
                transformer_features,
                quantized_features,
                code_perplexity,
                prob_perplexity,
            ) + outputs[2:]

        return Wav2Vec2ForPreTrainingOutput(
            loss=loss,
            projected_states=transformer_features,
            projected_quantized_states=quantized_features,
            codevector_perplexity=code_perplexity,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            contrastive_loss=contrastive_loss,
            diversity_loss=diversity_loss,
        )

    @staticmethod
    def compute_contrastive_logits(
        target_features: torch.FloatTensor,
        negative_features: torch.FloatTensor,
        predicted_features: torch.FloatTensor,
        temperature: int = 0.1,
    ):
        """
        Compute logits for contrastive loss based using cosine similarity as the distance measure between
        `[positive_feature, negative_features]` and `[predicted_features]`. Additionally, temperature can be applied.
        """
        target_features = torch.cat([target_features, negative_features], dim=0)

        logits = torch.cosine_similarity(
            predicted_features.float(), target_features.float(), dim=-1, eps=1e-4
        ).type_as(target_features)

        # apply temperature
        logits = logits / temperature
        return logits


@register(Wav2Vec2ForCTC)
class PipelinedWav2Vec2ForCTC(Wav2Vec2ForCTC, PipelineMixin):
    def change_wav2vec2_encoder_class(self, restore: bool):
        """Changes the encoder class to update its forward pass so that it uses our custom version.

        Args:
            restore: whether to restore the encoder to its original version or not.
        """
        if self.config.do_stable_layer_norm:
            new_cls = Wav2Vec2EncoderStableLayerNorm if restore else IPUWav2Vec2EncoderStableLayerNorm
        else:
            new_cls = Wav2Vec2Encoder if restore else IPUWav2Vec2Encoder
        self.wav2vec2.encoder.__class__ = new_cls

    def change_wav2vec2_adapter_class(self, restore: bool):
        """Changes the adapter class to update its forward pass so that it uses our custom version.

        Args:
            restore: whether to restore the adapter to its original version or not.
        """
        if self.config.add_adapter:
            self.wav2vec2.adapter.__class__ = Wav2Vec2Adapter if restore else IPUWav2Vec2Adapter

    def change_conv_eps(self, restore: bool):
        """Changes the epsilons in the layer norms of the conv layers to a value suitable for float16.

        Args:
            restore: whether to restore the epsilons to their original version or not.
        """
        if self.config.feat_extract_norm != "layer":
            # In this case there is no layer norm in the conv layers
            return
        if restore:
            for i, conv_layer in enumerate(self.wav2vec2.feature_extractor.conv_layers):
                # Restore the original values
                conv_layer.layer_norm.eps = self.original_eps[i]
        else:
            self.original_eps = []
            eps = 1e-4
            for conv_layer in self.wav2vec2.feature_extractor.conv_layers:
                # Save the original values, to restore later
                self.original_eps.append(conv_layer.layer_norm.eps)
                conv_layer.layer_norm.eps = eps

    def _add_begin_block(self, module, name, ipu_id):
        poptorch.BeginBlock(module, name, ipu_id)

    def parallelize(self):
        """
        Transform the model to run in an IPU pipeline.
        - Adds pipeline stages to the model
        - Replaces some layers with IPU-specialised ones
        - Set eps to a stable value in float16

        Recommended usage:
        ```
        model = PipelinedWav2Vec2ForPreTraining(config).parallelize().half()
        ```
        """
        super().parallelize()

        self.wav2vec2.__class__ = IPUWav2Vec2Model
        self.freeze_feature_encoder()
        self.change_wav2vec2_encoder_class(False)
        self.change_wav2vec2_adapter_class(False)
        self.change_conv_eps(False)

        if self.ipu_config.ipus_per_replica != 1:
            logger.info("---------- Device Allocation -----------")
            layers = []
            # Conv layers
            for index, layer in enumerate(self.wav2vec2.feature_extractor.conv_layers):
                layers.append((f"Conv {index:<2}", layer))
            # Positional Embedding
            layers.append(("Positional Embedding", self.wav2vec2.encoder.pos_conv_embed))
            # Encoder layers
            for index, layer in enumerate(self.wav2vec2.encoder.layers):
                recomputation_checkpoint(layer)
                layers.append((f"Encoder {index:<2}", layer))
            # Project Hidden
            layers.append(("Project Hidden", self.lm_head))

            layer_ipu = get_layer_ipu(self.ipu_config.layers_per_ipu, layers)

            for i, (name, layer) in enumerate(layers):
                logger.info(f"{name} --> IPU {layer_ipu[i]}")
                self._add_begin_block(layer, name, ipu_id=layer_ipu[i])

            logger.info("---------------------------------------")

    def deparallelize(self):
        """
        Undo the changes to the model done by `parallelize`.
        You should call this before doing `save_pretrained` so that the `model.state_dict` is
        fully compatible with `transformers.Wav2Vec2ForPreTraining`.
        """
        super().deparallelize()
        self.change_wav2vec2_encoder_class(True)
        self.change_wav2vec2_adapter_class(True)
        self.change_conv_eps(True)
        self.wav2vec2.__class__ = Wav2Vec2Model
        return self

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)

            # ctc_loss doesn't support fp16
            log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1).transpose(0, 1)

            loss_fn = torch.nn.CTCLoss(blank=self.config.pad_token_id, reduction=self.config.ctc_loss_reduction)
            loss = loss_fn(log_probs.float(), labels, input_lengths, target_lengths)
            loss = poptorch.identity_loss(loss, "sum")

        if not return_dict:
            if loss is not None:
                return loss, logits
            return (logits, hidden_states)
        return CausalLMOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states)


def _sample_negative_indices(
    features_shape: Tuple,
    num_negatives: int,
    mask_time_indices: Optional[np.ndarray] = None,
):
    """
    Sample `num_negatives` vectors from feature vectors.
    """
    batch_size, sequence_length = features_shape

    # generate indices of the positive vectors themselves, repeat them `num_negatives` times
    sequence_length_range = np.arange(sequence_length)

    # get `num_negatives` random vector indices from the same utterance
    sampled_negative_indices = np.zeros(shape=(batch_size, sequence_length, num_negatives), dtype=np.int32)

    mask_time_indices = (
        mask_time_indices.astype(np.bool) if mask_time_indices is not None else np.ones(features_shape, dtype=np.bool)
    )

    for batch_idx in range(batch_size):
        high = mask_time_indices[batch_idx].sum() - 1
        mapped_masked_indices = sequence_length_range[mask_time_indices[batch_idx]]

        feature_indices = np.broadcast_to(np.arange(high + 1)[:, None], (high + 1, num_negatives))
        sampled_indices = np.random.randint(0, high, size=(high + 1, num_negatives))
        # avoid sampling the same positive vector, but keep the distribution uniform
        sampled_indices[sampled_indices >= feature_indices] += 1

        # remap to actual indices
        sampled_negative_indices[batch_idx][mask_time_indices[batch_idx]] = mapped_masked_indices[sampled_indices]

        # Moved the offsetting into the model to stop issues with gradient accumulation
        # sampled_negative_indices[batch_idx] += batch_idx * sequence_length

    return sampled_negative_indices
