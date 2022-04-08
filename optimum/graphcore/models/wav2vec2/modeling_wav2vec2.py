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

from typing import Tuple, Optional
import numpy as np

import torch
import torch.nn.functional as F

import poptorch
from optimum.utils import logging
from transformers import (
    Wav2Vec2Model,
    Wav2Vec2ForPreTraining
)
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Encoder,
    Wav2Vec2EncoderStableLayerNorm,
    Wav2Vec2Adapter,
    Wav2Vec2GumbelVectorQuantizer,
    Wav2Vec2ForPreTrainingOutput
)
from .ipu_layer_drop import IPUWav2Vec2Encoder, IPUWav2Vec2EncoderStableLayerNorm, IPUWav2Vec2Adapter
from .ipu_gumbel_vector_quantizer import IPUWav2Vec2GumbelVectorQuantizer
from .ipu_wav2vec2_model import IPUWav2Vec2Model

from ...modeling_utils import PipelineMixin, register


logger = logging.get_logger(__name__)


@register(Wav2Vec2ForPreTraining)
class PipelinedWav2Vec2ForPreTraining(Wav2Vec2ForPreTraining, PipelineMixin):

    def change_wav2vec2_encoder_class(self, restore: bool):
        """Changes the encoder class to update its forward pass so that it uses our custom version.

        Args:
            restore: whether to restore the encoder to its original version or not.
        """
        if restore:
            if self.config.do_stable_layer_norm:
                encoder = Wav2Vec2EncoderStableLayerNorm(self.config)
            else:
                encoder = Wav2Vec2Encoder(self.config)
        else:
            # Inject IPU Layer Drop
            # We also pad the sequence length for self-attention
            # This makes the memory use across tiles more balanced
            if self.config.do_stable_layer_norm:
                encoder = IPUWav2Vec2EncoderStableLayerNorm(
                    self.config,
                    sequence_length_padding_divisor=4
                )
            else:
                encoder = IPUWav2Vec2Encoder(
                    self.config,
                    sequence_length_padding_divisor=4
                )
        self.wav2vec2.encoder = encoder


    def change_wav2vec2_adapter_class(self, restore: bool):
        """Changes the adapter class to update its forward pass so that it uses our custom version.

        Args:
            restore: whether to restore the adapter to its original version or not.
        """
        if self.config.add_adapter:
            if restore:
                adapter = Wav2Vec2Adapter(self.config)
            else:
                adapter = IPUWav2Vec2Adapter(self.config)
        else:
            adapter = None
        self.wav2vec2.adapter = adapter


    def change_quantizer_class(self, restore: bool):
        """Changes the quantizer class to update its forward pass so that it uses our custom version.

        Args:
            restore: whether to restore the quantizer to its original version or not.
        """
        if restore:
            quantizer = Wav2Vec2GumbelVectorQuantizer(self.config)
        else:
            quantizer = IPUWav2Vec2GumbelVectorQuantizer(self.config)
        self.quantizer = quantizer


    def change_conv_eps(self, restore: bool):
        """Changes the epsilons in the layer norms of the conv layers to a value suitable for float16.

        Args:
            restore: whether to restore the epsilons to their original version or not.
        """
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

        self._add_begin_block(
            self.wav2vec2.feature_extractor.conv_layers[0],
            name="Conv[0,2)", ipu_id=0
        )

        self._add_begin_block(
            self.wav2vec2.feature_extractor.conv_layers[2],
            name="Conv[2,3)", ipu_id=1
        )

        self._add_begin_block(
            self.wav2vec2.feature_extractor.conv_layers[3],
            name="Conv[3,7)+PCE", ipu_id=2
        )

        self._add_begin_block(
            self.wav2vec2.encoder.layers[0],
            name="EL[00,03)", ipu_id=3
        )

        self._add_begin_block(
            self.wav2vec2.encoder.layers[3],
            name="EL[03,06)", ipu_id=4
        )

        self._add_begin_block(
            self.wav2vec2.encoder.layers[6],
            name="EL[06,09)", ipu_id=5
        )

        self._add_begin_block(
            self.wav2vec2.encoder.layers[9],
            name="EL[09,12)+ELQ", ipu_id=6
        )

        self._add_begin_block(
            self.quantizer,
            name="Quantizer+Losses", ipu_id=7
        )


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
        input_values,
        attention_mask=None,
        mask_time_indices=None,
        sampled_negative_indices=None,
        gumbel_temperature=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        # Override the return_dict argument
        return_dict = False

        if mask_time_indices is not None:
            mask_time_indices = mask_time_indices.to(torch.bool)

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            mask_time_indices=mask_time_indices,
            return_dict=return_dict,
        )

        # 1. project all transformed features (including masked) to final vq dim
        transformer_features = self.project_hid(outputs[0])

        # 2. quantize all (unmasked) extracted features and project to final vq dim
        extract_features = self.dropout_features(outputs[1])

        if attention_mask is not None:
            # compute reduced attention_mask correponding to feature vectors
            attention_mask = self.wav2vec2._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        quantized_features, codevector_perplexity = self.quantizer(
            extract_features, gumbel_temperature, mask_time_indices=mask_time_indices
        )
        quantized_features = self.project_q(quantized_features)

        loss = contrastive_loss = diversity_loss = None
        if sampled_negative_indices is not None:
            batch_size, sequence_length, hidden_size = quantized_features.shape

            # for training, we sample negatives
            # 3. sample K negatives (distractors) quantized states for contrastive loss
            # if attention_mask is passed, make sure that padded feature vectors cannot be sampled
            # sample negative quantized vectors BTC => (BxT)C
            # Moved the negative sampling batch offsetting into the model
            # Commenting this out because of Poptorch issue. With batch size 1 it's not needed anyway
            # sampled_negative_indices += torch.arange(batch_size)[:, None, None] * sequence_length
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

            neg_is_pos = F.pad(neg_is_pos.type(torch.long), (0, 0, 0, 0, 1, 0)).type(torch.bool)
            logits = logits.masked_fill(neg_is_pos, -1e3)

            # 6. compute contrastive loss \mathbf{L}_m = cross_entropy(logs) =
            # -log(exp(sim(c_t, q_t)/\kappa) / \sum_{\sim{q}} exp(sim(c_t, \sim{q})/\kappa))
            logits = logits.permute(1, 2, 0).reshape(batch_size * sequence_length, -1)
            target = ((1 - mask_time_indices.long()) * -100).flatten()

            contrastive_loss = F.cross_entropy(logits.float(), target, reduction="sum")

            # 7. compute diversity loss: \mathbf{L}_d
            num_codevectors = self.config.num_codevectors_per_group * self.config.num_codevector_groups
            diversity_loss = ((num_codevectors - codevector_perplexity) / num_codevectors) * mask_time_indices.sum()

            # 8. \mathbf{L} = \mathbf{L}_m + \alpha * \mathbf{L}_d
            loss = contrastive_loss + self.config.diversity_loss_weight * diversity_loss

        if not return_dict:
            if loss is not None:
                return (loss, transformer_features, quantized_features, codevector_perplexity) + outputs[2:]
            return (transformer_features, quantized_features, codevector_perplexity) + outputs[2:]

        return Wav2Vec2ForPreTrainingOutput(
            loss=loss,
            projected_states=transformer_features,
            projected_quantized_states=quantized_features,
            codevector_perplexity=codevector_perplexity,
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

        logits = torch.cosine_similarity(predicted_features.float(), target_features.float(), dim=-1, eps=1e-4).type_as(
            target_features
        )

        # apply temperature
        logits = logits / temperature
        return logits


def _sample_negative_indices(
    features_shape: Tuple, num_negatives: int, mask_time_indices: Optional[np.ndarray] = None
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
