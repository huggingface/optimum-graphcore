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
from transformers import GPT2ForSequenceClassification, GPT2ForTokenClassification, GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

from ...modeling_utils import (
    PipelineMixin,
    SerializedEmbedding,
    SerializedLinear,
    get_layer_ipu,
    outline_attribute,
    recomputation_checkpoint,
    register,
)
from .optimized_gpt2_attn import OptimizedGPT2Attention


logger = logging.get_logger(__name__)


class GPT2PipelineMixin(PipelineMixin):
    def parallelize(self):
        """
        Transform the GPT2 model body to run in an IPU pipeline.
        - Adds pipeline stages to the model
        - (If enabled) Replaces the word embedding with a SerializedEmbedding
        - Adds recomputation checkpoints
        """
        super().parallelize()

        # Use optimized attention
        for layer in self.transformer.h:
            layer.attn.__class__ = OptimizedGPT2Attention

        if self.ipu_config.embedding_serialization_factor > 1:
            # Resize token embedding using padding if vocab_size is not a multiple of embedding_serialization_factor.
            self.actual_vocab_size = self.config.vocab_size
            new_vocab_size = (
                math.ceil(self.config.vocab_size / self.ipu_config.embedding_serialization_factor)
                * self.ipu_config.embedding_serialization_factor
            )
            if new_vocab_size > self.actual_vocab_size:
                self.resize_token_embeddings(new_vocab_size)

            self.transformer.wte = SerializedEmbedding(
                self.transformer.wte, self.ipu_config.embedding_serialization_factor
            )

        layer_ipu = get_layer_ipu(self.ipu_config.layers_per_ipu)

        logger.info("-------------------- Device Allocation --------------------")
        logger.info("Embedding  --> IPU 0")
        self.transformer.wte = poptorch.BeginBlock(self.transformer.wte, "Token embedding", ipu_id=0)
        self.transformer.wpe = poptorch.BeginBlock(self.transformer.wpe, "Position embedding", ipu_id=0)
        hs = outline_attribute(self.transformer.ln_f, "LayerNorm")
        self._hooks.extend(hs)

        for index, layer in enumerate(self.transformer.h):
            ipu = layer_ipu[index]
            if self.ipu_config.recompute_checkpoint_every_layer and index != self.config.num_hidden_layers - 1:
                h = recomputation_checkpoint(layer)
                self._hooks.append(h)
            self.transformer.h[index] = poptorch.BeginBlock(layer, f"Layer{index}", ipu_id=ipu)
            logger.info(f"Layer {index:<2}   --> IPU {ipu}")
        return self

    def deparallelize(self):
        """
        Undo the changes to the model done by `parallelize`.
        You should call this before doing `save_pretrained` so that the `model.state_dict` is
        fully compatible with `transformers` models.
        """
        super().deparallelize()

        if self.ipu_config.embedding_serialization_factor > 1:
            # Deserialize the serialized word embedding
            self.transformer.wte = self.transformer.wte.deserialize()

            # Resize token embeddings back to origianl vocab_size
            if self.config.vocab_size > self.actual_vocab_size:
                self.resize_token_embeddings(self.actual_vocab_size)

        # Switch back to non-optimized attention
        for layer in self.transformer.h:
            layer.attn.__class__ = GPT2Attention
        return self


@register(GPT2LMHeadModel)
class PipelinedGPT2LMHeadModel(GPT2LMHeadModel, PipelineMixin):
    def parallelize(self):
        """
        Transform the model to run in an IPU pipeline.
        - Adds pipeline stages to the model
        - Adds recomputation checkpoints

        Recommended usage:
        ```
        model = PipelinedGPT2LMHeadModel(config).parallelize().half()
        ```
        """
        PipelineMixin.parallelize(self)

        # Use optimized attention
        for layer in self.transformer.h:
            layer.attn.__class__ = OptimizedGPT2Attention

        if self.ipu_config.embedding_serialization_factor > 1:
            # Resize token embedding using padding if vocab_size is not a multiple of embedding_serialization_factor.
            self.actual_vocab_size = self.config.vocab_size
            new_vocab_size = (
                math.ceil(self.config.vocab_size / self.ipu_config.embedding_serialization_factor)
                * self.ipu_config.embedding_serialization_factor
            )
            if new_vocab_size > self.actual_vocab_size:
                # There is a tie_weights operation in resize_token_embeddings so the lm_head's weight is also resized.
                self.resize_token_embeddings(new_vocab_size)

            serialized_lm_head = SerializedLinear(
                self.config.n_embd,
                self.config.vocab_size,  # Note that if padding is done, self.config.vocab_size == new_vocab_size
                self.ipu_config.embedding_serialization_factor,
                bias=False,
                mode=poptorch.MatMulSerializationMode.OutputChannels,
            )
            serialized_lm_head.load_state_dict(self.lm_head.state_dict())
            self.lm_head = serialized_lm_head
            self.tie_weights()

        layer_ipu = get_layer_ipu(self.ipu_config.layers_per_ipu)

        logger.info("-------------------- Device Allocation --------------------")
        logger.info("Token Embedding     --> IPU 0")
        self.transformer.wte = poptorch.BeginBlock(self.transformer.wte, "Token embedding", ipu_id=0)
        logger.info("Position Embedding  --> IPU 1")
        self.transformer.wpe = poptorch.BeginBlock(self.transformer.wpe, "Position embedding", ipu_id=1)
        hs = outline_attribute(self.transformer.ln_f, "LayerNorm")
        self._hooks.extend(hs)

        for index, layer in enumerate(self.transformer.h):
            ipu = layer_ipu[index]
            if self.ipu_config.recompute_checkpoint_every_layer:
                h = recomputation_checkpoint(layer)
                self._hooks.append(h)
            self.transformer.h[index] = poptorch.BeginBlock(layer, f"Layer{index}", ipu_id=ipu)
            logger.info(f"Layer {index:<2}            --> IPU {ipu}")

        logger.info("Head                --> IPU 0")
        self.lm_head = poptorch.BeginBlock(self.lm_head, "LM head", ipu_id=0)
        logger.info("-----------------------------------------------------------")
        return self

    def deparallelize(self):
        PipelineMixin.deparallelize(self)

        if self.ipu_config.embedding_serialization_factor > 1:
            # Deserialize the serialized linear layer
            old_lm_head = nn.Linear(
                self.config.n_embd,
                self.config.vocab_size,  # Note that if padding is done, self.config.vocab_size == new_vocab_size
                bias=False,
            )
            old_lm_head.load_state_dict(self.lm_head.state_dict())
            self.lm_head = old_lm_head
            self.tie_weights()

            # Resize token embeddings back to origianl vocab_size.
            # There is a tie_weights operation in resize_token_embeddings so the lm_head's weight is also resized.
            if self.config.vocab_size > self.actual_vocab_size:
                self.resize_token_embeddings(self.actual_vocab_size)

        # Switch back to non-optimized attention
        for layer in self.transformer.h:
            layer.attn.__class__ = GPT2Attention
        return self

    def forward(self, input_ids, attention_mask, labels=None):
        transformer_outputs = self.transformer(input_ids, attention_mask=attention_mask)
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)
        if self.ipu_config.embedding_serialization_factor > 1 and self.config.vocab_size > self.actual_vocab_size:
            # Ignore the padding logits. Use masking because in-place modification on a slice is not supported yet.
            padding_mask = torch.cat(
                (
                    torch.ones(self.actual_vocab_size),
                    torch.zeros(self.config.vocab_size - self.actual_vocab_size),
                )
            )
            lm_logits = lm_logits * padding_mask + (1 - padding_mask) * -10000.0

            # TODO: Use the following line instead to ignore the padding logits
            # lm_logits[:, :, self.actual_vocab_size:] = -10000

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n. Use roll() + ignore_index instead of slicing for better efficiency on IPUs.
            labels = torch.roll(labels, -1, 1)
            # By default the ignore_index of CrossEntropyLoss is -100
            labels[:, -1] = -100
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        if self.ipu_config.embedding_serialization_factor > 1 and self.config.vocab_size > self.actual_vocab_size:
            output = (lm_logits[:, :, : self.actual_vocab_size],) + transformer_outputs[1:]
        else:
            output = (lm_logits,) + transformer_outputs[1:]
        return (loss,) if loss is not None else output


@register(GPT2ForSequenceClassification)
class PipelinedGPT2ForSequenceClassification(GPT2ForSequenceClassification, GPT2PipelineMixin):
    def parallelize(self):
        super().parallelize()
        last_ipu = self.ipu_config.ipus_per_replica - 1
        logger.info(f"Head       --> IPU {last_ipu}")
        self.score = poptorch.BeginBlock(self.score, "Score", ipu_id=last_ipu)
        logger.info("-----------------------------------------------------------")
        return self

    def forward(self, input_ids, attention_mask, labels=None):
        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=False,
        )
        # By default use_cache=True and the model would return past_key_values, which could be very large and cause OOM.
        # To prevent this we only return loss and logits during training and evaluation (i.e. when there are labels).
        if labels is not None:
            loss = output[0]
            logits = output[1]
        return (loss, logits) if labels is not None else output


@register(GPT2ForTokenClassification)
class PipelinedGPT2ForTokenClassification(GPT2ForTokenClassification, GPT2PipelineMixin):
    def parallelize(self):
        super().parallelize()
        last_ipu = self.ipu_config.ipus_per_replica - 1
        logger.info(f"Head       --> IPU {last_ipu}")
        self.classifier = poptorch.BeginBlock(self.classifier, "Classifier", ipu_id=last_ipu)
        logger.info("-----------------------------------------------------------")
        return self

    def forward(self, input_ids, attention_mask, labels=None):
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=False,
        )
