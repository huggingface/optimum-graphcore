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

from ...modeling_utils import (
    PipelineMixin,
    SerializedEmbedding,
    SerializedLinear,
    _get_layer_ipu,
    outline_attribute,
    recomputation_checkpoint,
    register,
)


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
        layer_ipu = _get_layer_ipu(self.config.layers_per_ipu)

        logger.info("-------------------- Device Allocation --------------------")
        logger.info("Embedding  --> IPU 0")
        if self.config.embedding_serialization_factor > 1:
            self.transformer.wte = SerializedEmbedding(
                self.transformer.wte, self.config.embedding_serialization_factor
            )
        self.transformer.wte = poptorch.BeginBlock(self.transformer.wte, "Token embedding", ipu_id=0)
        self.transformer.wpe = poptorch.BeginBlock(self.transformer.wpe, "Position embedding", ipu_id=0)
        hs = outline_attribute(self.transformer.ln_f, "LayerNorm")
        self._hooks.extend(hs)

        for index, layer in enumerate(self.transformer.h):
            ipu = layer_ipu[index]
            if self.config.recompute_checkpoint_every_layer and index != self.config.num_hidden_layers - 1:
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
        # Deserialize the serialized word embedding
        if self.config.embedding_serialization_factor > 1:
            self.transformer.wte = self.transformer.wte.deserialize()
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
        if self.config.embedding_serialization_factor > 1:
            # Resize token embedding using padding if vocab_size is not a multiple of embedding_serialization_factor
            self.actual_vocab_size = self.config.vocab_size
            new_vocab_size = (
                math.ceil(self.config.vocab_size / self.config.embedding_serialization_factor)
                * self.config.embedding_serialization_factor
            )
            if self.config.vocab_size % self.config.embedding_serialization_factor == 0:
                assert self.actual_vocab_size == new_vocab_size
            self.resize_token_embeddings(new_vocab_size)
            serialized_decoder = SerializedLinear(
                self.config.n_embd,
                self.config.vocab_size,
                self.config.embedding_serialization_factor,
                bias=False,
                mode=poptorch.MatMulSerializationMode.OutputChannels,
            )
            serialized_decoder.load_state_dict(self.lm_head.state_dict())
            self.lm_head = serialized_decoder
            self.tie_weights()

        PipelineMixin.parallelize(self)
        layer_ipu = _get_layer_ipu(self.config.layers_per_ipu)

        logger.info("-------------------- Device Allocation --------------------")
        logger.info("Token Embedding     --> IPU 0")
        self.transformer.wte = poptorch.BeginBlock(self.transformer.wte, "Token embedding", ipu_id=0)
        logger.info("Position Embedding  --> IPU 1")
        self.transformer.wpe = poptorch.BeginBlock(self.transformer.wpe, "Position embedding", ipu_id=1)
        hs = outline_attribute(self.transformer.ln_f, "LayerNorm")
        self._hooks.extend(hs)

        for index, layer in enumerate(self.transformer.h):
            ipu = layer_ipu[index]
            if self.config.recompute_checkpoint_every_layer:
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
        if self.config.embedding_serialization_factor > 1:
            # Resize token embeddings back to origianl vocab_size
            self.resize_token_embeddings(self.actual_vocab_size)

    def forward(self, input_ids, attention_mask, labels=None):
        transformer_outputs = self.transformer(input_ids, attention_mask=attention_mask)
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)
        if self.config.embedding_serialization_factor > 1:
            lm_logits = lm_logits[:, :, 0 : self.actual_vocab_size]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            labels = torch.roll(labels, -1, 1)
            # By default ignore_index = -100
            labels[:, -1] = -100
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        return loss


@register(GPT2ForSequenceClassification)
class PipelinedGPT2ForSequenceClassification(GPT2ForSequenceClassification, GPT2PipelineMixin):
    def parallelize(self):
        super().parallelize()
        last_ipu = self.config.ipus_per_replica - 1
        logger.info(f"Head       --> IPU {last_ipu}")
        self.score = poptorch.BeginBlock(self.score, "Score", ipu_id=last_ipu)
        logger.info("-----------------------------------------------------------")
        return self

    def forward(self, input_ids, attention_mask, labels=None):
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
            return_dict=False,
        )


@register(GPT2ForTokenClassification)
class PipelinedGPT2ForTokenClassification(GPT2ForTokenClassification, GPT2PipelineMixin):
    def parallelize(self):
        super().parallelize()
        last_ipu = self.config.ipus_per_replica - 1
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
