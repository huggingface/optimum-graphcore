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

import poptorch
from optimum.utils import logging
from transformers import (
    GPT2LMHeadModel,
    GPT2ForSequenceClassification,
    GPT2ForTokenClassification,
)

from ...modeling_utils import PipelineMixin, register


logger = logging.get_logger(__name__)


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


def recomputation_checkpoint(module: nn.Module) -> torch.utils.hooks.RemovableHandle:
    """Annotates the output of a module to be checkpointed instead of
    recomputed"""

    def recompute_outputs(module, inputs, outputs):
        if type(outputs) is list:
            return list(poptorch.recomputationCheckpoint(y) for y in outputs)
        elif type(outputs) is tuple:
            return tuple(poptorch.recomputationCheckpoint(y) for y in outputs)
        else:
            return poptorch.recomputationCheckpoint(outputs)

    return module.register_forward_hook(recompute_outputs)


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

        self._hooks = []
        layer_ipu = _get_layer_ipu(self.config.layers_per_ipu)

        logger.info("-------------------- Device Allocation --------------------")
        logger.info("Embedding  --> IPU 0")
        self.transformer.wte = poptorch.BeginBlock(self.transformer.wte, "Token embedding", ipu_id=0)
        self.transformer.wpe = poptorch.BeginBlock(self.transformer.wpe, "Position embedding", ipu_id=1)
        outline_attribute(self.transformer.ln_f, "LayerNorm")

        for index, layer in enumerate(self.transformer.h):
            ipu = layer_ipu[index]
            if self.config.recompute_checkpoint_every_layer:
                h = recomputation_checkpoint(layer)
                self._hooks.append(h)
            self.transformer.h[index] = poptorch.BeginBlock(layer, f"Layer{index}", ipu_id=ipu)
            logger.info(f"Layer {index:<2}   --> IPU {ipu}")

        print(f"Head       --> IPU 0")
        self.lm_head = poptorch.BeginBlock(self.lm_head, "LM head", ipu_id=0)
        logger.info("-----------------------------------------------------------")
        return self

    def deparallelize(self):
        """
        Undo the changes to the model done by `parallelize`.
        You should call this before doing `save_pretrained` so that the `model.state_dict` is
        fully compatible with `transformers.GPT2LMHeadModel`.
        """
        # Remove any hooks
        for h in self._hooks:
            h.remove()
        # Remove poptorch Blocks
        for m in self.modules():
            if m != self:
                poptorch.removeBlocks(m)
        return self

    # def forward(self, input_ids, attention_mask, labels=None):
    #     transformer_outputs = self.transformer(input_ids, attention_mask=attention_mask)
    #     hidden_states = transformer_outputs[0]
    #     # hidden_states = poptorch.recomputationCheckpoint(hidden_states)
    #     lm_logits = self.lm_head(hidden_states)

    #     loss = None
    #     if labels is not None:
    #         # Shift so that tokens < n predict n
    #         labels = torch.roll(labels, -1, 1)
    #         # By default ignore_index = -100
    #         labels[:, -1] = -100
    #         loss_fct = nn.CrossEntropyLoss()
    #         loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

    #     return loss

    def forward(self, input_ids, attention_mask, labels=None):
        transformer_outputs = self.transformer(input_ids, attention_mask=attention_mask)
        hidden_states = transformer_outputs[0]
        hidden_states = poptorch.recomputationCheckpoint(hidden_states)
        lm_logits = self.lm_head(hidden_states)

        # Shift so that tokens < n predict n
        labels = torch.roll(labels, -1, 1)
        # By default ignore_index = -100
        labels[:, -1] = -100

        enable_sequence_serialized = True
        serialized_seq_len = 128
        max_len = 1024
        if not enable_sequence_serialized:
            loss = None
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            return loss
        else:
            lm_logits = lm_logits.view(-1, lm_logits.size(-1))
            labels = labels.view(-1)
            loss_fct = nn.CrossEntropyLoss(reduction="sum")
            loss = None
            loss_weights = torch.sum((labels > -1).to(torch.float), dim=-1)
            for index, i in enumerate(range(serialized_seq_len, max_len+serialized_seq_len, serialized_seq_len)):
                logit = lm_logits[i - serialized_seq_len:i, :]
                label = labels[i - serialized_seq_len:i]
                if loss is None:
                    loss = loss_fct(logit, label).to(torch.float32)
                    loss = poptorch.recomputationCheckpoint(loss)
                else:
                    tmp_loss = loss_fct(logit, label).to(torch.float32)
                    tmp_loss = poptorch.recomputationCheckpoint(tmp_loss)
                    loss += tmp_loss
            mean_loss = loss / loss_weights
            total_loss = poptorch.identity_loss(mean_loss, reduction="none")
            return total_loss


@register(GPT2ForSequenceClassification)
class PipelinedGPT2ForSequenceClassification(GPT2ForSequenceClassification, PipelineMixin):
    def parallelize(self):
        """
        Transform the model to run in an IPU pipeline.
        - Adds pipeline stages to the model
        - Adds recomputation checkpoints

        Recommended usage:
        ```
        model = PipelinedGPT2ForSequenceClassification(config).parallelize().half()
        ```
        """
        self._hooks = []
        layer_ipu = _get_layer_ipu(self.config.layers_per_ipu)

        logger.info("-------------------- Device Allocation --------------------")
        logger.info("Embedding  --> IPU 0")
        self.transformer.wte = poptorch.BeginBlock(self.transformer.wte, "Token embedding", ipu_id=0)
        self.transformer.wpe = poptorch.BeginBlock(self.transformer.wpe, "Position embedding", ipu_id=0)

        for index, layer in enumerate(self.transformer.h):
            ipu = layer_ipu[index]
            if self.config.recompute_checkpoint_every_layer:
                h = recomputation_checkpoint(layer)
                self._hooks.append(h)
            self.transformer.h[index] = poptorch.BeginBlock(layer, f"Layer{index}", ipu_id=ipu)
            logger.info(f"Layer {index:<2}   --> IPU {ipu}")

        print(f"Head       --> IPU {ipu}")
        self.score = poptorch.BeginBlock(self.score, "Score", ipu_id=ipu)
        logger.info("-----------------------------------------------------------")
        return self

    def deparallelize(self):
        """
        Undo the changes to the model done by `parallelize`.
        You should call this before doing `save_pretrained` so that the `model.state_dict` is
        fully compatible with `transformers.GPT2ForSequenceClassification`.
        """
        # Remove any hooks
        for h in self._hooks:
            h.remove()
        # Remove poptorch Blocks
        for m in self.modules():
            if m != self:
                poptorch.removeBlocks(m)
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
class PipelinedGPT2ForTokenClassification(GPT2ForTokenClassification, PipelineMixin):
    def parallelize(self):
        """
        Transform the model to run in an IPU pipeline.
        - Adds pipeline stages to the model
        - Adds recomputation checkpoints

        Recommended usage:
        ```
        model = PipelinedGPT2ForSequenceClassification(config).parallelize().half()
        ```
        """
        self._hooks = []
        layer_ipu = _get_layer_ipu(self.config.layers_per_ipu)

        logger.info("-------------------- Device Allocation --------------------")
        logger.info("Embedding  --> IPU 0")
        self.transformer.wte = poptorch.BeginBlock(self.transformer.wte, "Token embedding", ipu_id=0)
        self.transformer.wpe = poptorch.BeginBlock(self.transformer.wpe, "Position embedding", ipu_id=0)

        for index, layer in enumerate(self.transformer.h):
            ipu = layer_ipu[index]
            if self.config.recompute_checkpoint_every_layer:
                h = recomputation_checkpoint(layer)
                self._hooks.append(h)
            self.transformer.h[index] = poptorch.BeginBlock(layer, f"Layer{index}", ipu_id=ipu)
            logger.info(f"Layer {index:<2}   --> IPU {ipu}")

        print(f"Head       --> IPU {ipu}")
        self.classifier = poptorch.BeginBlock(self.classifier, "Classifier", ipu_id=ipu)
        logger.info("-----------------------------------------------------------")
        return self

    def deparallelize(self):
        """
        Undo the changes to the model done by `parallelize`.
        You should call this before doing `save_pretrained` so that the `model.state_dict` is
        fully compatible with `transformers.GPT2ForTokenClassification`.
        """
        # Remove any hooks
        for h in self._hooks:
            h.remove()
        # Remove poptorch Blocks
        for m in self.modules():
            if m != self:
                poptorch.removeBlocks(m)
        return self

    def forward(self, input_ids, attention_mask, labels=None):
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=False,
        )
