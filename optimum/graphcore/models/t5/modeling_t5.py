#  Copyright 2022 The HuggingFace Team. All rights reserved.
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

import warnings
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

import poptorch
from optimum.utils import logging
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.models.t5.modeling_t5 import __HEAD_MASK_WARNING_MSG, T5LayerNorm

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
)
from ...fx.utils import symbolic_trace_pipelined_model
from ...generation_utils import IPUGenerationMixin
from ...modeling_utils import GenerationMethodsMixin, PipelineMixin, SharedEmbedding, get_layer_ipu, register


logger = logging.get_logger(__name__)

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


@register(T5ForConditionalGeneration)
class PipelinedT5ForConditionalGeneration(
    GenerationMethodsMixin, T5ForConditionalGeneration, PipelineMixin, IPUGenerationMixin
):
    def scale_down_weights(self, factor: float = 1, restore: bool = False):
        self.lm_scale_modifier = 1 if not restore else None
        # self.lm_scale_modifier = nn.Parameter(torch.ones(self.config.d_model, dtype=torch.float16)) if not restore else None

        emb_scaling = 1 / 32.0 * factor
        att_v_scaling = 1 / 4.0 * factor
        att_o_scaling = 1 / 8.0 * factor
        ff_wi_scaling = 1 / 4.0 * factor
        ff_wo_scaling = 1 / 4.0 * factor
        ff_ln_scaling = 1 / 2.0 * factor

        if restore:
            emb_scaling = 1 / emb_scaling
            att_v_scaling = 1 / att_v_scaling
            att_o_scaling = 1 / att_o_scaling
            ff_wi_scaling = 1 / ff_wi_scaling
            ff_wo_scaling = 1 / ff_wo_scaling
            ff_ln_scaling = 1 / ff_ln_scaling

        with torch.no_grad():
            self.shared.weight *= emb_scaling
            for unit in self.encoder.block:
                unit.layer[0].SelfAttention.v.weight *= att_v_scaling
                unit.layer[0].SelfAttention.o.weight *= att_o_scaling
                unit.layer[1].DenseReluDense.wi.weight *= ff_wi_scaling
                unit.layer[1].DenseReluDense.wo.weight *= ff_wo_scaling
                unit.layer[1].layer_norm.weight *= ff_ln_scaling
            for unit in self.decoder.block:
                unit.layer[0].SelfAttention.v.weight *= att_v_scaling
                unit.layer[0].SelfAttention.o.weight *= att_o_scaling
                unit.layer[1].EncDecAttention.v.weight *= att_v_scaling
                unit.layer[1].EncDecAttention.o.weight *= att_o_scaling
                unit.layer[2].DenseReluDense.wi.weight *= ff_wi_scaling
                unit.layer[2].DenseReluDense.wo.weight *= ff_wo_scaling
                unit.layer[2].layer_norm.weight *= ff_ln_scaling

            if not restore:
                self.lm_scale_modifier /= emb_scaling

    def get_transformations(self):
        log_insertions = self.ipu_config.log_insertions
        layer_ipu = get_layer_ipu(self.ipu_config.layers_per_ipu)
        transformations = [
            AddPoptorchBlock("Embedding", 0, "shared", log_insertions=log_insertions),
            AddPoptorchBlocksInSeries(
                "Encoder", layer_ipu[: self.config.num_layers], r"encoder.block.[0-9]+", log_insertions=log_insertions
            ),
            AddPoptorchBlocksInSeries(
                "Decoder",
                layer_ipu[self.config.num_layers - 1 :],
                r"decoder.block.[0-9]+",
                log_insertions=log_insertions,
            ),
            AddPoptorchBlock("LM Head Output", 0, "lm_head", log_insertions=log_insertions),
        ]
        if self.ipu_config.recompute_checkpoint_every_layer:
            transformations += [
                RecomputationCheckpoint(
                    "encoder.block.[0-9]+", to_exclude=f"encoder.block.{self.config.num_layers - 1}"
                ),
                RecomputationCheckpoint(
                    "decoder.block.[0-9]+", to_exclude=f"decoder.block.{self.config.num_layers - 1}"
                ),
            ]

        if not isinstance(self, torch.fx.GraphModule):
            if self.ipu_config.embedding_serialization_factor > 1:
                transformations += [
                    LinearToSerializedLinear("lm_head"),
                    TieWeights("shared", "lm_head"),
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
        model = PipelinedT5ForConditionalGeneration(config).parallelize().half()
        ```
        """
        PipelineMixin.parallelize(self)
        for mod in self.modules():
            if isinstance(mod, T5LayerNorm):
                mod.forward = poptorch.autocast(enabled=True)(mod.forward)
        traced = symbolic_trace_pipelined_model(self)
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
        fully compatible with `transformers.T5ForConditionalGeneration`.
        """
        # T5ForConditionalGeneration has a deparallelize method, so make sure that the PipelineMixin one is used here.
        PipelineMixin.deparallelize(self)
        transformations = self.get_transformations()
        transformations += _OPTIMIZATION_TRANSFORMATIONS
        transformations = [t for t in transformations if isinstance(t, ReversibleTransformation)]
        composition = compose(*transformations)
        self = composition(self, reverse=True)
        return self

    # def train(self, mode: bool = True) -> "PipelinedT5ForConditionalGeneration":
    #     mod = super(T5ForConditionalGeneration, self).train(mode=mode)
    #     # TODO: enable that once generation is supported.
    #     # mod.forward = mod._forward_for_train if mode else mod._forward_for_generate
    #     mod.forward = mod._forward_for_train
    #     return mod

    # def _forward_for_train(self, input_ids, attention_mask, decoder_input_ids, labels=None):
    #     outputs = super().forward(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         decoder_input_ids=decoder_input_ids,
    #         labels=labels,
    #         use_cache=False,
    #         return_dict=False,
    #     )
    #     # Only returning the loss to make the communication between the host and the device faster.
    #     return outputs[0:1]

    # def _forward_for_generate(self, encoder_outputs, decoder_input_ids, attention_mask, labels=None):
    #     outputs = super().forward(
    #         encoder_outputs=encoder_outputs,
    #         attention_mask=attention_mask,
    #         decoder_input_ids=decoder_input_ids,
    #         return_dict=False,
    #         use_cache=False,
    #         labels=labels,
    #     )
    #     # Only returning the loss (if labels is provided) and the logits.
    #     if labels is None:
    #         return outputs[:1]
    #     return outputs[:2]

    # forward = _forward_for_train
