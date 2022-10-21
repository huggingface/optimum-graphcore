# coding=utf-8
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
"""LXMERT model."""

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

from transformers.models.lxmert.modeling_lxmert import LxmertForQuestionAnswering, LxmertForQuestionAnsweringOutput

from ....fx.optimization import MergeLinears, compose
from ....utils import logging
from ...fx import (
    DEFAULT_TRANSFORMATION_MANAGER,
    AddPoptorchBlock,
    AddPoptorchBlocksInSeries,
    RecomputationCheckpoint,
    symbolic_trace_pipelined_model,
)
from ...modeling_utils import PipelineMixin, get_layer_ipu, register


TRANSFORMATION_MANAGER = DEFAULT_TRANSFORMATION_MANAGER.without(MergeLinears())

logger = logging.get_logger(__name__)


@register(LxmertForQuestionAnswering)
class PipelinedLxmertForQuestionAnswering(LxmertForQuestionAnswering, PipelineMixin):
    def get_transformations(self):
        log_insertions = self.ipu_config.log_insertions
        layer_ipu = get_layer_ipu(self.ipu_config.layers_per_ipu)
        # TODO: remove this line after testing.
        layer_ipu = get_layer_ipu([0, 7, 7, 5])
        language_layers_ipus = layer_ipu[: self.config.l_layers]
        visual_layers_ipus = layer_ipu[self.config.l_layers : self.config.l_layers + self.config.r_layers]
        cross_modality_layers_ipus = layer_ipu[self.config.l_layers + self.config.r_layers :]

        transformations = [
            AddPoptorchBlock("Embedding", 0, "lxmert.embeddings", log_insertions=log_insertions),
            AddPoptorchBlock("Image Embedding", 0, "lxmert.encoder.visn_fc", log_insertions=log_insertions),
            AddPoptorchBlocksInSeries(
                "Language Layer", language_layers_ipus, r"lxmert.encoder.layer.[0-9]+", log_insertions=log_insertions
            ),
            AddPoptorchBlocksInSeries(
                "Visual Layer", visual_layers_ipus, r"lxmert.encoder.r_layers.[0-9]+", log_insertions=log_insertions
            ),
            AddPoptorchBlocksInSeries(
                "Cross Modality Layer",
                cross_modality_layers_ipus,
                r"lxmert.encoder.x_layers.[0-9]+",
                log_insertions=log_insertions,
            ),
            AddPoptorchBlock("Pooler Output", layer_ipu[-1], "lxmert.pooler", log_insertions=log_insertions),
            AddPoptorchBlock("Head Output", layer_ipu[-1], "lxmert.answer_head", log_insertions=log_insertions),
        ]
        if self.ipu_config.recompute_checkpoint_every_layer:
            transformations += [
                RecomputationCheckpoint(
                    "lxmert.encoder.layer.[0-9]+", to_exclude=f"lxmert.encoder.layer.{self.config.l_layers - 1}"
                ),
                RecomputationCheckpoint(
                    "lxmert.encoder.r_layers.[0-9]+", to_exclude=f"lxmert.encoder.r_layers.{self.config.r_layers - 1}"
                ),
                RecomputationCheckpoint(
                    "lxmert.encoder.x_layers.[0-9]+", to_exclude=f"lxmert.encoder.x_layers.{self.config.x_layers - 1}"
                ),
            ]
        return transformations

    def parallelize(self):
        super().parallelize()
        traced = symbolic_trace_pipelined_model(self)
        transformations = self.get_transformations()
        transformations += TRANSFORMATION_MANAGER.get_reversible_transformations(self.ipu_config.optimization_level)
        composition = compose(*transformations)
        non_reversible_composition = TRANSFORMATION_MANAGER.compose_non_reversible_transformations(
            self.ipu_config.optimization_level
        )
        traced = composition(traced)
        traced = non_reversible_composition(traced)
        return traced

    def deparallelize(self):
        super().deparallelize()
        transformations = self.get_transformations()
        transformations += TRANSFORMATION_MANAGER.get_reversible_transformations(self.ipu_config.optimization_level)
        composition = compose(*transformations)
        self = composition(self, reverse=True)
        return self

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        visual_feats: Optional[torch.FloatTensor] = None,
        visual_pos: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        visual_attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[LxmertForQuestionAnsweringOutput, Tuple[torch.FloatTensor]]:
        r"""
        labels: (`Torch.Tensor` of shape `(batch_size)`, *optional*):
            A one-hot representation of the correct answer
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        lxmert_output = self.lxmert(
            input_ids=input_ids,
            visual_feats=visual_feats,
            visual_pos=visual_pos,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            visual_attention_mask=visual_attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        pooled_output = lxmert_output[2]
        answer_score = self.answer_head(pooled_output)
        loss = None
        if labels is not None:
            if labels.dim() == 1:
                loss = F.cross_entropy(answer_score.view(-1, self.num_qa_labels), labels.view(-1))
            # Soft labels for datasets such as VQA v2
            else:
                loss = F.binary_cross_entropy_with_logits(
                    answer_score.view(-1, self.num_qa_labels), labels.view(-1, self.num_qa_labels)
                )

        if not return_dict:
            output = (answer_score,) + lxmert_output[3:]
            return (loss,) + output if loss is not None else output

        return LxmertForQuestionAnsweringOutput(
            loss=loss,
            question_answering_score=answer_score,
            language_hidden_states=lxmert_output.language_hidden_states,
            vision_hidden_states=lxmert_output.vision_hidden_states,
            language_attentions=lxmert_output.language_attentions,
            vision_attentions=lxmert_output.vision_attentions,
            cross_encoder_attentions=lxmert_output.cross_encoder_attentions,
        )
