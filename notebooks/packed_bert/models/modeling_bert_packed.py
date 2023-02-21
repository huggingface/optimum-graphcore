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

import torch
import torch.nn as nn

import poptorch
from optimum.graphcore.models.bert.modeling_bert import BertPipelineMixin
from transformers import BertForQuestionAnswering, BertForSequenceClassification
from transformers.modeling_outputs import QuestionAnsweringModelOutput


class PackedBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_seq_per_pack = config.max_sequences_per_pack
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        """
        We "pool" the model by simply taking the hidden states corresponding
        to the last max_sequences_per_pack tokens. Note that the [CLS] tokens
        are always located at the end of the pack. When the actual number of
        sequences is lower than max_sequences_per_pack, we still slice out
        the last max_sequences_per_pack tokens, but we will not use all of
        them during loss calculation.
        """
        sh = hidden_states.shape
        last_tokens_tensors = hidden_states[:, -self.max_seq_per_pack :]
        last_reshape = last_tokens_tensors.reshape(sh[0] * self.max_seq_per_pack, sh[2])
        # output size: [bs x max_sequences_per_pack, hidden_size]
        output = self.dense(last_reshape)
        output = self.activation(output)

        return output


class PackedBertOutputsForMultiLabel(nn.Module):
    """
    This class handles the custom model output phase for multi-label sequence classification.
    """

    def __init__(self, config):
        super().__init__()
        self.max_seq_per_pack = config.max_sequences_per_pack
        self.multi_loss = torch.nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self,
        outputs: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        batch_dim: int,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:
        max_labels = torch.max(attention_mask[:, : -self.max_seq_per_pack], dim=-1).values.unsqueeze(1)

        # Create a mask corresponding to actual number of seqs in pack, to mask padding
        label_mask = torch.arange(0, self.max_seq_per_pack).unsqueeze(0).repeat(batch_dim, 1)
        label_mask = torch.where(
            label_mask < max_labels,
            torch.ones(batch_dim, self.max_seq_per_pack),
            torch.zeros(batch_dim, self.max_seq_per_pack),
        )
        label_mask = label_mask.view(-1).unsqueeze(1)

        # Adjust logits to rule out padding
        logits = label_mask * outputs.logits

        loss = None
        if labels is not None:
            # Flatten and adjust labels to rule out padding
            labels = labels.view(-1, *(labels.size()[2:])).to(torch.float32)
            labels = label_mask * labels

            # Adjust the loss to rule out the padding and CLS logits
            loss = self.multi_loss(logits, labels)
            loss *= label_mask

            # Take mean over each multi-class pred
            loss = torch.sum(loss) / (torch.sum(max_labels) * labels.shape[-1])
            loss = poptorch.identity_loss(loss, reduction="none")

            logits = logits.reshape([batch_dim, self.max_seq_per_pack, logits.shape[-1]])

            return (loss, logits)
        else:
            return logits


class PipelinedPackedBertForSequenceClassification(BertForSequenceClassification, BertPipelineMixin):
    """
    This class supports doing single-label/multi-label sequence-classification tasks with custom outputs.
    The problem_type must be passed to differentiate the two methods - multi_label_classification or single_label_classification. Multi-label requires a custom loss implementation to mask labels and logits, unlike single-label.

    In both cases:
        * The logits need to be reshaped at output to revert them from the 'unpacked' batch dimension to a batch dimension equivalent to that of the labels passed to the model in order for Optimum's trainer class to perform evaluation.

        * The attention mask is reshaped from the 'packed' attention mask to an equivalent binary 3D "extended" attention mask for BERT to recognise the sequences within a single packed input as unrelated sequences.
    """

    def __init__(self, config):
        super().__init__(config)
        self.max_seq_per_pack = config.max_sequences_per_pack
        self.problem_type = config.problem_type
        self.num_labels = config.num_labels

        self.bert.pooler = PackedBertPooler(config)
        self.multi_label_outputs = PackedBertOutputsForMultiLabel(config)

    def parallelize(self):
        super().parallelize()
        last_ipu = self.ipu_config.ipus_per_replica - 1
        self.classifier = poptorch.BeginBlock(self.classifier, "Classifier Output", ipu_id=last_ipu)
        return self

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple[torch.Tensor]:
        bs = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        attention_mask_3d = attention_mask[:, None, :].repeat(1, seq_len, 1)
        attention_mask_3d = (attention_mask_3d == attention_mask_3d.transpose(1, 2)) * (attention_mask_3d != 0)

        # Manual masking of logits and loss only needed for multi-label, single-label loss allows ignore_index
        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask_3d,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            labels=labels if labels is not None and self.problem_type == "single_label_classification" else None,
        )

        if self.problem_type == "single_label_classification":
            if labels is not None:
                logits = output.logits.reshape([-1, self.max_seq_per_pack, self.num_labels])
                output.logits = logits

        else:
            output = self.multi_label_outputs(
                outputs=output, attention_mask=attention_mask, batch_dim=bs, labels=labels
            )

        return output


class PackedBertOutputsForQA(nn.Module):
    """
    This class handles the custom output phase for a question-answering task.
    """

    def __init__(self, config):
        super().__init__()
        # Use the default QA model output formatting class to return outputs in the same form as the base model.
        self.output = QuestionAnsweringModelOutput
        self.max_sequences_per_pack = config.max_sequences_per_pack

    def forward(
        self,
        final_layer_output: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
        # Create unpacking mask to separate packed logits out into sequence-specific logits only
        unpacking_mask = attention_mask[:, None, :].repeat(1, self.max_sequences_per_pack, 1)
        pack_seq_ids = torch.arange(1, self.max_sequences_per_pack + 1).view(self.max_sequences_per_pack, 1)

        unpacking_mask = unpacking_mask == pack_seq_ids

        # Expand start logits using mask to isolate logits for each internal sequence in the pack
        unpacked_start_logits = final_layer_output.start_logits[:, None, :] * unpacking_mask
        unpacked_end_logits = final_layer_output.end_logits[:, None, :] * unpacking_mask

        # Calculate loss on logits/labels with initial [bs, mspp, ...] dims collapsed into one [bs*mspp, ...]
        total_loss = None
        if start_positions is not None and end_positions is not None:
            start_positions = start_positions.view(-1)
            end_positions = end_positions.view(-1)

            unpacked_start_logits = unpacked_start_logits.contiguous()
            unpacked_end_logits = unpacked_end_logits.contiguous()

            unpacked_start_logits = unpacked_start_logits.view(-1, unpacked_start_logits.shape[-1])
            unpacked_end_logits = unpacked_end_logits.view(-1, unpacked_end_logits.shape[-1])

            loss_fct = nn.CrossEntropyLoss()
            start_loss = loss_fct(unpacked_start_logits, start_positions)
            end_loss = loss_fct(unpacked_end_logits, end_positions)

            total_loss = (start_loss + end_loss) / 2

        return self.output(
            loss=total_loss,
            start_logits=unpacked_start_logits,
            end_logits=unpacked_end_logits,
            hidden_states=final_layer_output.hidden_states,
            attentions=final_layer_output.attentions,
        )


class PipelinedPackedBertForQuestionAnswering(BertForQuestionAnswering, BertPipelineMixin):
    """
    This class extends BertForQuestionAnswering with some differences required for packing. The 'packed' attention mask must be extended to a 3D binary "extended" attention mask for BERT to recognise the sequences within a single packed input as unrelated sequences. The output is extended to enable masking for padded labels, and then 'unpacking' the packed hidden state output before performing the loss calculation.
    """

    def __init__(self, config):
        super().__init__(config)
        self.max_seq_per_pack = self.config.max_sequences_per_pack
        self.packed_outputs = PackedBertOutputsForQA(config)

    def parallelize(self):
        super().parallelize()
        last_ipu = self.ipu_config.ipus_per_replica - 1
        self.qa_outputs = poptorch.BeginBlock(self.qa_outputs, "QA Outputs", ipu_id=last_ipu)
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
    ) -> Tuple[torch.Tensor]:
        # Create 3D attention mask for sequence specific attention in pack
        seq_len = input_ids.shape[1]
        packed_attention_mask = attention_mask[:, None, :].repeat(1, seq_len, 1)
        packed_attention_mask = (packed_attention_mask == packed_attention_mask.transpose(1, 2)) * (
            packed_attention_mask != 0
        )

        # Run forwards pass through model without labels
        final_layer_output = super().forward(
            input_ids, attention_mask=packed_attention_mask, token_type_ids=token_type_ids, position_ids=position_ids
        )

        # Custom PackedBert for SQuAD output, redirect from before loss function in transformers model class.
        output = self.packed_outputs(
            final_layer_output,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions,
        )

        if start_positions is not None and end_positions is not None:
            return poptorch.identity_loss(output.loss, reduction="mean"), output.start_logits, output.end_logits
        else:
            return output.start_logits, output.end_logits
