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

import itertools
import logging
import time

import numpy as np
from tqdm import tqdm

from .algorithms import LPFHP, SPFHP
from .dataset_templates import PackedClassificationDataset, PackedQuestionAnsweringDataset


"""
Currently enabled supported tasks:
* Single label classification with BERT
* Multi label classification with BERT
* Question-answering with BERT (SQuAD)
"""

logger = logging.getLogger("packing")


class PackedDatasetCreator:
    def __init__(
        self,
        tokenized_dataset,
        problem_type,
        num_labels: int = None,
        max_sequence_length: int = 384,
        max_sequences_per_pack: int = 6,
        training: bool = False,
        validation: bool = False,
        inference: bool = False,
        algorithm: str = "SPFHP",
        pad_to_global_batch_size: bool = False,
        global_batch_size: int = None,
        custom_label_key: str = "labels",
    ) -> None:
        # This list should contain all currently supported tasks (for BERT, currently)
        supported_problem_types = ["single_label_classification", "multi_label_classification", "question_answering"]

        self.max_seq_len = max_sequence_length
        self.max_seq_per_pack = max_sequences_per_pack
        self.num_labels = num_labels
        self.training = training
        self.validation = validation
        self.inference = inference
        self.algorithm = algorithm

        # Verify the problem type
        if problem_type in supported_problem_types:
            self.problem_type = problem_type
        else:
            logger.error(
                f"Unsupported problem type given - attempting to detect from number of labels (default 1, unless specifically passed). \
                Pass one of the supported types: {supported_problem_types}."
            )
            raise Exception

        # Verify the task
        if not training and not validation and not inference:
            logger.error(
                "One of 'training', 'validation' or 'inference' must be set to True when calling PackedDatasetCreator."
            )
            raise Exception

        # Verify num_labels if not inference
        if inference:
            logger.info("Inference mode has been set. This will override training/validation mode and ignore labels.")
        else:
            if num_labels == None:
                logger.error(
                    f'For validation (to evaluate) and training, num_labels must be passed to PackedDatasetCreator - num_labels got "None"!'
                )
                raise Exception

        # Get the unpacked default data columns
        self.unpacked_input_ids = tokenized_dataset["input_ids"]
        self.unpacked_attention_mask = tokenized_dataset["attention_mask"]
        self.unpacked_token_type_ids = tokenized_dataset["token_type_ids"]

        # Get the strategy to pack the dataset using the algorithm
        self.strategy = self.get_strategy()
        total_num_packs = np.sum(self.strategy[1])

        # Provide an option to pad the dataset to the given global batch size to avoid skipping samples
        if pad_to_global_batch_size and global_batch_size:
            if total_num_packs % global_batch_size != 0:
                difference_to_batch_size = global_batch_size - (total_num_packs % global_batch_size)
                total_num_packs += difference_to_batch_size

        self.total_num_packs = total_num_packs

        # Prepare the manually padded constant sized data
        self.shift_cls_tokens = True
        self.adjust_offset_positions = False

        self.packed_input_ids = np.zeros((self.total_num_packs, self.max_seq_len), dtype=int)
        self.packed_attention_mask = np.zeros((self.total_num_packs, self.max_seq_len), dtype=int)
        self.packed_token_type_ids = np.zeros((self.total_num_packs, self.max_seq_len), dtype=int)
        self.packed_position_ids = np.zeros((self.total_num_packs, self.max_seq_len), dtype=int)

        # Task-specific dataset categories and dataset class definitions
        if problem_type == "single_label_classification":
            self.dataset_class = PackedClassificationDataset

            if not self.inference:
                self.unpacked_labels = tokenized_dataset[custom_label_key]
                self.packed_labels = -100 * np.ones((self.total_num_packs, self.max_seq_per_pack), dtype=int)
            else:
                self.packed_labels = None

        elif problem_type == "multi_label_classification":
            self.dataset_class = PackedClassificationDataset

            if not self.inference:
                self.unpacked_labels = tokenized_dataset[custom_label_key]
                self.packed_labels = -100 * np.ones(
                    (self.total_num_packs, self.max_seq_per_pack, self.num_labels), dtype=int
                )
            else:
                self.packed_labels = None

        elif problem_type == "question_answering":
            if self.training:
                self.unpacked_start_positions = tokenized_dataset["start_positions"]
                self.unpacked_end_positions = tokenized_dataset["end_positions"]
                self.packed_start_positions = -100 * np.ones((self.total_num_packs, self.max_seq_per_pack), dtype=int)
                self.packed_end_positions = -100 * np.ones((self.total_num_packs, self.max_seq_per_pack), dtype=int)
            else:
                self.packed_start_positions = None
                self.packed_end_positions = None

            if self.validation or self.inference:
                self.unpacked_example_ids = tokenized_dataset["example_id"]
                self.unpacked_offset_mapping = tokenized_dataset["offset_mapping"]
                self.packed_example_ids = np.zeros((self.total_num_packs, self.max_seq_per_pack), dtype="<U32")
                self.packed_offset_mapping = -np.ones((self.total_num_packs, self.max_seq_len, 2), dtype=int)
            else:
                self.packed_example_ids = None
                self.packed_offset_mapping = None

            self.adjust_offset_positions = True
            self.shift_cls_tokens = False

            self.dataset_class = PackedQuestionAnsweringDataset

    # This function generates the histogram to be used by the histogram-based packing algorithm
    def generate_histogram(self):
        dataset_seq_lens = np.array([len(seq) for seq in self.unpacked_input_ids])
        histogram = np.zeros(self.max_seq_len, dtype=np.int64)
        seq_lens, counts = np.unique(dataset_seq_lens, return_counts=True)
        histogram[seq_lens - 1] = counts
        return histogram

    # This function runs the algorithm on the histogram to obtain the packing strategy
    def get_strategy(self):
        self.histogram = self.generate_histogram()

        if self.algorithm == "SPFHP":
            strategy = SPFHP(self.histogram, self.max_seq_len, self.max_seq_per_pack)
        elif self.algorithm == "LPFHP":
            strategy = LPFHP(self.histogram, self.max_seq_len, self.max_seq_per_pack)
        else:
            logger.error("Algorithm type unsupported. Pass one of: LPFHP, SPFHP")
            raise Exception

        return strategy

    # This function ßåcreates the strategy
    def create(self):
        strategy_set = self.strategy[0]
        strategy_repeat_count = self.strategy[1]
        skip_cls = int(self.shift_cls_tokens)

        # Sort the sequences by length
        dataset_seq_lens = np.array([len(seq) for seq in self.unpacked_input_ids])
        len_sorted_seq_idxs = np.argsort(dataset_seq_lens)
        len_sorted_seq_lens = dataset_seq_lens[len_sorted_seq_idxs]
        sorted_seqs = np.stack((len_sorted_seq_lens, len_sorted_seq_idxs))

        # Pack the data using the developed strategies
        pack_index = 0

        st = time.time()
        for i in range(len(strategy_repeat_count)):
            strategy = strategy_set[i]

            # This is the offset we apply to the start positions to account for the positional change of the logits when unmasking the pack to extract a set of logits for each sequence in the pack
            if self.adjust_offset_positions:
                positions_offset = [sum(strategy[:n]) for n in range(len(strategy))]

            for _ in range(strategy_repeat_count[i]):
                ref_inds = []
                for x in strategy:
                    ref_ind = np.argwhere(sorted_seqs[0] == x)[-1]
                    sorted_seqs[0, ref_ind] = -1
                    ref_inds.append(ref_ind)

                inds = sorted_seqs[1, ref_inds].ravel()

                # Exclude the CLS tokens to put them at the end later
                input_id_pack = list(itertools.chain(*[self.unpacked_input_ids[x][skip_cls:] for x in inds]))
                attention_mask_pack = list(
                    itertools.chain(
                        *[
                            itertools.repeat(n + 1, len(self.unpacked_attention_mask[v]) - skip_cls)
                            for n, v in enumerate(inds)
                        ]
                    )
                )
                token_type_ids_pack = list(
                    itertools.chain(*[self.unpacked_token_type_ids[x][skip_cls:] for x in inds])
                )
                position_ids_pack = list(
                    itertools.chain(
                        *[range(skip_cls, len(self.unpacked_attention_mask[v])) for n, v in enumerate(inds)]
                    )
                )

                # Create the equivalent tokenised packed dataset - we operate with python arrays due to inhomogenous dataset size
                self.packed_input_ids[pack_index, : len(input_id_pack)] = input_id_pack
                self.packed_attention_mask[pack_index, : len(attention_mask_pack)] = attention_mask_pack
                self.packed_token_type_ids[pack_index, : len(token_type_ids_pack)] = token_type_ids_pack
                self.packed_position_ids[pack_index, : len(position_ids_pack)] = position_ids_pack

                if self.training or self.validation:
                    if self.problem_type == "single_label_classification":
                        labels_pack = [self.unpacked_labels[x] for x in inds]
                        self.packed_labels[pack_index, : len(labels_pack)] = labels_pack

                    if self.problem_type == "multi_label_classification":
                        labels_pack = np.stack([self.unpacked_labels[x] for x in inds])
                        self.packed_labels[pack_index, : labels_pack.shape[0], :] = labels_pack

                if self.problem_type == "question_answering":
                    if self.training:
                        start_positions_pack = [
                            max(self.unpacked_start_positions[v] + positions_offset[n], 0) for n, v in enumerate(inds)
                        ]
                        end_positions_pack = [
                            max(self.unpacked_end_positions[v] + positions_offset[n], 0) for n, v in enumerate(inds)
                        ]
                        self.packed_start_positions[pack_index, : len(start_positions_pack)] = start_positions_pack
                        self.packed_end_positions[pack_index, : len(end_positions_pack)] = end_positions_pack

                    if self.validation or self.inference:
                        example_ids_pack = [self.unpacked_example_ids[x] for x in inds]
                        offset_mapping_pack = list(itertools.chain(*[self.unpacked_offset_mapping[x] for x in inds]))

                        self.packed_example_ids[pack_index, : len(example_ids_pack)] = example_ids_pack
                        self.packed_offset_mapping[pack_index, : len(offset_mapping_pack)] = offset_mapping_pack

                # Now add the CLS tokens and their masks at the end of the pack if classification task
                if skip_cls:
                    self.packed_input_ids[pack_index, -self.max_seq_per_pack :] = [
                        self.unpacked_input_ids[0][0] for _ in range(self.max_seq_per_pack)
                    ]
                    self.packed_attention_mask[pack_index, -self.max_seq_per_pack :] = list(
                        range(1, self.max_seq_per_pack + 1)
                    )

                pack_index += 1

        print(f"Packed dataset creation time: {round(time.time()-st, 4)}s")

        if self.problem_type == "single_label_classification" or self.problem_type == "multi_label_classification":
            return PackedClassificationDataset(
                input_ids=self.packed_input_ids,
                attention_mask=self.packed_attention_mask,
                token_type_ids=self.packed_token_type_ids,
                position_ids=self.packed_position_ids,
                labels=self.packed_labels,
            )

        if self.problem_type == "question_answering":
            return PackedQuestionAnsweringDataset(
                input_ids=self.packed_input_ids,
                attention_mask=self.packed_attention_mask,
                token_type_ids=self.packed_token_type_ids,
                position_ids=self.packed_position_ids,
                start_positions=self.packed_start_positions,
                end_positions=self.packed_end_positions,
                offset_mapping=self.packed_offset_mapping,
                example_ids=self.packed_example_ids,
            )
