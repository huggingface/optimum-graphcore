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

from torch.utils.data import Dataset


class PackedClassificationDataset(Dataset):
    def __init__(self, input_ids, attention_mask, token_type_ids, position_ids, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.position_ids = position_ids
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        input_ids = self.input_ids[index]
        attention_masks = self.attention_mask[index]
        token_type_ids = self.token_type_ids[index]
        position_ids = self.position_ids[index]
        labels = self.labels[index] if self.labels is not None else None

        sample = {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
        }

        if self.labels is not None:
            sample["labels"] = labels

        return sample


class PackedQuestionAnsweringDataset(Dataset):
    def __init__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        start_positions,
        end_positions,
        offset_mapping,
        example_ids,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.position_ids = position_ids
        self.start_positions = start_positions
        self.end_positions = end_positions
        self.offset_mapping = offset_mapping
        self.example_ids = example_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        input_ids = self.input_ids[index]
        attention_masks = self.attention_mask[index]
        token_type_ids = self.token_type_ids[index]
        position_ids = self.position_ids[index]

        start_positions = self.start_positions[index] if self.start_positions is not None else None
        end_positions = self.end_positions[index] if self.end_positions is not None else None

        offset_mapping = self.offset_mapping[index] if self.offset_mapping is not None else None
        example_ids = self.example_ids[index] if self.example_ids is not None else None

        sample = {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
        }

        if self.start_positions is not None and self.end_positions is not None:
            sample["start_positions"] = start_positions
            sample["end_positions"] = end_positions

        if self.offset_mapping is not None and self.example_ids is not None:
            sample["offset_mapping"] = offset_mapping
            sample["example_ids"] = example_ids

        return sample
