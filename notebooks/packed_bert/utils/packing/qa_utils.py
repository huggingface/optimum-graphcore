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

import collections

import numpy as np
from datasets import Dataset
from tqdm import tqdm

from transformers import AutoTokenizer


def preprocess_packed_qa(
    dataset,
    tokenizer,
    question_key: str = "question",
    context_key: str = "context",
    answer_key: str = "answer",
    sequence_length: int = 384,
    padding: bool = True,
    train: bool = True,
):
    # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.

    pad_on_right = tokenizer.padding_side == "right"

    tokenized_dataset = tokenizer(
        dataset[question_key if pad_on_right else context_key],
        dataset[context_key if pad_on_right else question_key],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=sequence_length,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=padding,
    )

    sample_mapping = tokenized_dataset.pop("overflow_to_sample_mapping")

    dataset_answers = dataset[answer_key]
    start_positions = []
    end_positions = []

    if train:
        offset_mapping = tokenized_dataset.pop("offset_mapping")

        for i, offsets in enumerate(tqdm(offset_mapping)):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_dataset["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_dataset.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = dataset_answers[sample_index]

            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                start_positions.append(cls_index)
                end_positions.append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])
                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1
                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1
                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    start_positions.append(cls_index)
                    end_positions.append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    start_positions.append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    end_positions.append(token_end_index + 1)

        tokenized_dataset["start_positions"] = start_positions
        tokenized_dataset["end_positions"] = end_positions

        return Dataset.from_dict(tokenized_dataset)

    else:
        # We keep the example_id that gave us this feature and we will store the offset mappings.
        tokenized_dataset["example_id"] = []
        dataset_ids = dataset["id"]

        for i in range(len(tokenized_dataset["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_dataset.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_dataset["example_id"].append(dataset_ids[sample_index])

            # Set to 0 the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_dataset["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else tuple((0, 0)))
                for k, o in enumerate(tokenized_dataset["offset_mapping"][i])
            ]

        return Dataset.from_dict(tokenized_dataset)


def postprocess_packed_qa_predictions(
    raw_val_dataset,
    tokenized_val_dataset,
    raw_predictions,
    n_best_size=20,
    max_answer_length=30,
    squad_v2=False,
    cls_token_id=101,
):
    all_start_logits, all_end_logits = raw_predictions

    # The dataloader drop_last affects the dataset size due to the global batch size, so the number of predictions may be slightly less than the total amount of validation samples available:
    dataloader_cap = all_start_logits.shape[0]

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(raw_val_dataset["id"])}

    features_per_example = collections.defaultdict(list)

    for i, feature in enumerate(tokenized_val_dataset):
        for j, example_id in enumerate(feature["example_ids"]):
            if example_id != "":
                features_per_example[example_id_to_index[example_id]].append([i, j])

    # The dictionaries we have to fill.
    predictions = collections.OrderedDict()

    # Logging.
    print(
        f"Post-processing {len(raw_val_dataset)} example predictions split into {len(tokenized_val_dataset)} features."
    )

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(raw_val_dataset)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_score = None  # Only used if squad_v2 is True.
        valid_answers = []

        context = example["context"]
        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # Separate the feature index and the pack index (i.e. the index of the feature in the pack)
            pack_index, sequence_in_pack_index = feature_index

            # We want to ignore any indices of packs which were ignored by the validation loop due to the dataloader dropping uneven batches.
            if pack_index >= dataloader_cap:
                continue

            # We grab the predictions of the model for this feature to map character-level spans from the offset.
            start_logits = all_start_logits[pack_index, sequence_in_pack_index]
            end_logits = all_end_logits[pack_index, sequence_in_pack_index]

            # Update minimum null prediction.
            offset_mapping = tokenized_val_dataset[pack_index]["offset_mapping"]

            # If squad_v2 dataset is used, we need to account for null predictions; we determine the minimum null score using input_ids to find the cls_index of the current sequence in the pack.
            if squad_v2:
                input_ids = tokenized_val_dataset[pack_index]["input_ids"]

                cls_indices = [k for k, v in enumerate(input_ids) if v == int(cls_token_id)]
                cls_index = cls_indices[sequence_in_pack_index]

                # Since we know the relevant CLS index for this sequence in the pack, the null score can be evaluated
                feature_null_score = start_logits[cls_index] + end_logits[cls_index]

                if min_null_score is None or min_null_score < feature_null_score:
                    min_null_score = feature_null_score

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                        or offset_mapping[start_index] == []
                        or offset_mapping[end_index] == []
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char:end_char],
                        }
                    )

        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            best_answer = {"text": "", "score": 0.0}

        # Let's pick our final answer: the best one or the null answer (only for squad_v2)
        if not squad_v2:
            predictions[example["id"]] = best_answer["text"]
        else:
            answer = best_answer["text"] if best_answer["score"] > min_null_score else ""
            predictions[example["id"]] = answer

    return predictions
