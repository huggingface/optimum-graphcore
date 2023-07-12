# Copyright 2021 The HuggingFace Team. All rights reserved.
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

from typing import List, Optional, Tuple

from transformers import TokenClassificationPipeline
from transformers.pipelines.token_classification import AggregationStrategy


class IPUTokenClassificationPipeline(TokenClassificationPipeline):
    def _sanitize_parameters(
        self,
        ignore_labels=None,
        grouped_entities: Optional[bool] = None,
        ignore_subwords: Optional[bool] = None,
        aggregation_strategy: Optional[AggregationStrategy] = None,
        offset_mapping: Optional[List[Tuple[int, int]]] = None,
        stride: Optional[int] = None,
        **tokenizer_kwargs,
    ):
        preprocess_params, forward_params, postprocess_params = super()._sanitize_parameters(
            ignore_labels=ignore_labels,
            grouped_entities=grouped_entities,
            ignore_subwords=ignore_subwords,
            aggregation_strategy=aggregation_strategy,
            offset_mapping=offset_mapping,
            stride=stride,
        )
        if tokenizer_kwargs:
            preprocess_params["tokenizer_params"] = {
                **preprocess_params.get("tokenizer_params", {}),
                **tokenizer_kwargs,
            }
        return preprocess_params, forward_params, postprocess_params
