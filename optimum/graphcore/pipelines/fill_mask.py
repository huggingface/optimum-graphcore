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

from typing import Dict

from transformers import FillMaskPipeline
from transformers.pipelines.base import GenericTensor, PipelineException


class IPUFillMaskPipeline(FillMaskPipeline):
    def _sanitize_parameters(self, top_k=None, targets=None, **tokenizer_kwargs):
        preprocess_params = tokenizer_kwargs

        postprocess_params = {}

        if targets is not None:
            target_ids = self.get_target_ids(targets, top_k)
            postprocess_params["target_ids"] = target_ids

        if top_k is not None:
            postprocess_params["top_k"] = top_k

        if self.tokenizer.mask_token_id is None:
            raise PipelineException(
                "fill-mask", self.model.base_model_prefix, "The tokenizer does not define a `mask_token`."
            )
        return preprocess_params, {}, postprocess_params

    def preprocess(self, inputs, return_tensors=None, **tokenizer_kwargs) -> Dict[str, GenericTensor]:
        if return_tensors is None:
            return_tensors = self.framework
        model_inputs = self.tokenizer(inputs, return_tensors=return_tensors, **tokenizer_kwargs)
        self.ensure_exactly_one_mask_token(model_inputs)
        return model_inputs
