# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker, cosine_distance

from ....modeling_utils import PipelineMixin


class IPUStableDiffusionSafetyChecker(StableDiffusionSafetyChecker, PipelineMixin):
    def forward(self, clip_input: torch.FloatTensor, images: torch.FloatTensor):
        # Adapted from StableDiffusionSafetyChecker.forward_onnx
        dtype = next(self.vision_model.parameters()).dtype
        clip_input = clip_input.to(dtype)

        pooled_output = self.vision_model(clip_input)[1]  # pooled_output
        image_embeds = self.visual_projection(pooled_output)

        special_cos_dist = cosine_distance(image_embeds, self.special_care_embeds)
        cos_dist = cosine_distance(image_embeds, self.concept_embeds)

        # increase this value to create a stronger `nsfw` filter
        # at the cost of increasing the possibility of filtering benign images
        adjustment = 0.0

        special_scores = special_cos_dist - self.special_care_embeds_weights + adjustment
        # special_scores = special_scores.round(decimals=3)
        special_care = torch.any(special_scores > 0, dim=1)
        special_adjustment = special_care * 0.01
        special_adjustment = special_adjustment.unsqueeze(1).expand(-1, cos_dist.shape[1])

        concept_scores = (cos_dist - self.concept_embeds_weights) + special_adjustment
        # concept_scores = concept_scores.round(decimals=3)
        has_nsfw_concepts = torch.any(concept_scores > 0, dim=1)

        # IPU mod
        # images[has_nsfw_concepts] = 0.0  # black image
        images = images * ~has_nsfw_concepts
        images = images.to(torch.float32)

        return images, has_nsfw_concepts
