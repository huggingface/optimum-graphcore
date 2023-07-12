# Copyright 2022 Meta Platforms, Inc. and The HuggingFace Inc. team. All rights reserved.
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

from transformers.models.convnext.modeling_convnext import ConvNextLayer


class OptimizedConvNextLayer(ConvNextLayer):
    def forward(self, hidden_states):
        """
        Merge the 2nd and 3rd dimensions of the tensor before pwconv, and restore the shape afterwards.
        This is because currently, nn.Linear() does not work efficiently on 4-dimensional inputs.
        """
        input = hidden_states
        x = self.dwconv(hidden_states)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.layernorm(x)
        N, H, W, C = x.shape
        # Reshape for running efficiently on IPUs
        x = x.view(N, -1, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        # Restore the shape
        x = x.view(N, H, W, C)
        if self.layer_scale_parameter is not None:
            x = self.layer_scale_parameter * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x
