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

import poptorch
from optimum.utils import logging
from transformers import HubertForSequenceClassification

from ...modeling_utils import PipelineMixin, get_layer_ipu, recomputation_checkpoint, register


logger = logging.get_logger(__name__)


@register(HubertForSequenceClassification)
class PipelinedHubertForSequenceClassification(HubertForSequenceClassification, PipelineMixin):
    def parallelize(self):
        super().parallelize()
        self.hubert.feature_extractor = poptorch.BeginBlock(self.hubert.feature_extractor, ipu_id=0)
        self.hubert.feature_projection = poptorch.BeginBlock(self.hubert.feature_projection, ipu_id=0)
        self.hubert.encoder = poptorch.BeginBlock(self.hubert.encoder, ipu_id=0)

        layer_ipu = get_layer_ipu(self.ipu_config.layers_per_ipu)
        for index, layer in enumerate(self.hubert.encoder.layers):
            # Put checkpoints on every encoder layer
            h = recomputation_checkpoint(layer)
            self._hooks.append(h)
            ipu = layer_ipu[index]
            self.hubert.encoder.layers[index] = poptorch.BeginBlock(layer, f"Encoder{index}", ipu_id=ipu)

        last_ipu = self.ipu_config.ipus_per_replica - 1
        self.projector = poptorch.BeginBlock(self.projector, ipu_id=last_ipu)
        self.classifier = poptorch.BeginBlock(self.classifier, ipu_id=last_ipu)
        return self

    @poptorch.autocast(enabled=True)
    def forward(
        self,
        input_values,
        labels=None,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        return super().forward(
            input_values,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
            labels=labels,
        )
