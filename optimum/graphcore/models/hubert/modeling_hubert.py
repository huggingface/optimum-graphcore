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
from transformers.models.hubert.modeling_hubert import HubertEncoder, HubertEncoderStableLayerNorm

from ...modeling_utils import PipelineMixin, get_layer_ipu, recomputation_checkpoint, register
from .ipu_layer_drop import IPUHubertEncoder, IPUHubertEncoderStableLayerNorm


logger = logging.get_logger(__name__)


@register(HubertForSequenceClassification)
class PipelinedHubertForSequenceClassification(HubertForSequenceClassification, PipelineMixin):
    def change_hubert_encoder_class(self, restore: bool):
        """Changes the encoder class to update its forward pass so that it uses our custom version.

        Args:
            restore: whether to restore the encoder to its original version or not.
        """
        if self.config.do_stable_layer_norm:
            new_cls = HubertEncoderStableLayerNorm if restore else IPUHubertEncoderStableLayerNorm
        else:
            new_cls = HubertEncoder if restore else IPUHubertEncoder
        self.hubert.encoder.__class__ = new_cls

    def parallelize(self):
        super().parallelize()

        self.change_hubert_encoder_class(False)

        self.hubert.feature_extractor = poptorch.BeginBlock(self.hubert.feature_extractor, ipu_id=0)
        self.hubert.feature_projection = poptorch.BeginBlock(self.hubert.feature_projection, ipu_id=0)
        self.hubert.encoder = poptorch.BeginBlock(self.hubert.encoder, ipu_id=0)

        layer_ipu = get_layer_ipu(self.ipu_config.layers_per_ipu, self.hubert.encoder.layers)
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

    def deparallelize(self):
        """
        Undo the changes to the model done by `parallelize`.
        """
        super().deparallelize()
        self.change_hubert_encoder_class(True)
        return self
