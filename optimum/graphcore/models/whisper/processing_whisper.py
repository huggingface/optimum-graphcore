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

from transformers import WhisperProcessor

from .feature_extraction_whisper import WhisperFeatureExtractorTorch


class WhisperProcessorTorch(WhisperProcessor):
    """
    Constructs a Whisper processor which wraps a Whisper feature extractor and a Whisper tokenizer into a single
    processor.

    The feature extractor is replaced by a more efficient version of the numpy based `WhisperFeatureExtractor`
    with a torch one.
    """

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)
        self.feature_extractor.__class__ = WhisperFeatureExtractorTorch
