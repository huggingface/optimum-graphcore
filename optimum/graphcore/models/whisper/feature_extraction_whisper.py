# Copyright 2022 The HuggingFace Inc. team.
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
import numpy as np
import torch
from transformers import WhisperFeatureExtractor


class WhisperFeatureExtractorTorch(WhisperFeatureExtractor):
    """
    A more efficient version of the numpy based `WhisperFeatureExtractor` which simply replaces
    most of the upstream transformers `_np_extract_fbank_features` with torch ops.

    When replacing numpy ops with torch ops, the resulting code is similar to the original OpenAI
    https://github.com/openai/whisper/blob/main/whisper/audio.py::log_mel_spectrogram licensed under
    the MIT License.

    Main differences from above function are omission of padding, and `mel_filters` are precomputed
    using the upstream transformers np implementation instead of librosa.
    """

    def _np_extract_fbank_features(self, waveform: np.array) -> np.ndarray:
        """
        Compute the log-Mel spectrogram of the provided audio. This is almost a copy of upstream,
        with np replaced by torch.
        """
        if not torch.is_tensor(self.mel_filters):
            self.mel_filters = torch.from_numpy(self.mel_filters).to(torch.float32).T

        waveform = torch.from_numpy(waveform)

        window = torch.hann_window(self.n_fft)

        stft = torch.stft(waveform, self.n_fft, self.hop_length, window=window, return_complex=True)
        magnitudes = torch.abs(stft[:, :-1]) ** 2

        mel_spec = self.mel_filters @ magnitudes

        log_spec = torch.log10(torch.clamp(mel_spec, min=1e-10))
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        return log_spec.numpy()
