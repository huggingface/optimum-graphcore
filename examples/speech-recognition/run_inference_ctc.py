#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 Graphcore Ltd. All rights reserved.
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

""" Run inference on a 🤗 Wav2Vec2 model """

import logging
from tqdm import tqdm
from dataclasses import dataclass, field

import torch
import poptorch

from datasets import load_dataset
from optimum.graphcore import IPUConfig
from optimum.graphcore.modeling_utils import to_pipelined
from transformers import (
    AutoModelForCTC,
    Wav2Vec2Processor,
    HfArgumentParser,
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.15.0.dev0")

require_version("datasets>=1.18.0", "To fix: pip install -r examples/pytorch/speech-recognition/requirements.txt")

logger = logging.getLogger(__name__)


@dataclass
class InferenceArguments:
    """
    Arguments for inference.
    """

    num_device_iterations: int = field(
        default=32,
        metadata={"help": "Number of iterations the IPU will run before returning to host."},
    )

    max_samples: int = field(
        default=320000,
        metadata={"help": "Length of vector span to mask along the time axis."},
    )

    use_large_model: bool = field(
        default=False,
        metadata={"help": "Use LARGE model, otherwise use BASE."},
    )

    do_librispeech: bool = field(
        default=True,
        metadata={"help": "Run inference on a portion of LibriSpeech."},
    )

    do_benchmark: bool = field(
        default=True,
        metadata={"help": "Run the inference model without data-loading and compute throughput."},
    )

    benchmark_iterations: int = field(
        default=100,
        metadata={"help": "Benchmark the inference model for this many host iterations."},
    )


def main():
    parser = HfArgumentParser(InferenceArguments)
    inference_args, = parser.parse_args_into_dataclasses()

    logger.info("Inference arguments %s", inference_args)

    num_device_iterations = inference_args.num_device_iterations

    # load model and tokenizer
    if inference_args.use_large_model:
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
        model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-large-960h")
        ipu_config = IPUConfig(matmul_proportion=0.1, inference_device_iterations=num_device_iterations,
                               layers_per_ipu=[17, 16],
                               ipus_per_replica=2)
    else:
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        ipu_config = IPUConfig(inference_device_iterations=num_device_iterations)

    # pipeline for ipu
    ipu_model = to_pipelined(model, ipu_config)
    ipu_model.parallelize()

    opts = ipu_config.to_options(for_inference=True)
    inference_model = poptorch.inferenceModel(ipu_model.half().eval(), options=opts)

    sample_batch = {'input_values': torch.zeros([num_device_iterations, inference_args.max_samples])}

    inference_model.compile(**sample_batch)

    if inference_args.do_librispeech:
        logger.info("*** LibriSpeech ***")
        # load dummy dataset and read soundfiles
        ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")

        # get batch
        x = torch.zeros([num_device_iterations, inference_args.max_samples])
        for i in range(num_device_iterations):
            input_values = processor(ds[i]["audio"]["array"], return_tensors="pt",
                                     padding="longest").input_values  # Batch size 1
            length = input_values.size(1)
            x[i, :length] = input_values[0]

        batch = {'input_values': x}

        # take argmax and decode
        output = inference_model(**batch)
        logits = output[0]
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
        print(transcription)

    if inference_args.do_benchmark:
        logger.info("*** Benchmark ***")
        for i in tqdm(range(inference_args.benchmark_iterations)):
            inference_model(**sample_batch)


if __name__ == "__main__":
    main()
