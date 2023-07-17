#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 Graphcore Ltd. All rights reserved.
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

import argparse
import os.path
import sys

import requests
import torch

from optimum.graphcore import IPUConfig, pipeline
from optimum.graphcore.models.whisper import WhisperFeatureExtractorTorch


torch.set_num_threads(4)

description = r"""Minimal example of the Whisper pipeline on the IPU
Try, for instance:
python run_whisper_pipeline.py http://www.archive.org/download/greatexpectations_01_dickens_128kb.mp3 \
    --use-cache \
    --use-encoder-output-buffer \
    --on_device-generation-steps 16 \
    --time-it \
    --batch-size 2 \
    --num-beams 5
Executable caching can be enabled by setting the POPLAR_EXECUTABLE_CACHE_DIR"""
parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)

parser.add_argument("input_file", nargs=1)
parser.add_argument("--model", type=str, default="openai/whisper-tiny.en")
parser.add_argument("--layers-per-ipu", type=int, nargs="*", default=None)
parser.add_argument("--use-cache", action="store_true", help="Enable self-attention KV cache")
parser.add_argument(
    "--use-encoder-output-buffer",
    action="store_true",
    help="Enable optimisation that transfers the encoder output to device only once per chunk",
)
parser.add_argument(
    "--use-cond-encoder",
    action="store_true",
    help="Place encoder and decoder on 1 IPU by conditionally running the encoder at the beginning of generation",
)
parser.add_argument(
    "--on-device-generation-steps",
    type=int,
    default=1,
    help="Number of token generation steps to perform on device before returning to host",
)
parser.add_argument("--do-sample", action="store_true")
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--num-beams", type=int, default=1)
parser.add_argument("--replication-factor", type=int, default=1)
parser.add_argument("--chunk-length-s", type=int, default=30)
parser.add_argument("--max-new-tokens", type=int, default=448)
parser.add_argument("--fp32", action="store_true")
parser.add_argument(
    "--time-it",
    action="store_true",
    help="Display the execution time of the transcription (NOTE: performs the transcription a second time)",
)

args = parser.parse_args()

default_layers_per_ipu = {
    "openai/whisper-tiny.en": [-1, -1],
    "openai/whisper-large-v2": [8, 8, 8, 8, 6, 9, 9, 8],
}
if args.use_cond_encoder:
    args.layers_per_ipu = [-1]
if args.layers_per_ipu is None:
    try:
        args.layers_per_ipu = default_layers_per_ipu[args.model]
    except KeyError:
        raise ValueError(f"No default setting for 'layers_per_ipu' for model '{args.model}'")

print("Parameters:", file=sys.stderr)
print(args, file=sys.stderr)

executable_cache_dir = os.getenv("POPLAR_EXECUTABLE_CACHE_DIR", None)
config_kwargs = {}
if executable_cache_dir is not None:
    config_kwargs.update(executable_cache_dir=executable_cache_dir)


ipu_config = IPUConfig(
    layers_per_ipu=args.layers_per_ipu,
    inference_replication_factor=args.replication_factor,
    parallelize_kwargs={
        "use_cache": args.use_cache,
        "use_encoder_output_buffer": args.use_encoder_output_buffer,
        "use_cond_encoder": args.use_cond_encoder,
        "on_device_generation_steps": args.on_device_generation_steps,
        "num_beams": args.num_beams,
        "batch_size": args.batch_size,
    },
    **config_kwargs,
)
ipu_pipeline = pipeline(
    task="automatic-speech-recognition",
    model=args.model,
    feature_extractor=WhisperFeatureExtractorTorch.from_pretrained(args.model),
    framework="pt",
    ipu_config=ipu_config,
    fp16=not args.fp32,
    batch_size=args.batch_size * args.replication_factor,
    chunk_length_s=args.chunk_length_s,
    ignore_warning=True,
    max_new_tokens=args.max_new_tokens,
)


input_file = args.input_file[0]

fn = os.path.basename(input_file)

if not os.path.exists(fn):
    if not (input_file.startswith("http://") or input_file.startswith("https://")):
        print("Please provide a valid file name or URL", file=sys.stderr)
        sys.exit(1)
    print("Trying to download", input_file, "to current directory", file=sys.stderr)
    r = requests.get(input_file)
    with open(fn, "wb") as fd:
        fd.write(r.content)

whisper_output = ipu_pipeline(
    [fn],
    generate_kwargs={"do_sample": args.do_sample, "num_beams": args.num_beams, "use_cache": args.use_cache},
)

# whisper can emit Unicode chars such as \u2014, em-dash, which can result in
# UnicodeEncodeErrors with some codecs (e.g. latin-1)
text = whisper_output[0]["text"].encode("utf-8").decode("latin-1")
print(f"'{text}'")

if args.time_it:
    import time

    print("\n\nNow running it again to time it...", file=sys.stderr)
    start = time.time()
    whisper_output = ipu_pipeline(
        [fn], generate_kwargs={"do_sample": args.do_sample, "num_beams": args.num_beams, "use_cache": args.use_cache}
    )
    end = time.time()
    print(f"Transcription performed in {end-start:.2f}s\n", file=sys.stderr)
