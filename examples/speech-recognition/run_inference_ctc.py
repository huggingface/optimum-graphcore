from tqdm import tqdm

import torch
import poptorch

from datasets import load_dataset
from optimum.graphcore import IPUConfig
from optimum.graphcore.modeling_utils import to_pipelined
from transformers import (
    AutoModelForCTC,
    Wav2Vec2Processor,
)

# load model and tokenizer
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h")
num_device_iterations = 10
ipu_config = IPUConfig(inference_device_iterations=num_device_iterations)

# processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-lv60")
# model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-large-lv60")
# num_device_iterations = 2 # must be multiple of pipeline stages (2)
# ipu_config = IPUConfig(matmul_proportion=0.1, inference_device_iterations=num_device_iterations,
#                        layers_per_ipu=[17, 16],
#                        ipus_per_replica=2)

# pipeline for ipu
ipu_model = to_pipelined(model, ipu_config)
ipu_model.parallelize()

opts = ipu_config.to_options(for_inference=True)
inference_model = poptorch.inferenceModel(ipu_model.half().eval(), options=opts)

sample_batch = {'input_values': torch.zeros([num_device_iterations, 320000])}

inference_model.compileAndExport("./exported_model", **sample_batch)

# load dummy dataset and read soundfiles
ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")

# get batch
x = torch.zeros([num_device_iterations, 320000])
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

# benchmark
for i in tqdm(range(10000)):
    output = inference_model(**sample_batch)
