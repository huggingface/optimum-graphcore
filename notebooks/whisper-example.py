#!/usr/bin/env python
# coding: utf-8

# # Speech transcription on IPU using Whisper
# 
# This notebook demonstrates speech transcription on the IPU using the [Whisper implementation in 🤗 Transformers library](https://huggingface.co/spaces/openai/whisper) alongside Optimum-Graphcore.
# 
# ##  🤗  Optimum-Graphcore
# 
# 🤗 Optimum Graphcore is the interface between the 🤗 Transformers library and [Graphcore IPUs](https://www.graphcore.ai/products/ipu).
# It provides a set of tools enabling model parallelization and loading on IPUs, training and fine-tuning on all the tasks already supported by Transformers while being compatible with the Hugging Face Hub and every model available on it out of the box.
# 
# 🤗 Optimum Graphcore was designed with one goal in mind: make training and evaluation straightforward for any 🤗 Transformers user while leveraging the complete power of IPUs.
# 



# Generic imports
import os
import torch
from datasets import load_dataset, Dataset

# IPU-specific imports
import poptorch
from optimum.graphcore import IPUConfig
from optimum.graphcore.modeling_utils import to_pipelined

# HF-related imports
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperConfig


# In[4]:


# A class to collect configuration related parameters
from dataclasses import dataclass

@dataclass
class IPUWhisperConf:
    """A data class to collect IPU-related config parameters"""
    model_spec: str
    ipus_per_replica: int
    pod_type: str

ipu_whisper = {
    "tiny": IPUWhisperConf(model_spec='openai/whisper-tiny.en', ipus_per_replica=2, pod_type="pod4"),
    "large": IPUWhisperConf(model_spec='openai/whisper-large-v2', ipus_per_replica=16, pod_type="pod16")
    # Other sizes will become available in due course
}


# In[5]:


# !rm -rf /tmp/whisper_exe_cache


# In[6]:


# Whisper parameters: model size and maximum sequence length
model_size = "tiny"
max_length = 448


# In[7]:


iwc = ipu_whisper[model_size]

# Instantiate processor and model
processor = WhisperProcessor.from_pretrained(iwc.model_spec)
model = WhisperForConditionalGeneration.from_pretrained(iwc.model_spec)


# In[8]:


# Choose here the index of the test example to use
test_idx = 4


# In[9]:


# load dummy dataset and read soundfiles
test_example = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")[test_idx]

input_features = processor(test_example["audio"]["array"], 
                           return_tensors="pt",
                           sampling_rate=test_example['audio']['sampling_rate']).input_features.half()


# In[10]:


print(f"Duration of audio file: {test_example['audio']['array'].shape[-1]/test_example['audio']['sampling_rate']:.1f}s")
print("Expected transcription:",test_example["text"])


# In[11]:


pod_type = os.getenv("GRAPHCORE_POD_TYPE", iwc.pod_type)
executable_cache_dir = os.getenv("POPLAR_EXECUTABLE_CACHE_DIR", "/tmp/whisper_exe_cache/") + "whisper_inference"


# In[12]:


# os.environ["POPLAR_ENGINE_OPTIONS"] = f'{{"autoReport.all":"true", "debug.allowOutOfMemory": "true", "autoReport.directory":"/localdata/paolot/profiles/copyBuffer"}}'


# In[13]:


ipu_config = IPUConfig(executable_cache_dir=executable_cache_dir, ipus_per_replica=iwc.ipus_per_replica)


# In[14]:


# Adapt whisper to run on the IPU
pipelined_model = to_pipelined(model, ipu_config)
pipelined_model = pipelined_model.parallelize(for_generation=True).half()


# In[15]:


# This triggers a compilation the first time around (unless a precompiled model is available)
sample_output = pipelined_model.generate(input_features, max_length=max_length, min_length=3)
transcription = processor.batch_decode(sample_output, skip_special_tokens=True)[0]
print("Transcription:",transcription)
print("The transcription consisted of",sample_output.shape[-1],"tokens.")


# In[16]:
import time
start = time.time()
for _ in range(100):
    sample_output = pipelined_model.generate(input_features, max_length=max_length, min_length=3)
    transcription = processor.batch_decode(sample_output, skip_special_tokens=True)[0]
print(f"Elapsed time for 100 transcriptions: {time.time()-start:0.1f}")

# In[ ]:




