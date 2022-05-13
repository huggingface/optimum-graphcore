import torch

import poptorch
from optimum.graphcore import IPUConfig, PipelinedBartForConditionalGeneration
from optimum.utils import logging
from transformers import AutoTokenizer


logger = logging.get_logger()

model_name = "hf-internal-testing/tiny-random-bart"
# model_name = "facebook/bart-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
ipu_config = IPUConfig.from_pretrained("examples/translation/tiny_ipu_config.json")
# ipu_config = IPUConfig.from_pretrained("examples/translation/ipu_config.json")
model = PipelinedBartForConditionalGeneration.from_pretrained_transformers(model_name, ipu_config)
model = model.parallelize()

from optimum.graphcore.generation_utils import IPUGenerationMixin

# model.forward = model.generate
model = poptorch.inferenceModel(model.eval(), options=ipu_config.to_options(for_inference=True))
# model._model.forward = IPUGenerationMixin.generate.__get__(model)
# model.__call__ = IPUGenerationMixin.generate.__get__(model)
# model.forward = IPUGenerationMixin.generate.__get__(model)

# Inputs for compilation
# encoder_inputs = tokenizer("UN Chief Says There Is No <mask> in Syria", return_tensors="pt")
# decoder_inputs = tokenizer("I am Michael and I don't care.", return_tensors="pt")
# inputs = {
#     "encoder_outputs": model.get_encoder().to(torch.float32)(**encoder_inputs, return_dict=False),
#     "attention_mask": encoder_inputs["attention_mask"],
# }
# # model.get_encoder().half()
# # inputs["encoder_outputs"] = tuple(x.to(torch.float16) for x in inputs["encoder_outputs"])
# inputs["decoder_input_ids"] = decoder_inputs["input_ids"]
# inputs = {k: v.to(torch.long) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
# bs = 20
# num_beams = 3
# inputs = {k: v.repeat(bs * num_beams, 1) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
# inputs["encoder_outputs"] = tuple(x.repeat(bs * num_beams, 1, 1) for x in inputs["encoder_outputs"])
#
# model.compile(**inputs)
# # model.generate = model.generate.__func__.__get__(model, poptorch.PoplarExecutor)
# model.model.forward = model.__call__

# Generation
max_length = 10
num_beams = 3
gen_kwargs = {
    "max_length": torch.tensor(max_length).repeat(10),
    "num_beams": torch.tensor(1).repeat(10),
    # "synced_gpus": False,  # True if is_deepspeed_zero3_enabled() else False,
}

inputs = tokenizer("Hey there baby!", return_tensors="pt")
inputs = {k: torch.zeros_like(v, dtype=torch.long) for (k, v) in inputs.items()}
# inputs = {k: v.repeat(bs, 1) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

print("Starting generate")

# generated_tokens = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], **gen_kwargs)
# import pdb; pdb.set_trace()
# from inspect import signature
# sig = signature(model.forward)
# inputs = {
#     **inputs,
#     **gen_kwargs
# }
# kwargs = {p.name: inputs.get(p.name, p.default) for p in sig.parameters.values()}
# model.compile(**kwargs)
model.compile(inputs["input_ids"].repeat(10, 1).clone(), torch.tensor(max_length).repeat(10).clone(), torch.tensor(1).repeat(10).clone(),  attention_mask=inputs["attention_mask"].repeat(10, 1).clone())
# generated_tokens = model.forward(inputs["input_ids"], attention_mask=inputs["attention_mask"], **gen_kwargs)
generated_tokens = model(inputs["input_ids"].repeat(10, 1), attention_mask=inputs["attention_mask"].repeat(10, 1), **gen_kwargs)

print("Done", generated_tokens)
