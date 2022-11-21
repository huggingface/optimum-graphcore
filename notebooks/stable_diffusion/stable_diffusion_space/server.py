import base64
import io
import os
import time

import torch

from fastapi import FastAPI
from ipu_models import IPUStableDiffusionPipeline
from pydantic import BaseModel


_IS_DEBUG = os.getenv("DEBUG", False)

if _IS_DEBUG:
    from PIL import Image
else:
    pipe = IPUStableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        revision="fp16",
        torch_dtype=torch.float16,
    )
    pipe.enable_attention_slicing()
    pipe("Pipeline warmup...")

app = FastAPI()


class StableDiffusionInputs(BaseModel):
    prompt: str
    guidance_scale: float = 7.5


@app.post("/inference/")
async def stable_diffusion(inputs: StableDiffusionInputs):
    start = time.time()
    if _IS_DEBUG:
        image = Image.new("RGB", (512, 512), "blue")
    else:
        images = pipe(inputs.prompt, guidance_scale=inputs.guidance_scale).images
    latency = time.time() - start
    images_b64 = []
    for image in images:
        image_byte_arr = io.BytesIO()
        image.save(image_byte_arr, format="PNG")
        image_byte_arr = image_byte_arr.getvalue()
        images_b64.append(base64.b64encode(image_byte_arr))
    content = {"images": images_b64, "latency": latency}
    return content


@app.get("/")
async def root():
    return {"message": "This is the server running Stable Diffusion v1.5 on Graphcore IPUs!"}
