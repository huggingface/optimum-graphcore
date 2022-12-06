import base64
import io
import os
import time

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKey, APIKeyHeader
from pydantic import BaseModel
from starlette.status import HTTP_403_FORBIDDEN


_IS_DEBUG = os.getenv("DEBUG", False)

API_KEY = os.getenv("API_KEY", None)
API_KEY_NAME = "access_token"

if _IS_DEBUG:
    from PIL import Image
else:
    import torch

    from ipu_models import IPUStableDiffusionPipeline

    pipe = IPUStableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        revision="fp16",
        torch_dtype=torch.float16,
    )
    pipe.enable_attention_slicing()
    pipe("Pipeline warmup...")

api_key = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

app = FastAPI()


async def get_api_key(api_key: str = Security(api_key)):
    if api_key == API_KEY:
        return api_key
    else:
        raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Could not validate credentials")


class StableDiffusionInputs(BaseModel):
    prompt: str
    guidance_scale: float = 7.5


@app.post("/inference/")
async def stable_diffusion(inputs: StableDiffusionInputs, _: APIKey = Depends(get_api_key)):
    start = time.time()
    if _IS_DEBUG:
        images = [Image.new("RGB", (512, 512), "blue")]
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
