import io
import torch
from ipu_models import IPUStableDiffusionPipeline

from fastapi import FastAPI
from fastapi.responses import Response

pipe = IPUStableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    revision="fp16",
    torch_dtype=torch.float16,
)
pipe.enable_attention_slicing()

app = FastAPI()

# TODO: make prompt a query parameter instead of a path parameter, here this is for convinience during testing.
@app.get("/inference/{prompt}")
def stable_diffusion(prompt: str, guidance_scale: float = 7.5):
    image = pipe(prompt, guidance_scale=guidance_scale).images[0]
    image_byte_arr = io.BytesIO()
    image.save(image_byte_arr, format="PNG")
    image_byte_arr = image_byte_arr.getvalue()
    return Response(content=image_byte_arr, media_type="image/png")


@app.get("/")
async def root():
    return {"message": "This is the server running Stable Diffusion v1.5 on Graphcore IPUs!"}
