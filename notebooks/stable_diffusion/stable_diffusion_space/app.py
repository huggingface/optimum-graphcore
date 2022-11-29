import os
import re

from datasets import load_dataset
from PIL import Image

import gradio as gr
import requests


# from share_btn import community_icon_html, loading_icon_html, share_js

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"


def infer(prompt: str, guidance_scale: float = 7.5):
    url = os.getenv("IPU_BACKEND")
    api_key = os.getenv("API_KEY")
    payload = {"prompt": prompt, "guidance_scale": guidance_scale}
    headers = {"access_token": api_key}
    response = requests.post(url, json=payload, headers=headers)
    response_content = response.json()
    images = []
    for image in response_content["images"]:
        image_b64 = f"data:image/jpeg;base64,{image}"
        images.append(image_b64)
    return images, response_content["latency"]


css = """
        .gradio-container {
            font-family: 'IBM Plex Sans', sans-serif;
        }
        .gr-button {
            color: white;
            border-color: black;
            background: black;
        }
        input[type='range'] {
            accent-color: black;
        }
        .dark input[type='range'] {
            accent-color: #dfdfdf;
        }
        .container {
            max-width: 730px;
            margin: auto;
            padding-top: 1.5rem;
        }
        #gallery {
            min-height: 22rem;
            margin-bottom: 15px;
            margin-left: auto;
            margin-right: auto;
            border-bottom-right-radius: .5rem !important;
            border-bottom-left-radius: .5rem !important;
        }
        #gallery>div>.h-full {
            min-height: 20rem;
        }
        .details:hover {
            text-decoration: underline;
        }
        .gr-button {
            white-space: nowrap;
        }
        .gr-button:focus {
            border-color: rgb(147 197 253 / var(--tw-border-opacity));
            outline: none;
            box-shadow: var(--tw-ring-offset-shadow), var(--tw-ring-shadow), var(--tw-shadow, 0 0 #0000);
            --tw-border-opacity: 1;
            --tw-ring-offset-shadow: var(--tw-ring-inset) 0 0 0 var(--tw-ring-offset-width) var(--tw-ring-offset-color);
            --tw-ring-shadow: var(--tw-ring-inset) 0 0 0 calc(3px var(--tw-ring-offset-width)) var(--tw-ring-color);
            --tw-ring-color: rgb(191 219 254 / var(--tw-ring-opacity));
            --tw-ring-opacity: .5;
        }
        #advanced-btn {
            font-size: .7rem !important;
            line-height: 19px;
            margin-top: 12px;
            margin-bottom: 12px;
            padding: 2px 8px;
            border-radius: 14px !important;
        }
        #advanced-options {
            display: none;
            margin-bottom: 20px;
        }
        .footer {
            margin-bottom: 45px;
            margin-top: 35px;
            text-align: center;
            border-bottom: 1px solid #e5e5e5;
        }
        .footer>p {
            font-size: .8rem;
            display: inline-block;
            padding: 0 10px;
            transform: translateY(10px);
            background: white;
        }
        .dark .footer {
            border-color: #303030;
        }
        .dark .footer>p {
            background: #0b0f19;
        }
        .acknowledgments h4{
            margin: 1.25em 0 .25em 0;
            font-weight: bold;
            font-size: 115%;
        }
        #container-advanced-btns{
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
        }
        .animate-spin {
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            from {
                transform: rotate(0deg);
            }
            to {
                transform: rotate(360deg);
            }
        }
        #share-btn-container {
            display: flex; padding-left: 0.5rem !important; padding-right: 0.5rem !important; background-color: #000000; justify-content: center; align-items: center; border-radius: 9999px !important; width: 13rem;
        }
        #share-btn {
            all: initial; color: #ffffff;font-weight: 600; cursor:pointer; font-family: 'IBM Plex Sans', sans-serif; margin-left: 0.5rem !important; padding-top: 0.25rem !important; padding-bottom: 0.25rem !important;
        }
        #share-btn * {
            all: unset;
        }
        .gr-form{
            flex: 1 1 50%; border-top-right-radius: 0; border-bottom-right-radius: 0;
        }
        #prompt-container{
            gap: 0;
        }
"""

block = gr.Blocks(css=css)

examples = [
    ["A high tech solarpunk utopia in the Amazon rainforest"],
    ["A pikachu fine dining with a view to the Eiffel Tower"],
    ["A mecha robot in a favela in expressionist style"],
    ["An insect robot preparing a delicious meal"],
    ["A small cabin on top of a snowy mountain in the style of Disney, artstation"],
]


with block:
    gr.HTML(
        """
            <div style="text-align: center; max-width: 650px; margin: 0 auto;">
              <div
                style="
                  display: inline-flex;
                  align-items: center;
                  gap: 0.8rem;
                  font-size: 1.75rem;
                "
              >
                <svg
                  width="0.65em"
                  height="0.65em"
                  viewBox="0 0 115 115"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <rect width="23" height="23" fill="white"></rect>
                  <rect y="69" width="23" height="23" fill="white"></rect>
                  <rect x="23" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="23" y="69" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="46" width="23" height="23" fill="white"></rect>
                  <rect x="46" y="69" width="23" height="23" fill="white"></rect>
                  <rect x="69" width="23" height="23" fill="black"></rect>
                  <rect x="69" y="69" width="23" height="23" fill="black"></rect>
                  <rect x="92" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="92" y="69" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="115" y="46" width="23" height="23" fill="white"></rect>
                  <rect x="115" y="115" width="23" height="23" fill="white"></rect>
                  <rect x="115" y="69" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="92" y="46" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="92" y="115" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="92" y="69" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="46" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="115" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="69" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="46" y="46" width="23" height="23" fill="black"></rect>
                  <rect x="46" y="115" width="23" height="23" fill="black"></rect>
                  <rect x="46" y="69" width="23" height="23" fill="black"></rect>
                  <rect x="23" y="46" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="23" y="115" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="23" y="69" width="23" height="23" fill="black"></rect>
                </svg>
                <h1 style="font-weight: 900; margin-bottom: 7px;">
                  Stable Diffusion Demo
                </h1>
              </div>
              <p style="margin-bottom: 10px; font-size: 94%">
                Stable Diffusion is a state of the art text-to-image model that generates
                images from text.
              </p>
              <p>
                Model by <a href="https://huggingface.co/runwayml" style="text-decoration: underline;" target="_blank">Runway</a> and the backend is running on <a href="https://github.com/huggingface/optimum-graphcore", style="text-decoration: underline;" target="_blank">Optimum Graphcore</a> and <a href="https://huggingface.co/Graphcore" style="text-decoration: underline;" target="_blank">Graphcore IPUs</a>
                </p>
              </p>
            </div>
        """
    )
    with gr.Group():
        with gr.Box():
            with gr.Row(elem_id="prompt-container").style(mobile_collapse=False, equal_height=True):
                text = gr.Textbox(
                    label="Enter your prompt",
                    show_label=False,
                    max_lines=1,
                    placeholder="Enter your prompt",
                    elem_id="prompt-text-input",
                ).style(
                    border=(True, False, True, True),
                    rounded=(True, False, False, True),
                    container=False,
                )
                btn = gr.Button("Generate image").style(
                    margin=False,
                    rounded=(False, True, True, False),
                    full_width=False,
                )

        gallery = gr.Gallery(label="Generated images", show_label=False, elem_id="gallery").style(
            grid=[1], height="auto"
        )

        latency = gr.Textbox(label="Compute time (in seconds)")

        with gr.Group(elem_id="container-advanced-btns"):
            advanced_button = gr.Button("Advanced options", elem_id="advanced-btn")
            # with gr.Group(elem_id="share-btn-container"):
            #     community_icon = gr.HTML(community_icon_html)
            #     loading_icon = gr.HTML(loading_icon_html)
            #     share_button = gr.Button("Share to community", elem_id="share-btn")

        # with gr.Row(elem_id="advanced-options"):
        #     gr.Markdown("Advanced settings are temporarily unavailable")
        #     samples = gr.Slider(label="Images", minimum=1, maximum=4, value=4, step=1)
        #     steps = gr.Slider(label="Steps", minimum=1, maximum=50, value=45, step=1)
        #     scale = gr.Slider(
        #         label="Guidance Scale", minimum=0, maximum=50, value=7.5, step=0.1
        #     )
        #     seed = gr.Slider(
        #         label="Seed",
        #         minimum=0,
        #         maximum=2147483647,
        #         step=1,
        #         randomize=True,
        #     )

        ex = gr.Examples(examples=examples, fn=infer, inputs=text, cache_examples=False)
        # outputs=[gallery, community_icon, loading_icon, share_button], cache_examples=False)
        ex.dataset.headers = [""]

        text.submit(infer, inputs=text, outputs=[gallery, latency], postprocess=False)
        btn.click(infer, inputs=text, outputs=[gallery, latency], postprocess=False)

        advanced_button.click(
            None,
            [],
            text,
            _js="""
            () => {
                const options = document.querySelector("body > gradio-app").querySelector("#advanced-options");
                options.style.display = ["none", ""].includes(options.style.display) ? "flex" : "none";
            }""",
        )
        # share_button.click(
        #     None,
        #     [],
        #     [],
        #     _js=share_js,
        # )
        gr.HTML(
            """
                <div class="acknowledgments">
                    <p><h4>LICENSE</h4>
The model is licensed with a <a href="https://huggingface.co/spaces/CompVis/stable-diffusion-license" style="text-decoration: underline;" target="_blank">CreativeML Open RAIL-M</a> license. The authors claim no rights on the outputs you generate, you are free to use them and are accountable for their use which must not go against the provisions set in this license. The license forbids you from sharing any content that violates any laws, produce any harm to a person, disseminate any personal information that would be meant for harm, spread misinformation and target vulnerable groups. For the full list of restrictions please <a href="https://huggingface.co/spaces/CompVis/stable-diffusion-license" target="_blank" style="text-decoration: underline;" target="_blank">read the license</a></p>
                    <p><h4>Biases and content acknowledgment</h4>
                    While the capabilities of image generation models are impressive, they can also reinforce or exacerbate social biases. Stable Diffusion v1 was trained on subsets of LAION-2B(en), which consists of images that are primarily limited to English descriptions. Texts and images from communities and cultures that use other languages are likely to be insufficiently accounted for. This affects the overall output of the model, as white and western cultures are often set as the default. Further, the ability of the model to generate content with non-English prompts is significantly worse than with English-language prompts.
               </div>
           """
        )

block.queue(concurrency_count=40, max_size=20).launch(max_threads=150)
