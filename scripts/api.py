from fastapi import FastAPI, Body

from modules.api.models import *
from modules.api import api

import cv2
import gradio as gr
import numpy as np
from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config, HDStrategy, LDMSampler, SDSampler
from modules import devices, script_callbacks
from PIL import Image


def run_cleaner(input_image, mask_image, cleaner_model_id):
    model = ModelManager(name=cleaner_model_id, device=devices.device)
    input_image = np.array(input_image)
    mask_image = np.array(mask_image.convert("L"))
    config = Config(
        ldm_steps=20,
        ldm_sampler=LDMSampler.ddim,
        hd_strategy=HDStrategy.ORIGINAL,
        hd_strategy_crop_margin=32,
        hd_strategy_crop_trigger_size=512,
        hd_strategy_resize_limit=512,
        prompt="",
        sd_steps=20,
        sd_sampler=SDSampler.ddim
    )
    output_image = model(image=input_image, mask=mask_image, config=config)
    output_image = cv2.cvtColor(output_image.astype(np.uint8), cv2.COLOR_BGR2RGB)
    output_image = Image.fromarray(output_image)
    del model
    return output_image


def inpaint_anything_api(_: gr.Blocks, app: FastAPI):
    @app.post("/sdapi/v2/inpaint-anything/remove-object")
    async def remove_object(
        input_image: str = Body("", title='Remove object input image'),
        mask_image: str = Body("", title='Remove object mask'),
        model: str = Body("", title='Cleaner model id'),
    ):
        image = api.decode_base64_to_image(input_image)
        mask = api.decode_base64_to_image(mask_image)
        output_image = run_cleaner(image, mask, model)
        return {
                "success": True,
                "image": api.encode_pil_to_base64(output_image)
            }


try:
    import modules.script_callbacks as script_callbacks
    script_callbacks.on_app_started(inpaint_anything_api)
except:
    pass
