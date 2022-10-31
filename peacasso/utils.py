from ast import Str
import base64
import json
from typing import Any, List
import os
from PIL import Image
import io

from traitlets import Float

from peacasso.pipelines import StableDiffusionPipeline


def get_dirs(path: str) -> List[str]:
    return next(os.walk(path))[1]


def base64_to_pil(base64_string: str) -> Image:
    # base64_string = base64_string.replace("data:image/png;base64,", "")
    base64_string = base64_string.split(",")[1]
    img_bytes = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_bytes))
    # print number of channels
    mask = None
    if img.mode == "RGBA":
        mask = img.getchannel("A")
        img = img.convert("RGB")
    return img, mask


def pil_to_base64(img: Image) -> str:
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img.close()
    return base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")


    