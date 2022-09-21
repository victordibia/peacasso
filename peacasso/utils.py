import base64
import json
from typing import Any, List
import os
from PIL import Image
import io


def get_dirs(path: str) -> List[str]:
    return next(os.walk(path))[1]


def base64_to_pil(base64_string: str) -> Image:
    # base64_string = base64_string.replace("data:image/png;base64,", "")
    base64_string = base64_string.split(",")[1]
    img_bytes = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_bytes))
    # print number of channels
    print(img.mode)
    mask = None
    if img.mode == "RGBA":
        mask = img.getchannel("A")
        img = img.convert("RGB")
        mask.save("mask.png")
        img.save("img.png")
    return img, mask
