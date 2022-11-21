import base64
from typing import List
import os
from PIL import Image
import io
import PIL
import torch
import numpy as np

from peacasso.datamodel import GeneratorConfig


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


def sanitize_config(config: GeneratorConfig) -> GeneratorConfig:
    """Sanitize config, remove non serializable fields"""

    if config.init_image:
        config.init_image = None
    if config.mask_image:
        config.mask_image = None
    config.callback = None
    return config


def preprocess_image(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def preprocess_mask(mask):
    w, h = mask.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    mask = mask.resize((w // 8, h // 8), resample=PIL.Image.NEAREST)
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = np.tile(mask, (4, 1, 1))
    mask = mask[None].transpose(0, 1, 2, 3)  # what does this step do?
    # mask = 1 - mask  # repaint white, keep black
    mask = torch.from_numpy(mask)
    return mask


def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """ helper function to spherically interpolate two arrays v1 v2 """

    with torch.no_grad():
        if not isinstance(v0, np.ndarray):
            inputs_are_torch = True
            input_device = v0.device
            v0 = v0.cpu().numpy()
            v1 = v1.cpu().numpy()

        dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
        if np.abs(dot) > DOT_THRESHOLD:
            v2 = (1 - t) * v0 + t * v1
        else:
            theta_0 = np.arccos(dot)
            sin_theta_0 = np.sin(theta_0)
            theta_t = theta_0 * t
            sin_theta_t = np.sin(theta_t)
            s0 = np.sin(theta_0 - theta_t) / sin_theta_0
            s1 = sin_theta_t / sin_theta_0
            v2 = s0 * v0 + s1 * v1

        if inputs_are_torch:
            v2 = torch.from_numpy(v2).to(input_device)

        return v2
