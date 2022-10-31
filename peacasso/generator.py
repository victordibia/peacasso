from dataclasses import asdict
import logging
from torch import autocast
from PIL import Image
from typing import List, Optional
# from diffusers import StableDiffusionPipeline

import os
import torch
import time

from peacasso.datamodel import GeneratorConfig
from peacasso.utils import prompt_arithmetic
from peacasso.pipelines import StableDiffusionPipeline

logger = logging.getLogger(__name__)


class ImageGenerator:
    """Generate image from prompt"""

    def __init__(
        self,
        model: str = "runwayml/stable-diffusion-v1-5",
        token: str = os.environ.get("HF_API_TOKEN"),
        cuda_device: int = 0,
        revision: str = "fp16",
        torch_dtype: Optional[torch.FloatTensor] = torch.float16
    ) -> None:

        assert token is not None, "HF_API_TOKEN environment variable must be set."
        self.device = f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu"
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model,
            revision=revision,
            torch_dtype=torch_dtype,
            use_auth_token=token,
        ).to(self.device)

    def generate(self, config: GeneratorConfig) -> Image:
        """Generate image from prompt"""
        return self.pipe(**asdict(config))

    def list_cuda(self) -> List[int]:
        """List available cuda devices
        Returns:
            List[int]: List of available cuda devices
        """
        available_gpus = [i for i in range(torch.cuda.device_count())]
        return available_gpus
