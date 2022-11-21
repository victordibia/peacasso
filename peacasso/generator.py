from dataclasses import asdict
import logging
from PIL import Image
from typing import List, Optional
# from diffusers import StableDiffusionPipeline

import os
import torch
import time

from peacasso.datamodel import GeneratorConfig, ModelConfig
from peacasso.applications import Runner
from peacasso.pipelines import StableDiffusionPipeline

logger = logging.getLogger(__name__)


class ImageGenerator:
    """Class to handle generating image from prompt"""

    def __init__(
        self,
        model_config: ModelConfig
    ) -> None:

        self.create_model(model_config)

    def create_model(self, model_config: ModelConfig):
        # assert model_config. token is not None, "HF_API_TOKEN environment variable must be set."

        device = model_config.device if torch.cuda.is_available() else "cpu"
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_config.model,
            revision=model_config.revision,
            torch_dtype=torch.float16,
            use_auth_token=model_config.token
        ).to(device)

        self.runner = Runner()

    def generate(self, config: GeneratorConfig) -> Image:
        """Generate image from prompt"""
        return self.runner.run(config, self.pipe)

    def reload(self, model_config) -> None:
        """Clear pipe models from memory, reload pipe with new parameters"""
        del self.pipe
        torch.cuda.empty_cache()
        self.create_model(model_config)

    def list_cuda(self) -> List[int]:
        """List available cuda devices
        Returns:
            List[int]: List of available cuda devices
        """
        available_gpus = [i for i in range(torch.cuda.device_count())]
        return available_gpus
