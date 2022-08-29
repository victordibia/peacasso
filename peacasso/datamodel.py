# from dataclasses import dataclass
from dataclasses import field
from typing import Optional
from pydantic.dataclasses import dataclass


@dataclass
class PromptConfig:
    """Configuration for a generation"""

    prompt: str
    num_images: int = 1
    width: int = 512
    height: int = 512
    guidance_scale: float = 7.5
    num_inference_steps: int = 50

 