# from dataclasses import dataclass
from dataclasses import field
from random import seed
from typing import Any, List, Optional, Union
from pydantic.dataclasses import dataclass


@dataclass
class GeneratorConfig:
    """Configuration for a generation"""

    prompt: Union[str, List[str]]
    num_images: int = 1
    height: Optional[int] = 512
    width: Optional[int] = 512
    num_inference_steps: Optional[int] = 50
    guidance_scale: Optional[float] = 7.5
    eta: Optional[float] = 0.0
    # generator: Optional[Any] = None
    output_type: Optional[str] = "pil"
    strength: float = 0.8
    init_image: Optional[Any] = None
    seed: Optional[Union[int, None]] = None  # e.g. 2147483647
    return_intermediates: bool = False
    mask_image: Optional[Any] = None
    attention_slice: Optional[Union[str, int]] = None
    negative_prompt: Optional[Union[str, List[str]]] = None
    latents: Optional[Any] = None
