# from dataclasses import dataclass
from dataclasses import field
from typing import Any, List,  Optional, Union
from pydantic.dataclasses import dataclass


@dataclass
class GeneratorConfig:
    """Configuration for a generation"""

    prompt: Union[str, List[str]]
    num_images: int = 1
    mode: str = "prompt"   # prompt, image, mask
    height: Optional[int] = 512
    width: Optional[int] = 512
    num_inference_steps: Optional[int] = 50
    guidance_scale: Optional[float] = 7.5
    eta: Optional[float] = 0.0
    # generator: Optional[Any] = None
    output_type: Optional[str] = "pil"
    strength: float = 0.8
    init_image: Any = None
    return_intermediates: bool = False
