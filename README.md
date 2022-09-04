# Peacasso

Peacasso is a UI tool to help you generate art (and experiment) with multimodal (text, image) AI models (stable diffusion).

![](docs/images/screenpc.png)

## Requirements and Installation

- Step 1: **HuggingFace Access**

  Access to the diffusion model weights requires a HuggingFace model account and access token. Please create an account at [huggingface.co](https://huggingface.co/), get an [access token](https://huggingface.co/settings/tokens) and agree to the model terms [here](https://huggingface.co/CompVis/stable-diffusion-v1-4). Next, create a `HF_API_TOKEN` environment variable containing your token. `export HF_API_TOKEN=your_token`. Note that the first time you run peacasso, the weights for the SD model are [cached locally](https://huggingface.co/transformers/v4.0.1/installation.html#caching-models) on your machine.

- Step 2: **Verify Environment - Pythong 3.7+ and CUDA**
  Setup and verify that your python environment is `python 3.7` or higher (preferably, use Conda). Also verify that you have CUDA installed correctly (`torch.cuda.is_available()` is true) and your GPU has about [7GB of VRAM memory](https://stability.ai/blog/stable-diffusion-public-release).

Once requirements are met, run the following command to install the library:

```bash
pip install peacasso
```

## Usage - UI and Python API

You can use the library from the ui by running the following command:

```bash
peacasso ui  --port=8080
```

Then navigate to http://localhost:8080/ in your browser.

You can also use the python api by running the following command:

```python

import os
from dotenv import load_dotenv
from peacasso.generator import ImageGenerator
from peacasso.datamodel import GeneratorConfig

token = os.environ.get("HF_API_TOKEN")
gen = ImageGenerator(token=token)
prompt = "A sea lion wandering the streets of post apocalyptic London"

prompt_config = GeneratorConfig(
    prompt=prompt,
    num_images=3,
    width=512,
    height=512,
    guidance_scale=7.5,
    num_inference_steps=50,
)

result = gen.generate(prompt_config)
for i, image in enumerate(result["images"]):
    image.save(f"image_{i}.png")
```

## Features and Road Map

- [x] Command line interface
- [x] UI Features. Query models with multiple parametrs
  - [x] Text prompting
  - [ ] Image based prompting
  - [ ] Image inpainting (masking)
  - [ ] Latent space exploration
- [ ] Curation/sharing experiment results

## Acknowledgement

This work builds on the stable diffusion model and code is adapted from the HuggingFace [implementation](https://huggingface.co/blog/stable_diffusion). Please note the - [CreativeML Open RAIL-M](https://huggingface.co/spaces/CompVis/stable-diffusion-license) license associated with the stable diffusion model.
