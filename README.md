# Peacasso

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/victordibia/peacasso/blob/master/notebooks/tutorial.ipynb)

Peacasso [Beta] is a UI tool to help you generate art (and experiment) with multimodal (text, image) AI models (stable diffusion). This project is still in development (see roadmap below).

![](https://github.com/victordibia/peacasso/blob/master/docs/images/screenpc.png?raw=true)

## Why Use Peacasso?

<img width="100%" src="https://github.com/victordibia/peacasso/blob/master/docs/images/peacasso.gif?raw=true" />

Because you deserve a nice UI and great workflow that makes exploring stable diffusion models fun! But seriously, here are a few things that make Peacasson interesting:

- **Easy installation**. Instead of cobbling together command line scripts, Peacasso provides a `pip install` flow and a UI that supports a set of curated default operations.
- **UI with good defaults**. The current implementation of Peacasso provides a UI for basic operations - text and image based prompting (+ inpainting), remixing generated images as prompts, model parameter selection, image download. Also covers the little things .. like light and dark mode.
- **Python API**. While the UI is the focus here, there is an underlying python api which will bake in experimentation features (e.g. saving intermediate images in the sampling loop, exploring model explanations etc. . see roadmap below).

Clearly, Peacasso (UI) might not be for those interested in low level code.

## Requirements and Installation

- Step 1: **Verify Environment - Pythong 3.7+ and CUDA**
  Setup and verify that your python environment is `python 3.7` or higher (preferably, use Conda). Also verify that you have CUDA installed correctly (`torch.cuda.is_available()` is true) and your GPU has about [7GB of VRAM memory](https://stability.ai/blog/stable-diffusion-public-release).

Once requirements are met, run the following command to install the library:

```bash
pip install peacasso
```

Want to stay on the bleeding edge of updates (which might be buggy)? Install directly from the repo:

```bash
git clone https://github.com/victordibia/peacasso.git
cd  peacasso
pip install -e .
```

Don't have a GPU, you can still use the python api and UI in a colab notebook. See this [colab notebook](https://colab.research.google.com/github/victordibia/peacasso/blob/master/notebooks/tutorial.ipynb) for more details.

## Usage - UI and Python API

You can use the library from the ui by running the following command:

```bash
peacasso ui  --port=8080
```

Then navigate to http://localhost:8080/ in your browser.

You can also use the python api by running the following command:

```python

from peacasso.generator import ImageGenerator
from peacasso.datamodel import GeneratorConfig, ModelConfig

# model configuration
model_config: ModelConfig = ModelConfig(
    device="cuda:0" , # device ..cpu, cuda, cuda:0
    model="nitrosocke/mo-di-diffusion",
    revision="main", # HF model branch
    token=None, # HF_TOKEN here if needed
)

prompt = "victorian ampitheater of sand, pillars with statues on top, lamps on ground, by peter mohrbacher dan mumford craig mullins nekro, cgsociety, pixiv, volumetric light, 3 d render"
prompt_config = GeneratorConfig(
    prompt=prompt,
    num_images=3,
    width=512,
    height=512,
    guidance_scale=7.5,
    num_inference_steps=20,
    return_intermediates=True, # return intermediate images during diffusion sampling
    seed=6010691039
)
result = gen.generate(prompt_config)
result = gen.generate(prompt_config)
for i, image in enumerate(result["images"]):
    image.save(f"image_{i}.png")

# result["intermediates"] contains the intermediate images
```

![](https://github.com/victordibia/peacasso/blob/master/docs/images/prompt_result.png?raw=true)

visualizing intermediate images during the diffusion loop.
![](https://github.com/victordibia/peacasso/blob/master/docs/images/intermediates.png?raw=true)

## Design Philosophy

Features in `Peacasso` are being designed based on insights from communication theory [^1] and also research on Human-AI interaction design [^2]. Learn more about the design and components in peacasso in the paper [here](https://github.com/victordibia/peacasso/blob/master/docs/images/paper.pdf).

<img width="100%" src="https://github.com/victordibia/peacasso/blob/master/docs/images/mrt.png?raw=true" />

A general vision for the Peacasso architecture is shown below (parts of this are still being implemented):

<img width="100%" src="https://github.com/victordibia/peacasso/blob/master/docs/images/peacasso_arch.png?raw=true" />

## Features and Road Map

- [x] Command line interface
- [x] UI Features. Query models with multiple parametrs
  - [x] Prompting: Text prompting (text2img), Image based prompting (img2img), Inpainting (img2img)
  - [ ] Editor (for outpainting)
  - [ ] Latent space exploration
- [ ] Experimentation tools
  - [x] Save intermediate images in the sampling loop
  - [x] Weighted prompt mixing
  - [ ] Prompt recommendation
  - [ ] Curation/sharing experiment results
  - [ ] Defined Workflows (e.g., tiles, composition etc.)
  - [ ] Model explanations

## Acknowledgement

This work builds on the stable diffusion model and code is adapted from the HuggingFace [implementation](https://huggingface.co/blog/stable_diffusion). Please note the - [CreativeML Open RAIL-M](https://huggingface.co/spaces/CompVis/stable-diffusion-license) license associated with the stable diffusion model.

## Citation

If you use `peacasso` for in your research or adopt the design guidelines used to build `peacasso`, please consider citing as follows:

```bibtex
@misc{dibia2022peacasso,
      title={Interaction Design for Systems that Integrate Image Generation Models: A Case Study with Peacasso},
      author={Victor Dibia},
      year={2022},
      publisher={GitHub},
      journal={GitHub repository},
      year={2021},
      primaryClass={cs.CV}
}
```

## References

[^1]:
    Richard L Daft and Robert H Lengel. 1986. Organizational information require-
    ments, media richness and structural design. Management science

[^2]:
    Saleema Amershi, Dan Weld, Mihaela Vorvoreanu, Adam Fourney, Besmira Nushi,
    Penny Collisson, Jina Suh, Shamsi Iqbal, Paul N Bennett, Kori Inkpen, et al. 2019.
    Guidelines for human-AI interaction. In Proceedings of the 2019 chi conference on
    human factors in computing systems.
