# based on
# https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/stable_diffusion
import inspect
import logging
import time
from typing import Callable, List, Optional, Union
import PIL
import torch
import numpy as np

from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

# from ...models import AutoencoderKL, UNet2DConditionModel
# from ...pipeline_utils import DiffusionPipeline
# from ...schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
# from .safety_checker import StableDiffusionSafetyChecker

from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)

logger = logging.getLogger(__name__)


def preprocess(image):
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


def decode_image(
    latents,
    vae,
):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents.to(vae.dtype)).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    return image


class StableDiffusionPipeline(DiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        feature_extractor: CLIPFeatureExtractor,
        safety_checker: StableDiffusionSafetyChecker,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
            safety_checker=safety_checker,
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        strength: float = 0.8,
        init_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        seed: Optional[int] = 2147483647,
        attention_slice: Optional[Union[str, int]] = "auto",
        mask_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        num_images: Optional[int] = 1,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor, PIL.Image.Image], None]] = None,
        callback_steps: Optional[int] = 1,
        return_intermediates: Optional[bool] = False,
        prompt_weights: Optional[Union[List[float], float]] = None,
        use_prompt_weights: Optional[bool] = False,
        text_embeddings: Optional[torch.FloatTensor] = None,
        text_input_ids: Optional[torch.LongTensor] = None,
        mask: Optional[torch.FloatTensor] = None,
        filter_nsfw: Optional[bool] = False,
        **kwargs,
    ):

        intermediate_images = []
        if (callback_steps is None) or (callback_steps is not None and (
                not isinstance(callback_steps, int) or callback_steps <= 0)):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        # set seed
        if seed is not None and seed != "":
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            # create a random seed and use it to create a generator
            seed = torch.randint(0, 2 ** 32, (1,)).item()
            generator = torch.Generator(device=self.device).manual_seed(seed)
        # enable attention slicing if attention_slice is not None
        if attention_slice:
            if attention_slice == "auto":
                self.unet.set_attention_slice(self.unet.config.attention_head_dim // 2)
            else:
                self.unet.set_attention_slice(attention_slice)

        start_time = time.time()
        batch_size = 1 if (
            isinstance(
                prompt, str) or (
                use_prompt_weights and prompt_weights is not None)) else len(prompt)

        self.scheduler.set_timesteps(num_inference_steps)
        timesteps_tensor = self.scheduler.timesteps.to(self.device)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        if init_image:
            init_latents_orig = latents
            offset = self.scheduler.config.get("steps_offset", 0)
            init_timestep = int(num_inference_steps * strength) + offset
            init_timestep = min(init_timestep, num_inference_steps)

            timesteps = self.scheduler.timesteps[-init_timestep]
            timesteps = torch.tensor([timesteps] * batch_size * num_images, device=self.device)

            # add noise to latents using the timesteps
            noise = torch.randn(
                latents.shape,
                generator=generator,
                device=self.device,
                dtype=latents.dtype)
            latents = self.scheduler.add_noise(latents, noise, timesteps)
            # t_start = max(num_inference_steps - init_timestep + offset, 0)
            # timesteps = self.scheduler.timesteps[t_start:].to(self.device)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        for i, t in enumerate((timesteps_tensor)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat(
                [latents] * 2) if do_classifier_free_guidance else latents

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * \
                    (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
            if init_image is not None and mask is not None:
                init_latents_proper = self.scheduler.add_noise(init_latents_orig, noise, t)
                latents = (init_latents_proper * mask) + (latents * (1 - mask))

            decoded_image = None
            if return_intermediates:
                decoded_image = self.numpy_to_pil(decode_image(latents, self.vae))
                intermediate_images.append(decoded_image)

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents, decoded_image)

        # scale and decode the image latents with vae
        image = decode_image(latents, self.vae)
        has_nsfw_concept = None
        if filter_nsfw:

            safety_cheker_input = self.feature_extractor(
                self.numpy_to_pil(image), return_tensors="pt"
            ).to(self.device)
            image, has_nsfw_concept = self.safety_checker(
                images=np.asarray(image), clip_input=safety_cheker_input.pixel_values.to(
                    text_embeddings.dtype))
        image = self.numpy_to_pil(image)

        return {
            "images": image,
            "intermediates": intermediate_images,
            "nsfw_content_detected": has_nsfw_concept,
            "time": time.time() - start_time,
            "seed": seed,
        }
