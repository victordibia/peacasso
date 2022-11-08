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
        **kwargs,
    ):

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
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )

        self.scheduler.set_timesteps(num_inference_steps)
        timesteps_tensor = self.scheduler.timesteps.to(self.device)

        # get prompt text embeddings
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
            truncation=True,
        )
        text_input_ids = text_inputs.input_ids

        # provide warning if prompt is too long for the model
        if text_input_ids.shape[-1] > self.tokenizer.model_max_length:
            removed_text = self.tokenizer.batch_decode(
                text_input_ids[:, self.tokenizer.model_max_length:])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}")
            text_input_ids = text_input_ids[:, : self.tokenizer.model_max_length]
        text_embeddings = self.text_encoder(text_input_ids.to(self.device))[0]

        # apply prompt weights if provided
        if use_prompt_weights and prompt_weights is not None:
            logging.info("applying prompt_weights" + str(prompt_weights))
            if len(prompt_weights) != len(prompt):
                raise ValueError("Length of prompt should be same as weights.")
            else:
                for i in range(len(prompt_weights)):
                    text_embeddings[i] = text_embeddings[i] * prompt_weights[i]
                # sum embeddings
                text_embeddings = torch.sum(text_embeddings, dim=0, keepdim=True)
            text_embeddings = torch.mean(text_embeddings, dim=0).unsqueeze(0)
            batch_size = text_embeddings.shape[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_images, seq_len, -1)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        if init_image is None:  # text prompt mode
            if height % 8 != 0 or width % 8 != 0:
                raise ValueError(
                    f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
                )
            # get the intial random noise
            latents_shape = (
                batch_size * num_images,
                self.unet.in_channels,
                height // 8,
                width // 8)
            latents_dtype = text_embeddings.dtype
            if latents is None:
                if self.device.type == "mps":
                    # randn does not work reproducibly on mps
                    latents = torch.randn(
                        latents_shape,
                        generator=generator,
                        device="cpu",
                        dtype=latents_dtype).to(
                        self.device)
                else:
                    latents = torch.randn(
                        latents_shape,
                        generator=generator,
                        device=self.device,
                        dtype=latents_dtype)
            else:
                if latents.shape != latents_shape:
                    raise ValueError(
                        f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")
                latents = latents.to(self.device)
        else:  # image prompt mode
            if strength < 0 or strength > 1:
                raise ValueError(
                    f"The value of strength should in [0.0, 1.0] but is {strength}"
                )
            if not isinstance(init_image, torch.FloatTensor):
                init_image = preprocess(init_image)
            latents_dtype = text_embeddings.dtype
            init_image = init_image.to(device=self.device, dtype=latents_dtype)
            init_latent_dist = self.vae.encode(init_image).latent_dist
            init_latents = init_latent_dist.sample(generator=generator)
            init_latents = 0.18215 * init_latents

            # expand init_latents for batch_size
            init_latents = torch.cat([init_latents] * num_images, dim=0)
            init_latents_orig = init_latents

            # handle mask if provided
            if init_image is not None and mask_image is not None:
                if not isinstance(mask_image, torch.FloatTensor):
                    mask_image = preprocess_mask(mask_image)
                mask_image = mask_image.to(device=self.device, dtype=latents_dtype)
                mask = torch.cat([mask_image] * batch_size * num_images)

                # check sizes
                if not mask.shape == init_latents.shape:
                    raise ValueError("The mask and init_image should be the same size!")

            # get the original timestep using init_timestep
            offset = self.scheduler.config.get("steps_offset", 0)
            init_timestep = int(num_inference_steps * strength) + offset
            init_timestep = min(init_timestep, num_inference_steps)

            timesteps = self.scheduler.timesteps[-init_timestep]
            timesteps = torch.tensor([timesteps] * batch_size * num_images, device=self.device)

            # add noise to latents using the timesteps
            noise = torch.randn(
                init_latents.shape,
                generator=generator,
                device=self.device,
                dtype=latents_dtype)
            init_latents = self.scheduler.add_noise(init_latents, noise, timesteps)
            latents = init_latents
            t_start = max(num_inference_steps - init_timestep + offset, 0)
            timesteps = self.scheduler.timesteps[t_start:].to(self.device)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""]
            elif not isinstance(prompt, type(negative_prompt)):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}.")
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`.")
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

            # duplicate unconditional embeddings for each generation per prompt, using
            # mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(batch_size, num_images, 1)
            uncond_embeddings = uncond_embeddings.view(
                batch_size * num_images, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # if we use LMSDiscreteScheduler, let's make sure latents are mulitplied by sigmas
        # if isinstance(self.scheduler, LMSDiscreteScheduler):
        #     latents = latents * self.scheduler.sigmas[0]

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        for i, t in enumerate(self.progress_bar(timesteps_tensor)):
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
            if init_image is not None and mask_image is not None:
                init_latents_proper = self.scheduler.add_noise(init_latents_orig, noise, t)
                latents = (init_latents_proper * mask) + (latents * (1 - mask))

            # call the callback, if provided
            if callback is not None and i % callback_steps == 0:
                if return_intermediates:
                    decoded_image = self.numpy_to_pil(decode_image(latents, self.vae))
                    callback(i, t, latents, decoded_image)
                else:
                    callback(i, t, None, None)

        # scale and decode the image latents with vae
        has_nsfw_concept = None
        image = decode_image(latents, self.vae)
        safety_cheker_input = self.feature_extractor(
            self.numpy_to_pil(image), return_tensors="pt"
        ).to(self.device)
        image, has_nsfw_concept = self.safety_checker(
            images=np.asarray(image), clip_input=safety_cheker_input.pixel_values.to(
                text_embeddings.dtype))
        image = self.numpy_to_pil(image)

        # get generation config as key value dict from method signature
        # generation_config = inspect.signature(self.generate).parameters

        return {
            "images": image,
            "nsfw_content_detected": has_nsfw_concept,
            "time": time.time() - start_time,
            "seed": seed,
        }
