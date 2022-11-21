import copy
from dataclasses import asdict
import logging
from typing import List
import numpy as np
import torch

import torch
from tqdm import tqdm

from peacasso.datamodel import GeneratorConfig
from peacasso.utils import preprocess_image, preprocess_mask, slerp
from peacasso.pipelines import StableDiffusionPipeline

logger = logging.getLogger("peacasso")


class Runner:
    """Class to image generation applications for Peacasso"""

    def __init__(
        self,
    ) -> None:
        pass

    def get_latents(self, config: GeneratorConfig, pipe: StableDiffusionPipeline) -> torch.Tensor:
        """Get latent vector"""
        mask = None
        latents_dtype = pipe.text_encoder.dtype
        if config.seed is not None and config.seed != "":
            generator = torch.Generator(device=pipe.device).manual_seed(config.seed)
        else:
            # create a random seed and use it to create a generator
            seed = torch.randint(0, 2 ** 32, (1,)).item()
            generator = torch.Generator(device=pipe.device).manual_seed(seed)

        batch_size = 1 if (
            isinstance(
                config.prompt, str) or (
                config.use_prompt_weights and config.prompt_weights is not None)) else len(
            config.prompt)
        # get latent vector
        if config.init_image is None:  # text prompt mode
            if config.height % 8 != 0 or config.width % 8 != 0:
                raise ValueError(
                    f"`height` and `width` have to be divisible by 8 but are {config.height} and {config.width}."
                )
            # get the intial random noise
            latents_shape = (
                batch_size * config.num_images,
                pipe.unet.in_channels,
                config.height // 8,
                config.width // 8)
            if config.latents is None:
                if pipe.device.type == "mps":
                    # randn does not work reproducibly on mps
                    latents = torch.randn(
                        latents_shape,
                        generator=generator,
                        device="cpu",
                        dtype=latents_dtype).to(
                        pipe.device)
                else:
                    latents = torch.randn(
                        latents_shape,
                        generator=generator,
                        device=pipe.device,
                        dtype=latents_dtype)
            else:
                if config.latents.shape != latents_shape:
                    raise ValueError(
                        f"Unexpected latents shape, got {config.latents.shape}, expected {latents_shape}")
                latents = config.latents.to(pipe.device)
        else:  # image prompt mode
            if config.strength < 0 or config.strength > 1:
                raise ValueError(
                    f"The value of strength should in [0.0, 1.0] but is {config.strength}"
                )
            if not isinstance(config.init_image, torch.FloatTensor):
                init_image = preprocess_image(config.init_image)

            init_image = init_image.to(device=pipe.device, dtype=latents_dtype)
            init_latent_dist = pipe.vae.encode(init_image).latent_dist
            init_latents = init_latent_dist.sample(generator=generator)
            init_latents = 0.18215 * init_latents

            # expand init_latents for batch_size
            init_latents = torch.cat([init_latents] * (batch_size * config.num_images), dim=0)

            # handle mask if provided
            if init_image is not None and config.mask_image is not None:
                if not isinstance(config.mask_image, torch.FloatTensor):
                    mask_image = preprocess_mask(config.mask_image)
                mask_image = mask_image.to(device=pipe.device, dtype=latents_dtype)
                mask = torch.cat([mask_image] * batch_size * config.num_images)

                # check sizes
                if not mask.shape == init_latents.shape:
                    raise ValueError(
                        f"The mask {str(mask.shape)} and init_latents {str(init_latents.shape)} should be the same size!")

            latents = init_latents
        return latents, mask

    def get_text_embedding(self, config: GeneratorConfig, pipe: StableDiffusionPipeline) -> None:
        """Get text embedding"""
        # get prompt text embeddings

        if config.text_embeddings is not None:
            return config.text_embeddings

        if isinstance(config.prompt, str):
            batch_size = 1
        elif isinstance(config.prompt, list):
            batch_size = len(config.prompt)
        else:
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(config.prompt)}"
            )
        text_inputs = pipe.tokenizer(
            config.prompt,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            return_tensors="pt",
            truncation=True,
        )
        text_input_ids = text_inputs.input_ids

        # provide warning if prompt is too long for the model
        if text_input_ids.shape[-1] > pipe.tokenizer.model_max_length:
            removed_text = pipe.tokenizer.batch_decode(
                text_input_ids[:, pipe.tokenizer.model_max_length:])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {pipe.tokenizer.model_max_length} tokens: {removed_text}")
            text_input_ids = text_input_ids[:, : pipe.tokenizer.model_max_length]
        text_embeddings = pipe.text_encoder(text_input_ids.to(pipe.device))[0]

        # apply prompt weights if provided
        if config.use_prompt_weights and config.prompt_weights is not None:
            logging.info("applying prompt_weights" + str(config.prompt_weights))
            if len(config.prompt_weights) != len(config.prompt):
                raise ValueError("Length of prompt should be same as weights.")
            else:
                for i in range(len(config.prompt_weights)):
                    text_embeddings[i] = text_embeddings[i] * config.prompt_weights[i]
                # sum embeddings
                text_embeddings = torch.sum(text_embeddings, dim=0, keepdim=True)
            text_embeddings = torch.mean(text_embeddings, dim=0).unsqueeze(0)
            batch_size = text_embeddings.shape[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, config.num_images, 1)
        text_embeddings = text_embeddings.view(bs_embed * config.num_images, seq_len, -1)

        do_classifier_free_guidance = config.guidance_scale > 1.0
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if config.negative_prompt is None:
                uncond_tokens = [""]
            elif not isinstance(config.prompt, type(config.negative_prompt)):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(config.negative_prompt)} !="
                    f" {type(config.prompt)}.")
            elif isinstance(config.negative_prompt, str):
                uncond_tokens = [config.negative_prompt]
            elif batch_size != len(config.negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {config.negative_prompt} has batch size {len(config.negative_prompt)}, but `prompt`:"
                    f" {config.prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`.")
            else:
                uncond_tokens = config.negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = pipe.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(pipe.device))[0]

            # duplicate unconditional embeddings for each generation per prompt, using
            # mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(batch_size, config.num_images, 1)
            uncond_embeddings = uncond_embeddings.view(
                batch_size * config.num_images, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def run(self, config: GeneratorConfig, pipe: StableDiffusionPipeline) -> None:
        """Run application"""
        if config.application is None or "type" not in config.application:
            text_embeddings = self.get_text_embedding(config, pipe)
            latents, mask = self.get_latents(config, pipe)

            config = asdict(config)
            config["text_embeddings"] = text_embeddings
            config["latents"] = latents
            config["mask"] = mask
            return pipe(**config)

        else:
            if "type" in config.application:
                app_type = config.application["type"]
                app_config = config.application["config"]
                if app_type == "interpolate":
                    config.num_images = 1  # batch size must be 1 for interpolation
                    if app_config is None:
                        raise ValueError(
                            "application_config must be provided for interpolate application")
                    else:
                        if "num_steps" not in app_config:
                            raise ValueError(
                                "num_steps must be provided for interpolate configuration")
                        else:
                            num_steps = app_config["num_steps"]
                            start_config, end_config = copy.copy(config), copy.copy(config)
                            # get start and end latents. note that these will be based on seed if
                            # provided, images if provided and masks if provided
                            if "seed" in app_config:
                                start_config.seed = app_config["seed"]["start"]
                                end_config.seed = app_config["seed"]["end"]
                            if "image" in app_config:
                                start_config.init_image = app_config["image"]["start"]
                                end_config.init_image = app_config["image"]["end"]
                            start_latents, start_mask = self.get_latents(start_config, pipe)
                            end_latents, end_mask = self.get_latents(end_config, pipe)
                            if "prompt" in app_config:
                                start_config.prompt = app_config["prompt"]["start"]
                                end_config.prompt = app_config["prompt"]["end"]
                                start_text_embeddings = self.get_text_embedding(start_config, pipe)
                                end_text_embeddings = self.get_text_embedding(end_config, pipe)

                            # interpolate between start and end configs
                            interpolation_results = []
                            text_embeddings = self.get_text_embedding(config, pipe)
                            conf_dict = asdict(config)
                            for t in tqdm(np.linspace(0, 1, num_steps)):
                                latents = slerp(float(t), start_latents, end_latents)
                                if "prompt" in app_config:
                                    text_embeddings = slerp(
                                        float(t), start_text_embeddings, end_text_embeddings)
                                # mask = slerp(float(t), start_mask, end_mask)
                                conf_dict["text_embeddings"] = text_embeddings
                                conf_dict["latents"] = latents
                                conf_dict["mask"] = end_mask
                                result = pipe(**conf_dict)
                                interpolation_results.append(result)
                            return interpolation_results
                else:
                    raise ValueError(f"application type {app_type} not supported")
