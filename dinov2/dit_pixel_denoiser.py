"""
DiT Pixel-Space Denoiser

Note: Adding noise in pixel space then encoding is not exactly equivalent to
adding noise directly in latent space (due to VAE's clean encoding).
However, for single-step denoising and small t, this approximation often works reasonably well.
For best results with multi-step denoising, consider adding noise directly in latent space.
"""
import torch
import torch.nn as nn
from diffusers.models import AutoencoderKL
from typing import Optional, Tuple

from dinov2.thirdparty.DiT.models import DiT_models
from dinov2.thirdparty.DiT.diffusion import create_diffusion


class DiTPixelDenoiser:
    def __init__(
        self,
        dit_model_path: str,
        dit_model_name: str = "DiT-XL/2",
        image_size: int = 256,
        vae_model: str = "stabilityai/sd-vae-ft-mse",
        device: str = "cuda",
    ):
        """
        Args:
            dit_model_path: Path to pretrained DiT checkpoint (str) or pre-loaded state_dict (dict)
            dit_model_name: DiT model architecture name (e.g., "DiT-XL/2")
            image_size: Input image size (256 or 512)
            vae_model: VAE model name from diffusers
            device: Device to run on
        """
        self.device = device
        self.image_size = image_size
        self.latent_size = image_size // 8
        
        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(vae_model).to(device)
        self.vae.eval()
        self._requires_grad(self.vae, False)
        
        # Load DiT model
        self.dit_model = DiT_models[dit_model_name](
            input_size=self.latent_size
        ).to(device)
        
        # Load checkpoint - handle both file path and pre-loaded state_dict
        if isinstance(dit_model_path, dict):
            checkpoint = dit_model_path
        else:
            checkpoint = torch.load(dit_model_path, map_location=device)
        
        if 'ema' in checkpoint:
            state_dict = checkpoint['ema']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        self.dit_model.load_state_dict(state_dict)
        
        self.dit_model.eval()
        self._requires_grad(self.dit_model, False)
        
        # Create diffusion object for noise schedule
        self.diffusion = create_diffusion(timestep_respacing="")
        
        # Get num_classes for null class (unconditional)
        self.num_classes = self.dit_model.y_embedder.num_classes
        self.has_null_class = self.dit_model.y_embedder.dropout_prob > 0
        if not self.has_null_class:
            raise ValueError(
                f"DiT model does not support CFG (dropout_prob=0). "
                f"Model must be trained with class_dropout_prob > 0 for unconditional behavior."
            )
    
    def _requires_grad(self, model, flag=True):
        """Set requires_grad flag for all parameters in a model."""
        for p in model.parameters():
            p.requires_grad = flag
    
    @torch.no_grad()
    def add_noise_pixel_space(
        self,
        images: torch.Tensor,
        timestep: int,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        images_normalized = images * 2.0 - 1.0
        
        if noise is None:
            noise = torch.randn_like(images_normalized)
        
        batch_size = images_normalized.shape[0]
        t = torch.full((batch_size,), timestep, device=self.device, dtype=torch.long)
        noised = self.diffusion.q_sample(images_normalized, t, noise=noise)
        
        return noised
    
    @torch.no_grad()
    def denoise_one_step(
        self,
        noised_images: torch.Tensor,
        timestep: int,
        class_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Denoise images one step using DiT.
        
        Args:
            noised_images: Noised images in pixel space, shape (B, 3, H, W) in range [-1, 1]
            timestep: Current diffusion timestep
            class_labels: Optional class labels. If None, uses null class (unconditional)
            
        Returns:
            Denoised images in pixel space, shape (B, 3, H, W) in range [-1, 1]
        """
        batch_size = noised_images.shape[0]
        
        latents = self.vae.encode(noised_images).latent_dist.sample()
        latents = latents * 0.18215
        
        t = torch.full((batch_size,), timestep, device=self.device, dtype=torch.long)
        if class_labels is None:
            y = torch.full((batch_size,), self.num_classes, dtype=torch.long, device=self.device)
        else:
            y = class_labels.to(self.device)
        
        p_mean_var = self.diffusion.p_mean_variance(
            model=lambda x, t, **kwargs: (self.dit_model.forward(x, t, y), None),
            x=latents,
            t=t,
            clip_denoised=False,
            model_kwargs={}
        )
        
        predicted_latent = p_mean_var["pred_xstart"]
        predicted_latent = predicted_latent / 0.18215
        denoised_images = self.vae.decode(predicted_latent).sample
        
        return denoised_images
    
    @torch.no_grad()
    def denoise(
        self,
        noised_images: torch.Tensor,
        timestep: int,
        class_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.denoise_one_step(noised_images, timestep, class_labels)
    
    @torch.no_grad()
    def add_noise_and_denoise(
        self,
        clean_images: torch.Tensor,
        timestep: int,
        class_labels: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to clean images and denoise in one call.
        
        Args:
            clean_images: Clean images in pixel space, shape (B, 3, H, W) in range [0, 1]
            timestep: Diffusion timestep to add noise at
            class_labels: Optional class labels. If None, uses null class (unconditional)
            noise: Optional noise tensor (standard normal). If None, generates random noise.
            
        Returns:
            Tuple of (noised_images, denoised_images), both in pixel space in range [-1, 1]
        """
        noised_images = self.add_noise_pixel_space(clean_images, timestep, noise=noise)
        denoised_images = self.denoise_one_step(noised_images, timestep, class_labels)
        return noised_images, denoised_images

