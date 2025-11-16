"""
Module to extract DiT intermediate features for alignment with DINOv2.
"""
import torch
import torch.nn as nn
from diffusers.models import AutoencoderKL

# Import DiT models from thirdparty (following DINOv2's pattern for external dependencies)
from dinov2.thirdparty.DiT.models import DiT_models


class DiTFeatureExtractor:
    """
    Extracts intermediate features from pretrained DiT models.
    """
    def __init__(
        self,
        dit_model_path: str,
        dit_model_name: str = "DiT-XL/2",
        image_size: int = 256,
        vae_model: str = "stabilityai/sd-vae-ft-mse",
        dit_extraction_layer: int = -1,  # -1 means last layer, 0-indexed
        dit_timestep: float = 1.0,  # timestep for DiT forward pass (1.0 = most clean)
        device: str = "cuda",
    ):
        """
        Args:
            dit_model_path: Path to pretrained DiT checkpoint
            dit_model_name: DiT model architecture name (e.g., "DiT-XL/2")
            image_size: Input image size (256 or 512)
            vae_model: VAE model name from diffusers
            dit_extraction_layer: Which DiT layer to extract features from (-1 = last layer)
            dit_timestep: Timestep for DiT forward pass (default 1.0 = most clean)
            device: Device to run on
        """
        self.device = device
        self.image_size = image_size
        self.latent_size = image_size // 8
        self.dit_extraction_layer = dit_extraction_layer
        self.dit_timestep = dit_timestep
        
        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(vae_model).to(device)
        self.vae.eval()
        requires_grad(self.vae, False)
        
        # Load DiT model
        self.dit_model = DiT_models[dit_model_name](
            input_size=self.latent_size
        ).to(device)
        
        # Load checkpoint
        checkpoint = torch.load(dit_model_path, map_location=device)
        # DiT checkpoints from train.py have structure: {"model": ..., "ema": ..., "opt": ..., "args": ...}
        # Prefer "ema" weights if available (better for inference), then "model", then direct state_dict
        if 'ema' in checkpoint:
            state_dict = checkpoint['ema']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        self.dit_model.load_state_dict(state_dict)
        
        self.dit_model.eval()
        requires_grad(self.dit_model, False)
        
        # Determine actual extraction layer
        self.num_blocks = len(self.dit_model.blocks)
        if self.dit_extraction_layer == -1:
            self.actual_extraction_layer = self.num_blocks - 1
        else:
            self.actual_extraction_layer = min(self.dit_extraction_layer, self.num_blocks - 1)

        self.num_classes = self.dit_model.y_embedder.num_classes
        if not self.dit_model.y_embedder.dropout_prob > 0:
            raise ValueError(
                f"DiT model does not support CFG (dropout_prob=0). "
                f"Model must be trained with class_dropout_prob > 0 for unconditional behavior."
            )
    
    @torch.no_grad()
    def extract_features(
        self, 
        images: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract DiT intermediate features from images using unconditional (null class) behavior.
        Always uses null class (num_classes) for CFG, matching DiT's unconditional inference.
        
        Args:
            images: Input images of shape (B, 3, H, W) in range [0, 1]
            
        Returns:
            Features of shape (B, N_patches, hidden_dim)
        """
        # Convert from [0, 1] range to [-1, 1] range for DiT VAE
        images = images * 2.0 - 1.0
        
        batch_size = images.shape[0]
        
        # Encode images to latents using VAE
        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * 0.18215  # VAE scaling factor
        
        # Prepare timestep (convert to tensor matching batch size)
        t = torch.full((batch_size,), self.dit_timestep, device=self.device, dtype=torch.float32)
        
        # Always use null class (num_classes) for unconditional CFG behavior
        # This matches DiT's classifier-free guidance where null class is at index num_classes
        y = torch.full((batch_size,), self.num_classes, dtype=torch.long, device=self.device)
        
        # Forward through DiT with intermediate feature extraction
        features = self._forward_dit_with_extraction(latents, t, y)
        
        return features
    
    def _forward_dit_with_extraction(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor, 
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through DiT with intermediate feature extraction.
        
        Args:
            x: Latents of shape (B, 4, H, W)
            t: Timesteps of shape (B,)
            y: Class labels of shape (B,)
            
        Returns:
            Extracted features of shape (B, N_patches, hidden_dim)
        """
        # Use DiT's forward method with capture_intermediates
        # We'll modify it to capture at specific layer
        # Patch embedding
        x = self.dit_model.x_embedder(x) + self.dit_model.pos_embed
        N, T, D = x.shape
        
        # Timestep and class embedding
        t_embed = self.dit_model.t_embedder(t)
        y_embed = self.dit_model.y_embedder(y, train=False)
        c = t_embed + y_embed
        
        # Forward through blocks until extraction layer
        for i, block in enumerate(self.dit_model.blocks):
            x = block(x, c)
            if i == self.actual_extraction_layer:
                # Extract features at this layer
                return x
        
        # Should not reach here if extraction_layer is valid
        return x


def requires_grad(model, flag=True):
    """Set requires_grad flag for all parameters in a model."""
    for p in model.parameters():
        p.requires_grad = flag

