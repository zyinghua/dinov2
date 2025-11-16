"""
Utilities for aligning DINOv2 and DiT features, including patch token matching.
"""
import torch
import torch.nn.functional as F
from torchvision.transforms import Normalize
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from typing import Optional


def preprocess_for_alignment(
    images: torch.Tensor,
    patch_size: int,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """Preprocess images for DINOv2 alignment features."""

    if device is None:
        device = images.device
    
    images = images.to(device)
    
    # Get resolution (assuming square images)
    resolution = images.shape[-1]
    
    if images.max() > 1.0:
        # Assume [0, 255] range
        images = images / 255.0
    elif images.min() < 0:
        # Assume [-1, 1] range
        images = (images + 1) / 2
    
    normalize = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    images = normalize(images)
    
    # Compute target size to match DiT's 16x16 patches
    # DiT: 256 -> 32x32 latent -> patch_size=2 -> 16x16 patches
    # We need: target_size / patch_size = 16, so target_size = 16 * patch_size
    target_size = 16 * patch_size
    
    # Only resize if current resolution doesn't match target
    if resolution != target_size:
        images = F.interpolate(
            images, 
            size=(target_size, target_size), 
            mode='bicubic'
        )
    
    return images
