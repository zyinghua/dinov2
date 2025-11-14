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
    """
    Preprocess images to match patch token resolution between DINOv2 and DiT.
    
    DiT produces 16x16 patches (256x256 image -> VAE -> 32x32 latent -> patch_size=2 -> 16x16).
    To match: target_size / patch_size = 16, so target_size = 16 * patch_size.
    
    For patch_size=16: target_size = 256 (no resize if input is 256)
    For patch_size=14: target_size = 224 (resize needed, like REPA)
    
    Preprocessing steps (matching REPA):
    - x = x / 255. (convert to [0, 1])
    - x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    - x = interpolate(x, target_size, mode='bicubic') if needed
    
    Args:
        images: Input images of shape (B, 3, H, W) in range [0, 255] or [0, 1] or [-1, 1]
        patch_size: DINOv2 patch size (e.g., 16 or 14)
        device: Device to process on
        
    Returns:
        Preprocessed images with matching patch count to DiT (16x16 patches)
    """
    if device is None:
        device = images.device
    
    images = images.to(device)
    
    # Get resolution (assuming square images)
    resolution = images.shape[-1]
    
    # Convert to [0, 1] range (matching REPA: x = x / 255.)
    if images.max() > 1.0:
        # Assume [0, 255] range
        images = images / 255.0
    elif images.min() < 0:
        # Assume [-1, 1] range
        images = (images + 1) / 2
    
    # Normalize with ImageNet stats (matching REPA order)
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
