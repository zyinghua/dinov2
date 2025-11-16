"""
Module to extract DINOv2 intermediate features for alignment sanity checking.
Simplified version for testing alignment correctness.
"""
import torch
import torch.nn as nn
from typing import Optional

from dinov2.models import vision_transformer as vits
from dinov2.models import build_model_from_cfg


class DINOv2FeatureExtractor:
    """
    Extracts intermediate features from pretrained DINOv2 models.
    Simplified version for testing alignment correctness.
    Uses the same model building approach as the original training code.
    """
    def __init__(
        self,
        cfg=None,
        model_path: Optional[str] = None,
        arch: str = "vit_small",
        patch_size: int = 16,
        img_size: int = 224,
        extraction_layer: int = 4,  # Layer to extract features from (0-indexed)
        device: str = "cuda",
        # Optional: match training config exactly
        layerscale: float = 1.0,
        ffn_layer: str = "mlp",
        block_chunks: int = 0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        num_register_tokens: int = 0,
        interpolate_offset: float = 0.0,
        interpolate_antialias: bool = False,
    ):

        self.device = device
        
        # Use config if provided, otherwise use individual parameters
        if cfg is not None:
            # Get model path from config (MODEL.WEIGHTS or student.pretrained_weights)
            if not model_path:
                model_path = getattr(cfg.MODEL, 'WEIGHTS', '') or getattr(cfg.student, 'pretrained_weights', '')
            if not model_path:
                raise ValueError("model_path must be provided either as argument or in cfg.MODEL.WEIGHTS")
            
            # Get extraction layer from config if available
            if hasattr(cfg, 'extraction') and hasattr(cfg.extraction, 'extraction_layer'):
                extraction_layer = cfg.extraction.extraction_layer
            # Otherwise use the function parameter default
            
            # Get model parameters from config
            arch = cfg.student.arch
            patch_size = cfg.student.patch_size
            img_size = cfg.crops.global_crops_size
            layerscale = getattr(cfg.student, 'layerscale', 1.0)
            ffn_layer = getattr(cfg.student, 'ffn_layer', 'mlp')
            block_chunks = getattr(cfg.student, 'block_chunks', 0)
            qkv_bias = getattr(cfg.student, 'qkv_bias', True)
            proj_bias = getattr(cfg.student, 'proj_bias', True)
            ffn_bias = getattr(cfg.student, 'ffn_bias', True)
            num_register_tokens = getattr(cfg.student, 'num_register_tokens', 0)
            interpolate_offset = getattr(cfg.student, 'interpolate_offset', 0.0)
            interpolate_antialias = getattr(cfg.student, 'interpolate_antialias', False)
            
            # Build model using the same approach as training code
            student_backbone, _, _ = build_model_from_cfg(cfg)
            self.model = student_backbone
        else:
            # Build model using individual parameters (legacy approach)
            if not model_path:
                raise ValueError("model_path must be provided when cfg is not provided")
            
            # Build model using the same approach as build_model() in models/__init__.py
            arch_clean = arch.removesuffix("_memeff")
            vit_kwargs = dict(
                img_size=img_size,
                patch_size=patch_size,
                init_values=layerscale,
                ffn_layer=ffn_layer,
                block_chunks=block_chunks,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                num_register_tokens=num_register_tokens,
                interpolate_offset=interpolate_offset,
                interpolate_antialias=interpolate_antialias,
            )
            
            self.model = vits.__dict__[arch_clean](**vit_kwargs)
        
        self.extraction_layer = extraction_layer
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        # Handle different checkpoint formats
        if 'teacher' in checkpoint:
            # Teacher checkpoint format from training
            state_dict = checkpoint['teacher']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Remove prefixes to match model structure (same as load_pretrained_weights in utils.py)
        # Remove `module.` prefix (for DDP-wrapped models)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # Remove `backbone.` prefix (for ModuleDict-wrapped models like in ssl_meta_arch)
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        
        # Filter out head keys (dino_head, ibot_head) since we only load the backbone
        # This allows us to use strict=True for exact matching
        state_dict = {k: v for k, v in state_dict.items() 
                     if not k.startswith("dino_head.") and not k.startswith("ibot_head.")}
        
        # Load state dict and check what was actually loaded
        model_keys = set(self.model.state_dict().keys())
        checkpoint_keys = set(state_dict.keys())
        matched_keys = model_keys & checkpoint_keys
        missing_keys = model_keys - checkpoint_keys
        unexpected_keys = checkpoint_keys - model_keys
        
        load_result = self.model.load_state_dict(state_dict, strict=True)
        
        # Log loading results for debugging
        import logging
        logger = logging.getLogger("dinov2")
        logger.info(f"DINOv2FeatureExtractor: Loading checkpoint from {model_path}")
        logger.info(f"  - Matched keys: {len(matched_keys)}/{len(model_keys)}")
        if missing_keys:
            logger.warning(f"  - Missing keys (not in checkpoint): {len(missing_keys)} keys")
            if len(missing_keys) <= 10:
                logger.warning(f"    Examples: {list(missing_keys)[:5]}")
        if unexpected_keys:
            logger.warning(f"  - Unexpected keys (in checkpoint but not in model): {len(unexpected_keys)} keys")
            if len(unexpected_keys) <= 10:
                logger.warning(f"    Examples: {list(unexpected_keys)[:5]}")
        
        # Warn if too many keys are missing (likely a loading problem)
        if len(missing_keys) > len(model_keys) * 0.1:  # More than 10% missing
            logger.error(f"  - WARNING: {len(missing_keys)}/{len(model_keys)} keys missing! Checkpoint may not have loaded correctly.")
        
        self.model.eval()
        self.model.to(device)
        
        # Disable gradients
        for p in self.model.parameters():
            p.requires_grad = False
        
        # Verify extraction layer is valid
        self.num_blocks = len(self.model.blocks)
        if self.extraction_layer >= self.num_blocks:
            raise ValueError(
                f"extraction_layer {self.extraction_layer} >= num_blocks {self.num_blocks}"
            )
    
    @torch.no_grad()
    def extract_features(
        self, 
        images: torch.Tensor,
        masks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract DINOv2 intermediate features from images.
        
        Args:
            images: Input images of shape (B, 3, H, W) in range [0, 1] or [-1, 1]
            masks: Optional masks (not used in simplified version)
            
        Returns:
            Features of shape (B, N_patches, embed_dim)
        """
        # Prepare tokens
        x = self.model.prepare_tokens_with_masks(images, masks)
        
        # Forward through blocks until extraction layer
        for i, block in enumerate(self.model.blocks):
            x = block(x)
            if i == self.extraction_layer:
                # Extract features at this layer (patch tokens only, exclude cls token)
                # x shape: (B, 1 + num_patches, embed_dim)
                # Extract patch tokens: (B, num_patches, embed_dim)
                patch_tokens = x[:, 1:]  # Remove cls token
                return patch_tokens
        
        # Should not reach here if extraction_layer is valid
        patch_tokens = x[:, 1:]  # Remove cls token
        return patch_tokens

