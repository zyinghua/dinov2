"""
Wrapper to add alignment support to DINOv2 models.
"""
import torch
import torch.nn as nn
from typing import Optional, List, Tuple
from dinov2.models.vision_transformer import DinoVisionTransformer


def build_mlp(hidden_size, projector_dim, z_dim):
    """
    Build MLP projector (same as REPA).
    
    Args:
        hidden_size: Input dimension (DINOv2 embed_dim)
        projector_dim: Hidden dimension of projector
        z_dim: Output dimension (DiT hidden_dim)
    
    Returns:
        MLP projector module
    """
    return nn.Sequential(
        nn.Linear(hidden_size, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, z_dim),
    )


class DINOv2WithAlignment(nn.Module):
    """
    Wrapper around DINOv2 model that adds alignment support.
    Extracts features at specified layer and projects them to match DiT dimensions.
    """
    def __init__(
        self,
        base_model: DinoVisionTransformer,
        alignment_depth: int = 4,
        dit_hidden_dim: int = 1152,  # DiT-XL hidden dimension
        projector_dim: int = 2048,  # Projector hidden dimension (same as REPA)
    ):
        """
        Args:
            base_model: Base DINOv2 model
            alignment_depth: Which DINOv2 layer to extract features from (0-indexed)
                            -1 means no alignment
            dit_hidden_dim: DiT hidden dimension to project to
            projector_dim: Projector MLP hidden dimension
        """
        super().__init__()
        self.base_model = base_model
        self.alignment_depth = alignment_depth
        self.dit_hidden_dim = dit_hidden_dim
        self.projector_dim = projector_dim
        
        # Determine actual extraction layer
        self.num_blocks = len(self.base_model.blocks)
        if self.alignment_depth == -1:
            self.actual_extraction_layer = None
            self.projector = None
        else:
            if self.alignment_depth >= self.num_blocks:
                raise ValueError(
                    f"alignment_depth {self.alignment_depth} >= num_blocks {self.num_blocks}"
                )
            self.actual_extraction_layer = self.alignment_depth
            
            # Create projector to map DINOv2 features to DiT dimension
            dinov2_embed_dim = self.base_model.embed_dim
            self.projector = build_mlp(
                hidden_size=dinov2_embed_dim,
                projector_dim=projector_dim,
                z_dim=dit_hidden_dim
            )
    
    def forward(
        self, 
        x, 
        masks=None,
        is_training=False,
        return_alignment_features: bool = False
    ):
        """
        Forward pass with optional alignment feature extraction.
        Supports both single tensor and list inputs (for training with crops).
        
        Args:
            x: Input images of shape (B, 3, H, W) or list of tensors
            masks: Optional masks or list of masks
            is_training: Whether in training mode
            return_alignment_features: Whether to return alignment features
            
        Returns:
            If return_alignment_features=False:
                - Standard DINOv2 output
            If return_alignment_features=True:
                - Standard DINOv2 output
                - List of projected alignment features (one element if alignment enabled)
        """
        # Handle list inputs (for training with multiple crops)
        if isinstance(x, list):
            return self.forward_features_list(x, masks, return_alignment_features)
        
        # Single tensor input
        return self.forward_features(x, masks, return_alignment_features)
    
    def forward_features(
        self, 
        x: torch.Tensor, 
        masks: Optional[torch.Tensor] = None,
        return_alignment_features: bool = False
    ):
        """
        Forward features for single tensor input.
        When return_alignment_features=False, delegates to base model for exact same behavior.
        """
        # If not extracting alignment features, delegate to base model to keep original pipeline intact
        if not return_alignment_features or self.alignment_depth == -1:
            return self.base_model.forward_features(x, masks)
        
        # Prepare tokens
        x = self.base_model.prepare_tokens_with_masks(x, masks)
        
        # Forward through blocks until extraction layer
        alignment_features = None
        for i, blk in enumerate(self.base_model.blocks):
            x = blk(x)
            if i == self.actual_extraction_layer:
                # Extract features at this layer (patch tokens only, exclude cls token)
                # x shape: (B, 1 + num_patches, embed_dim)
                # Extract patch tokens: (B, num_patches, embed_dim)
                patch_tokens = x[:, 1:]  # Remove cls token
                
                # Project to DiT dimension
                B, N, D = patch_tokens.shape
                projected = self.projector(patch_tokens.reshape(-1, D))  # (B*N, dit_hidden_dim)
                projected = projected.reshape(B, N, self.dit_hidden_dim)  # (B, N, dit_hidden_dim)
                
                alignment_features = [projected]
                break
        
        # Continue forward through remaining blocks if needed
        if self.actual_extraction_layer < self.num_blocks - 1:
            for i in range(self.actual_extraction_layer + 1, self.num_blocks):
                x = self.base_model.blocks[i](x)
        
        # Apply final norm
        x_norm = self.base_model.norm(x)
        
        # Standard DINOv2 output format
        output = {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.base_model.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.base_model.num_register_tokens + 1 :],
            "x_prenorm": x,
            "masks": masks,
        }
        
        return output, alignment_features


    
    def forward_features_list(
        self, 
        x_list: List[torch.Tensor], 
        masks_list: Optional[List[torch.Tensor]] = None,
        return_alignment_features: bool = False
    ):
        """
        Forward features for list inputs (used in training with multiple crops).
        For alignment, we only extract from the first element (global crops).
        When return_alignment_features=False, delegates to base model for exact same behavior.
        """
        # If not extracting alignment features, delegate to base model to keep original pipeline intact
        if not return_alignment_features or self.alignment_depth == -1:
            return self.base_model.forward_features_list(x_list, masks_list)
        
        # Extract alignment features from first crop only
        alignment_features = None
        x_first = self.base_model.prepare_tokens_with_masks(
            x_list[0], 
            masks_list[0] if masks_list else None
        )
        
        for i, blk in enumerate(self.base_model.blocks):
            x_first = blk(x_first)
            if i == self.actual_extraction_layer:
                patch_tokens = x_first[:, 1:]  # Remove cls token
                B, N, D = patch_tokens.shape
                projected = self.projector(patch_tokens.reshape(-1, D))
                projected = projected.reshape(B, N, self.dit_hidden_dim)
                alignment_features = [projected]
                break
        
        # Standard forward for all crops (delegate to base model to ensure exact same behavior)
        output = self.base_model.forward_features_list(x_list, masks_list)
        
        return output, alignment_features