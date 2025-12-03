"""
Wrapper to add alignment support to DINOv2 models.
"""
import warnings
import torch
import torch.nn as nn
from typing import Optional
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
    Extracts features at specified layer and projects them to match the target model dimensions.
    Uses DINOv2's built-in get_intermediate_layers method for FSDP compatibility.
    """
    def __init__(
        self,
        base_model: DinoVisionTransformer,
        alignment_depth: int = 4,
        dit_hidden_dim: Optional[int] = 1152,  # Backward compatible default
        projector_dim: Optional[int] = 2048,  # Projector hidden dimension (same as REPA)
        target_hidden_dim: Optional[int] = None,  # Generic target hidden dimension
    ):
        """
        Args:
            base_model: Base DINOv2 model (only used to get embed_dim for projector initialization)
            alignment_depth: Which DINOv2 layer to extract features from (0-indexed)
                            -1 means no alignment
            dit_hidden_dim: DiT hidden dimension to project to (kept for backward compatibility)
            target_hidden_dim: Generic target hidden dimension override
            projector_dim: Projector MLP hidden dimension
        """
        super().__init__()
        self.alignment_depth = alignment_depth
        self.projector_dim = projector_dim
        self.target_hidden_dim = target_hidden_dim if target_hidden_dim is not None else dit_hidden_dim
        self.dit_hidden_dim = self.target_hidden_dim  # alias for older checkpoints
        
        if self.alignment_depth == -1:
            self.projector = None
        else:
            if self.target_hidden_dim is not None and projector_dim is not None:
                dinov2_embed_dim = base_model.embed_dim
                self.projector = build_mlp(
                    hidden_size=dinov2_embed_dim,
                    projector_dim=projector_dim,
                    z_dim=self.target_hidden_dim
                )
            else:
                # No projector - return raw features (for sanity check with DINOv2 teacher)
                self.projector = None
    
    def forward_features(
        self, 
        backbone: DinoVisionTransformer,
        x: torch.Tensor, 
        masks: Optional[torch.Tensor] = None,
    ):
        # Use hooks with forward_features (which FSDP handles correctly)
        # get_intermediate_layers doesn't work with FSDP, but forward_features does
        extracted_features = {}
        
        def hook_fn(module, input, output):
            extracted_features['features'] = output
        
        # Register hook on the target block - access blocks directly (FSDP handles it)
        blocks = backbone.blocks
        if self.alignment_depth >= len(blocks):
            warnings.warn(f"Alignment depth {self.alignment_depth} >= num_blocks {len(blocks)}")
            return None
        
        handle = blocks[self.alignment_depth].register_forward_hook(hook_fn)
        
        try:
            # Call backbone the same way as normal training loop (not forward_features directly)
            # This ensures FSDP state is correct
            _ = backbone([x], masks=[masks], is_training=True)
            
            if 'features' not in extracted_features:
                warnings.warn("No features extracted from alignment depth")
                return None
            
            x_intermediate = extracted_features['features']

            if isinstance(x_intermediate, list):
                x_intermediate = x_intermediate[0]
            # Extract patch tokens (exclude cls token)
            patch_tokens = x_intermediate[:, 1:]  # (B, num_patches, embed_dim)
            
            # raw_std = patch_tokens.std(dim=1).mean().item()
            # print(f"Raw DINOv2 features std (layer {self.alignment_depth}): {raw_std:.4f}")
        finally:
            handle.remove()
        
        # Project to DiT dimension if projector is enabled
        # Note: Following REPA's approach, we do NOT normalize before projection
        # Normalization happens in the loss function instead
        if self.projector is not None:
            B, N, D = patch_tokens.shape
            # Ensure dtype matches projector (FSDP might use mixed precision for backbone)
            projector_dtype = next(self.projector.parameters()).dtype
            patch_tokens = patch_tokens.to(dtype=projector_dtype)
            projected = self.projector(patch_tokens.reshape(-1, D))  # (B*N, target_hidden_dim)
            projected = projected.reshape(B, N, self.target_hidden_dim)  # (B, N, target_hidden_dim)
            alignment_features = [projected]
        else:
            # Return raw features (normalization happens in loss, matching REPA)
            alignment_features = [patch_tokens]
        
        return alignment_features


# Backward compatibility with previous name
DINOv2WithAlignment_DiT = DINOv2WithAlignment
