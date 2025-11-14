"""
Meta loss function for aligning DINOv2 with DiT features.
"""
import torch
import torch.nn.functional as F
from typing import Optional, List, Callable


def mean_flat(x):
    """Take the mean over all non-batch dimensions."""
    return torch.mean(x, dim=list[int](range(1, len(x.size()))))


class AlignmentLoss:
    """
    Meta loss function for aligning DINOv2 features with DiT features.
    Supports multiple loss types via loss_type argument.
    """
    def __init__(
        self,
        loss_type: str = "cosine_sim",
        **kwargs
    ):
        """
        Args:
            loss_type: Type of loss to use. Options: "cosine_sim", "mse", "l1"
            **kwargs: Additional arguments (not used in meta loss, but kept for compatibility)
        """
        self.loss_type = loss_type
        
        # Map loss type to loss function
        self.loss_functions = {
            "cosine_sim": self._cosine_similarity_loss,
            "mse": self._mse_loss,
            "l1": self._l1_loss,
        }
        
        if loss_type not in self.loss_functions:
            raise ValueError(f"Unknown loss_type: {loss_type}. Available: {list(self.loss_functions.keys())}")
    
    def __call__(
        self,
        dinov2_features: List[torch.Tensor],
        dit_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute alignment loss between DINOv2 and DiT features.
        
        Args:
            dinov2_features: List of DINOv2 feature tensors, each of shape (B, N_patches, embed_dim)
            dit_features: DiT feature tensor of shape (B, N_patches, hidden_dim)
            
        Returns:
            Scalar loss value
        """
        loss_fn = self.loss_functions[self.loss_type]
        return loss_fn(dinov2_features, dit_features)
    
    def _cosine_similarity_loss(
        self,
        dinov2_features: List[torch.Tensor],
        dit_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cosine similarity loss (same as REPA's projection loss).
        
        Args:
            dinov2_features: List of DINOv2 feature tensors
            dit_features: DiT feature tensor
            
        Returns:
            Scalar loss value
        """
        proj_loss = 0.0
        bsz = dinov2_features[0].shape[0]
        
        # Match REPA's nested loop structure: iterate over feature sets, then batch items
        for dinov2_feat in dinov2_features:
            # Iterate over batch dimension (matching REPA's inner loop)
            for j in range(bsz):
                dinov2_feat_j = dinov2_feat[j]  # (N, D) - features for batch item j
                dit_feat_j = dit_features[j]  # (N, D) - DiT features for batch item j
                
                # Normalize both features
                dinov2_feat_j_norm = F.normalize(dinov2_feat_j, dim=-1)  # (N, D)
                dit_feat_j_norm = F.normalize(dit_feat_j, dim=-1)  # (N, D)
                
                # Compute negative cosine similarity per patch
                # (z_j * z_tilde_j).sum(dim=-1) gives cosine similarity per patch
                # We negate it because we want to maximize similarity (minimize negative similarity)
                cosine_sim = (dinov2_feat_j_norm * dit_feat_j_norm).sum(dim=-1)  # (N,)
                proj_loss += mean_flat(-cosine_sim)  # Average over patches for this batch item
        
        # Average over number of DINOv2 feature sets and batch size (matching REPA)
        proj_loss = proj_loss / (len(dinov2_features) * bsz)
        
        return proj_loss
    
    def _mse_loss(
        self,
        dinov2_features: List[torch.Tensor],
        dit_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mean squared error loss.
        
        Args:
            dinov2_features: List of DINOv2 feature tensors
            dit_features: DiT feature tensor
            
        Returns:
            Scalar loss value
        """
        mse_loss = 0.0
        
        for dinov2_feat in dinov2_features:
            # Ensure same dimensions (might need projection if dims don't match)
            if dinov2_feat.shape[-1] != dit_features.shape[-1]:
                # If dimensions don't match, we can't directly compute MSE
                # This would require a projection layer, which should be handled in the model
                raise ValueError(
                    f"Feature dimension mismatch: DINOv2 {dinov2_feat.shape[-1]} vs DiT {dit_features.shape[-1]}. "
                    "Use projector in model to match dimensions."
                )
            
            mse = F.mse_loss(dinov2_feat, dit_features, reduction='mean')
            mse_loss += mse
        
        mse_loss = mse_loss / len(dinov2_features)
        return mse_loss
    
    def _l1_loss(
        self,
        dinov2_features: List[torch.Tensor],
        dit_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        L1 loss.
        
        Args:
            dinov2_features: List of DINOv2 feature tensors
            dit_features: DiT feature tensor
            
        Returns:
            Scalar loss value
        """
        l1_loss = 0.0
        
        for dinov2_feat in dinov2_features:
            # Ensure same dimensions
            if dinov2_feat.shape[-1] != dit_features.shape[-1]:
                raise ValueError(
                    f"Feature dimension mismatch: DINOv2 {dinov2_feat.shape[-1]} vs DiT {dit_features.shape[-1]}. "
                    "Use projector in model to match dimensions."
                )
            
            l1 = F.l1_loss(dinov2_feat, dit_features, reduction='mean')
            l1_loss += l1
        
        l1_loss = l1_loss / len(dinov2_features)
        return l1_loss

