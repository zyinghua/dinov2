"""
Meta loss function for aligning DINOv2 with DiT features.
"""
import torch
import torch.nn.functional as F
from typing import Optional, List, Callable
import logging

logger = logging.getLogger("dinov2")

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
        enable_diagnostics: bool = False,
        diagnostic_freq: int = 100,
        **kwargs
    ):
        """
        Args:
            loss_type: Type of loss to use. Options: 
                "cosine_sim", "mse", "l1", "cosine_whitened", "global", "contrastive"
            enable_diagnostics: Whether to log diagnostic statistics
            diagnostic_freq: How often to log diagnostics (every N calls)
            **kwargs: Additional arguments (not used in meta loss, but kept for compatibility)
        """
        self.loss_type = loss_type
        self.enable_diagnostics = enable_diagnostics
        self.diagnostic_freq = diagnostic_freq
        self.call_count = 0
        
        # Map loss type to loss function
        self.loss_functions = {
            "cosine_sim": self._cosine_similarity_loss,
            "mse": self._mse_loss,
            "l1": self._l1_loss,
            "cosine_whitened": self._cosine_whitened_loss,
            "global": self._global_alignment_loss,
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
        self.call_count += 1
        loss_fn = self.loss_functions[self.loss_type]
        loss = loss_fn(dinov2_features, dit_features)
        
        # Log diagnostics periodically
        if self.enable_diagnostics and self.call_count % self.diagnostic_freq == 0:
            self._log_diagnostics(dinov2_features, dit_features, loss)
        
        return loss
    
    def _log_diagnostics(
        self,
        dinov2_features: List[torch.Tensor],
        dit_features: torch.Tensor,
        loss: torch.Tensor,
    ):
        """Log diagnostic statistics to detect collapse."""
        with torch.no_grad():
            # Use first feature set for diagnostics
            dinov2_feat = dinov2_features[0]  # (B, N, D)
            B, N, D = dinov2_feat.shape
            
            # 1. Variance across patches (within each image)
            dinov2_patch_std = dinov2_feat.std(dim=1).mean().item()  # Average std across patches
            dit_patch_std = dit_features.std(dim=1).mean().item()
            
            # 2. Variance across images (within each patch position)
            dinov2_image_std = dinov2_feat.std(dim=0).mean().item()
            dit_image_std = dit_features.std(dim=0).mean().item()
            
            # 3. Cosine similarity between random patches in same image
            dinov2_norm = F.normalize(dinov2_feat, dim=-1)
            dit_norm = F.normalize(dit_features, dim=-1)
            
            # Sample random patch pairs within same image
            cos_same_img_dinov2 = (dinov2_norm[0] @ dinov2_norm[0].T).mean().item()
            cos_same_img_dit = (dit_norm[0] @ dit_norm[0].T).mean().item()
            
            # 4. Cosine similarity between different images
            if B > 1:
                cos_diff_img_dinov2 = (dinov2_norm[0] @ dinov2_norm[1].T).mean().item()
                cos_diff_img_dit = (dit_norm[0] @ dit_norm[1].T).mean().item()
            else:
                cos_diff_img_dinov2 = cos_diff_img_dit = 0.0
            
            # 5. Mean cosine between aligned features (current loss metric)
            aligned_cosine = (dinov2_norm * dit_norm).sum(dim=-1).mean().item()
            
            logger.info(
                f"ALIGNMENT_DIAG -- loss: {loss.item():.4f}, "
                f"aligned_cosine: {aligned_cosine:.4f}, "
                f"dinov2_patch_std: {dinov2_patch_std:.4f}, dit_patch_std: {dit_patch_std:.4f}, "
                f"dinov2_image_std: {dinov2_image_std:.4f}, dit_image_std: {dit_image_std:.4f}, "
                f"cos_same_img_dinov2: {cos_same_img_dinov2:.4f}, cos_same_img_dit: {cos_same_img_dit:.4f}, "
                f"cos_diff_img_dinov2: {cos_diff_img_dinov2:.4f}, cos_diff_img_dit: {cos_diff_img_dit:.4f}"
            )
    
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
        Note: MSE is scale-sensitive, so we normalize features before computing loss
        to avoid being dominated by magnitude differences.
        
        Args:
            dinov2_features: List of DINOv2 feature tensors
            dit_features: DiT feature tensor
            
        Returns:
            Scalar loss value
        """
        mse_loss = 0.0
        bsz = dinov2_features[0].shape[0]
        
        for dinov2_feat in dinov2_features:
            # Ensure same dimensions
            if dinov2_feat.shape[-1] != dit_features.shape[-1]:
                raise ValueError(
                    f"Feature dimension mismatch: DINOv2 {dinov2_feat.shape[-1]} vs DiT {dit_features.shape[-1]}. "
                    "Use projector in model to match dimensions."
                )
            
            # Normalize both features before MSE (MSE is scale-sensitive)
            # This matches the pattern of normalizing in loss function
            for j in range(bsz):
                dinov2_feat_j = dinov2_feat[j]  # (N, D)
                dit_feat_j = dit_features[j]  # (N, D)
                
                # Normalize both features
                dinov2_feat_j_norm = F.normalize(dinov2_feat_j, dim=-1)  # (N, D)
                dit_feat_j_norm = F.normalize(dit_feat_j, dim=-1)  # (N, D)
                
                mse = F.mse_loss(dinov2_feat_j_norm, dit_feat_j_norm, reduction='mean')
                mse_loss += mse
        
        mse_loss = mse_loss / (len(dinov2_features) * bsz)
        return mse_loss
    
    def _l1_loss(
        self,
        dinov2_features: List[torch.Tensor],
        dit_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        L1 loss.
        Note: L1 is scale-sensitive, so we normalize features before computing loss
        to avoid being dominated by magnitude differences.
        
        Args:
            dinov2_features: List of DINOv2 feature tensors
            dit_features: DiT feature tensor
            
        Returns:
            Scalar loss value
        """
        l1_loss = 0.0
        bsz = dinov2_features[0].shape[0]
        
        for dinov2_feat in dinov2_features:
            # Ensure same dimensions
            if dinov2_feat.shape[-1] != dit_features.shape[-1]:
                raise ValueError(
                    f"Feature dimension mismatch: DINOv2 {dinov2_feat.shape[-1]} vs DiT {dit_features.shape[-1]}. "
                    "Use projector in model to match dimensions."
                )
            
            # Normalize both features before L1 (L1 is scale-sensitive)
            # This matches the pattern of normalizing in loss function
            for j in range(bsz):
                dinov2_feat_j = dinov2_feat[j]  # (N, D)
                dit_feat_j = dit_features[j]  # (N, D)
                
                # Normalize both features
                dinov2_feat_j_norm = F.normalize(dinov2_feat_j, dim=-1)  # (N, D)
                dit_feat_j_norm = F.normalize(dit_feat_j, dim=-1)  # (N, D)
                
                l1 = F.l1_loss(dinov2_feat_j_norm, dit_feat_j_norm, reduction='mean')
                l1_loss += l1
        
        l1_loss = l1_loss / (len(dinov2_features) * bsz)
        return l1_loss
    
    def _normalize_for_alignment(self, x, eps=1e-5):
        # Center over patches (remove mean direction)
        x = x - x.mean(dim=0, keepdim=True)
        # Normalize per-dimension (whiten-ish)
        std = x.std(dim=0, keepdim=True)
        x = x / (std + eps)
        # Final L2 normalize per patch
        x = F.normalize(x, dim=-1)
        return x
    
    def _cosine_whitened_loss(
        self,
        dinov2_features: List[torch.Tensor],
        dit_features: torch.Tensor,
    ) -> torch.Tensor:
        proj_loss = 0.0
        bsz = dinov2_features[0].shape[0]
        
        for dinov2_feat in dinov2_features:
            for j in range(bsz):
                dinov2_feat_j = dinov2_feat[j]  # (N, D)
                dit_feat_j = dit_features[j]  # (N, D)
                
                # Whitened normalization
                dinov2_feat_j_norm = self._normalize_for_alignment(dinov2_feat_j)
                dit_feat_j_norm = self._normalize_for_alignment(dit_feat_j)
                
                cosine_sim = (dinov2_feat_j_norm * dit_feat_j_norm).sum(dim=-1)  # (N,)
                proj_loss += mean_flat(-cosine_sim)
        
        proj_loss = proj_loss / (len(dinov2_features) * bsz)
        return proj_loss
    
    def _global_alignment_loss(
        self,
        dinov2_features: List[torch.Tensor],
        dit_features: torch.Tensor,
    ) -> torch.Tensor:
        proj_loss = 0.0
        bsz = dinov2_features[0].shape[0]
        
        for dinov2_feat in dinov2_features:
            for j in range(bsz):
                dinov2_feat_j = dinov2_feat[j]  # (N, D)
                dit_feat_j = dit_features[j]  # (N, D)
                
                # Global pooling (mean over patches)
                dinov2_global = dinov2_feat_j.mean(dim=0)  # (D,)
                dit_global = dit_feat_j.mean(dim=0)  # (D,)
                
                # Normalize
                dinov2_global = F.normalize(dinov2_global, dim=-1)
                dit_global = F.normalize(dit_global, dim=-1)
                
                # Cosine similarity (scalar)
                cosine_sim = (dinov2_global * dit_global).sum()
                proj_loss += -cosine_sim
        
        proj_loss = proj_loss / (len(dinov2_features) * bsz)
        return proj_loss
    