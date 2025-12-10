"""
Meta loss function for aligning DINOv2 with DiT features.
Supports multiple loss types including cosine similarity, MSE, L1, CKA (linear/kernel), and RKD.
"""
import torch
import torch.nn.functional as F
from typing import Optional, List, Callable, Tuple
import logging

logger = logging.getLogger("dinov2")

def mean_flat(x):
    """Take the mean over all non-batch dimensions."""
    return torch.mean(x, dim=list(range(1, len(x.size()))))


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
        temp: float = 1.0,
        use_angle: bool = True,
        use_distance: bool = True,
        **kwargs
    ):
        """
        Args:
            loss_type: Type of loss to use. Options: 
                "cosine_sim", "mse", "l1", "cosine_whitened", "global", "cka-linear", "cka-kernel", "rkd"
            enable_diagnostics: Whether to log diagnostic statistics
            diagnostic_freq: How often to log diagnostics (every N calls)
            temp: Temperature for RKD distance scaling
            use_angle: Whether to use angle matching in RKD
            use_distance: Whether to use distance matching in RKD
            **kwargs: Additional arguments (not used in meta loss, but kept for compatibility)
        """
        self.loss_type = loss_type
        self.enable_diagnostics = enable_diagnostics
        self.diagnostic_freq = diagnostic_freq
        self.call_count = 0
        
        # RKD-specific parameters
        self.rkd_temp = temp
        self.rkd_use_angle = use_angle
        self.rkd_use_distance = use_distance

        # Map loss type to loss function
        self.loss_functions = {
            "cosine_sim": self._cosine_similarity_loss,
            "mse": self._mse_loss,
            "l1": self._l1_loss,
            "cosine_whitened": self._cosine_whitened_loss,
            "global": self._global_alignment_loss,
            "cka": self._cka_linear_loss,  # Backward compatibility: "cka" -> "cka-linear"
            "cka-linear": self._cka_linear_loss,
            "cka-kernel": self._cka_kernel_loss,
            "rkd": self._rkd_loss,
        }
        
        # Handle backward compatibility
        if loss_type == "cka":
            logger.warning("loss_type='cka' is deprecated. Use 'cka-linear' or 'cka-kernel' instead. Using 'cka-linear'.")
        
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
    
    @staticmethod
    def _linear_cka(
        x: torch.Tensor,
        y: torch.Tensor,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """
        Compute linear CKA similarity between two feature matrices.

        Args:
            x: Feature matrix of shape (N, D_x)
            y: Feature matrix of shape (N, D_y)
            eps: Numerical stability epsilon

        Returns:
            Scalar tensor with CKA similarity in [0, 1]
        """
        x_centered = x - x.mean(dim=0, keepdim=True)
        y_centered = y - y.mean(dim=0, keepdim=True)

        cross_cov = x_centered.transpose(0, 1) @ y_centered  # (D_x, D_y)
        numerator = (cross_cov ** 2).sum()

        x_gram = x_centered.transpose(0, 1) @ x_centered
        y_gram = y_centered.transpose(0, 1) @ y_centered

        denom = torch.norm(x_gram, p="fro") * torch.norm(y_gram, p="fro")
        denom = torch.clamp(denom, min=eps)

        return numerator / denom

    def _cka_linear_loss(
        self,
        dinov2_features: List[torch.Tensor],
        dit_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Centered Kernel Alignment (Linear) loss.
        
        Uses linear kernel (Gram matrix). Fastest CKA variant.
        Suitable for most cases where dimensionality is similar.

        Args:
            dinov2_features: List of tensors with shape (B, N, D_i)
            dit_features: Tensor with shape (B, N, D_j)

        Returns:
            Scalar tensor with average CKA loss.
        """
        total_loss = 0.0
        bsz = dit_features.shape[0]

        for dinov2_feat in dinov2_features:
            if dinov2_feat.shape[:2] != dit_features.shape[:2]:
                raise ValueError(
                    "CKA requires matching batch size and number of patches: "
                    f"DINOv2 {dinov2_feat.shape[:2]} vs DiT {dit_features.shape[:2]}"
                )

            for j in range(bsz):
                dinov2_feat_j = dinov2_feat[j]
                dit_feat_j = dit_features[j]

                cka_value = self._linear_cka(dinov2_feat_j, dit_feat_j)
                total_loss += (1.0 - cka_value)

        total_loss = total_loss / (len(dinov2_features) * bsz)
        return total_loss
    
    def _cka_kernel_loss(
        self,
        dinov2_features: List[torch.Tensor],
        dit_features: torch.Tensor,
        kernel_type: str = "rbf",
        gamma: float = 1.0,
    ) -> torch.Tensor:
        """
        Centered Kernel Alignment (Kernel) loss.
        
        Uses non-linear kernels (RBF, polynomial, etc) for better representation matching.
        More expensive than linear CKA but can capture non-linear relationships.

        Args:
            dinov2_features: List of tensors with shape (B, N, D_i)
            dit_features: Tensor with shape (B, N, D_j)
            kernel_type: Type of kernel - "rbf", "poly", "cosine"
            gamma: Kernel parameter (sigma for RBF, degree for poly)

        Returns:
            Scalar tensor with average kernel CKA loss.
        """
        total_loss = 0.0
        bsz = dit_features.shape[0]

        for dinov2_feat in dinov2_features:
            if dinov2_feat.shape[:2] != dit_features.shape[:2]:
                raise ValueError(
                    "CKA requires matching batch size and number of patches: "
                    f"DINOv2 {dinov2_feat.shape[:2]} vs DiT {dit_features.shape[:2]}"
                )

            for j in range(bsz):
                dinov2_feat_j = dinov2_feat[j]
                dit_feat_j = dit_features[j]

                cka_value = self._kernel_cka(
                    dinov2_feat_j, dit_feat_j, 
                    kernel_type=kernel_type, gamma=gamma
                )
                total_loss += (1.0 - cka_value)

        total_loss = total_loss / (len(dinov2_features) * bsz)
        return total_loss
    
    @staticmethod
    def _kernel_cka(
        x: torch.Tensor,
        y: torch.Tensor,
        kernel_type: str = "rbf",
        gamma: float = 1.0,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """
        Compute kernel CKA similarity between two feature matrices.

        Args:
            x: Feature matrix of shape (N, D_x)
            y: Feature matrix of shape (N, D_y)
            kernel_type: "rbf" (default), "poly", or "cosine"
            gamma: Kernel hyperparameter
                - For RBF: sigma (bandwidth)
                - For polynomial: degree (int)
            eps: Numerical stability epsilon

        Returns:
            Scalar tensor with kernel CKA similarity in [0, 1]
        """
        # Compute kernel matrices
        K_x = AlignmentLoss._compute_kernel(x, x, kernel_type, gamma)
        K_y = AlignmentLoss._compute_kernel(y, y, kernel_type, gamma)
        K_xy = AlignmentLoss._compute_kernel(x, y, kernel_type, gamma)
        
        # Center kernels
        n = K_x.shape[0]
        H = torch.eye(n, device=x.device) - torch.ones(n, n, device=x.device) / n
        
        K_x_centered = H @ K_x @ H
        K_y_centered = H @ K_y @ H
        K_xy_centered = H @ K_xy @ H
        
        # Compute CKA
        numerator = (K_xy_centered ** 2).sum()
        denominator = torch.norm(K_x_centered, p="fro") * torch.norm(K_y_centered, p="fro")
        denominator = torch.clamp(denominator, min=eps)
        
        return numerator / (denominator + eps)
    
    @staticmethod
    def _compute_kernel(
        x: torch.Tensor,
        y: torch.Tensor,
        kernel_type: str = "rbf",
        gamma: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute kernel matrix between two feature sets.

        Args:
            x: Feature matrix of shape (N, D_x)
            y: Feature matrix of shape (M, D_y)
            kernel_type: "rbf", "poly", or "cosine"
            gamma: Kernel parameter

        Returns:
            Kernel matrix of shape (N, M)
        """
        if kernel_type == "rbf":
            # RBF kernel: exp(-gamma * ||x - y||^2)
            distances = torch.cdist(x, y, p=2.0) ** 2
            return torch.exp(-gamma * distances)
        
        elif kernel_type == "poly":
            # Polynomial kernel: (x·y + 1)^degree
            # gamma is degree in this case
            degree = int(gamma)
            K = x @ y.T + 1
            return K ** degree
        
        elif kernel_type == "cosine":
            # Cosine similarity kernel
            x_norm = F.normalize(x, dim=-1)
            y_norm = F.normalize(y, dim=-1)
            return x_norm @ y_norm.T
        
        else:
            raise ValueError(f"Unknown kernel_type: {kernel_type}")
    
    def _rkd_loss(
        self,
        dinov2_features: List[torch.Tensor],
        dit_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Relational Knowledge Distillation (RKD) loss.
        
        Matches relational structures (distances and angles between samples) 
        rather than absolute feature values. Particularly effective for:
        - Distilling teacher-student models
        - Domain adaptation
        - Cross-modal alignment
        
        References:
        - RKD: https://arxiv.org/abs/1904.05068
        - Applied to feature alignment: Preserves structural relationships

        Args:
            dinov2_features: List of tensors with shape (B, N, D_i)
            dit_features: Tensor with shape (B, N, D_j)

        Returns:
            Scalar tensor with average RKD loss.
        """
        total_loss = 0.0
        bsz = dinov2_features[0].shape[0]
        
        for dinov2_feat in dinov2_features:
            if dinov2_feat.shape[0] != dit_features.shape[0]:
                raise ValueError(
                    f"Batch size mismatch: DINOv2 {dinov2_feat.shape[0]} vs DiT {dit_features.shape[0]}"
                )
            
            # Compute RKD for all batch items using stored parameters
            rkd_loss_batch = self._compute_rkd(
                dinov2_feat, dit_features,
                temp=self.rkd_temp,
                use_angle=self.rkd_use_angle,
                use_distance=self.rkd_use_distance
            )
            total_loss += rkd_loss_batch
        
        total_loss = total_loss / len(dinov2_features)
        return total_loss
    
    @staticmethod
    def _compute_rkd(
        feat1: torch.Tensor,
        feat2: torch.Tensor,
        temp: float = 1.0,
        use_angle: bool = True,
        use_distance: bool = True,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """
        Compute RKD loss between two feature tensors.
        
        RKD has two components:
        1. Distance-wise loss: Matches pairwise distances
        2. Angle-wise loss: Matches angles between triplets

        Args:
            feat1: Feature tensor of shape (B, N, D1) or (N, D1)
            feat2: Feature tensor of shape (B, N, D2) or (N, D2)
            temp: Temperature for distance scaling
            use_angle: Whether to use angle loss
            use_distance: Whether to use distance loss
            eps: Numerical stability

        Returns:
            Scalar RKD loss
        """
        # Handle both batch and non-batch inputs
        if feat1.dim() == 3:
            # (B, N, D) case
            loss = 0.0
            for b in range(feat1.shape[0]):
                loss += AlignmentLoss._compute_rkd(
                    feat1[b], feat2[b],
                    temp=temp, use_angle=use_angle, use_distance=use_distance, eps=eps
                )
            return loss / feat1.shape[0]
        
        # Now handle 2D case: (N, D)
        loss = 0.0
        
        # Normalize features for fair distance computation
        feat1_norm = F.normalize(feat1, dim=-1)
        feat2_norm = F.normalize(feat2, dim=-1)
        
        if use_distance:
            # Compute pairwise distances in normalized space
            # Euclidean distance: ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x·y
            # In normalized space: = 2 - 2*x·y = 2(1 - cosine_sim)
            
            dist1 = torch.cdist(feat1_norm, feat1_norm, p=2.0)
            dist2 = torch.cdist(feat2_norm, feat2_norm, p=2.0)
            
            # Scale distances by temperature for soft matching
            dist1_scaled = dist1 / (temp + eps)
            dist2_scaled = dist2 / (temp + eps)
            
            # KL divergence between distance distributions
            # Use softmax over distances to create soft similarity matrices
            D1 = F.softmax(-dist1_scaled, dim=-1)
            D2 = F.softmax(-dist2_scaled, dim=-1)
            
            # KL(D1 || D2)
            distance_loss = F.kl_div(D2.log() + eps, D1, reduction='batchmean')
            loss += distance_loss
        
        if use_angle:
            # Angle matching: for each triplet (i, j, k), compute angle at j
            # angle = arccos((x_i - x_j) · (x_k - x_j) / (||x_i - x_j|| * ||x_k - x_j||))
            
            # To avoid O(N^3) computation, sample random triplets
            n = feat1_norm.shape[0]
            
            if n >= 3:
                # Sample 100 random triplets or all if fewer
                num_triplets = min(100, n * (n - 1) // 2)
                
                angle_loss = 0.0
                for _ in range(num_triplets):
                    # Random sample 3 distinct indices
                    indices = torch.randperm(n, device=feat1.device)[:3]
                    i, j, k = indices[0], indices[1], indices[2]
                    
                    # Compute angles for both features
                    vec1_ij = feat1_norm[i] - feat1_norm[j]
                    vec1_kj = feat1_norm[k] - feat1_norm[j]
                    vec2_ij = feat2_norm[i] - feat2_norm[j]
                    vec2_kj = feat2_norm[k] - feat2_norm[j]
                    
                    # Cosine of angle: (u · v) / (||u|| * ||v||)
                    cos_angle1 = (vec1_ij * vec1_kj).sum() / (
                        torch.norm(vec1_ij) * torch.norm(vec1_kj) + eps
                    )
                    cos_angle2 = (vec2_ij * vec2_kj).sum() / (
                        torch.norm(vec2_ij) * torch.norm(vec2_kj) + eps
                    )
                    
                    # Clamp to valid range [-1, 1] for numerical stability
                    cos_angle1 = torch.clamp(cos_angle1, -1.0, 1.0)
                    cos_angle2 = torch.clamp(cos_angle2, -1.0, 1.0)
                    
                    # L2 loss on angles
                    angle_loss += F.mse_loss(
                        torch.acos(cos_angle1), 
                        torch.acos(cos_angle2)
                    )
                
                angle_loss = angle_loss / num_triplets
                loss += angle_loss
        
        return loss
