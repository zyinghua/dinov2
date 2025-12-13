# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from enum import Enum
import logging
from typing import Any, Dict, Optional, Union
import numpy as np
import torch
from torch import Tensor
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.utilities.data import dim_zero_cat, select_topk


logger = logging.getLogger("dinov2")


class MetricType(Enum):
    MEAN_ACCURACY = "mean_accuracy"
    MEAN_PER_CLASS_ACCURACY = "mean_per_class_accuracy"
    PER_CLASS_ACCURACY = "per_class_accuracy"
    IMAGENET_REAL_ACCURACY = "imagenet_real_accuracy"

    @property
    def accuracy_averaging(self):
        return getattr(AccuracyAveraging, self.name, None)

    def __str__(self):
        return self.value


class AccuracyAveraging(Enum):
    MEAN_ACCURACY = "micro"
    MEAN_PER_CLASS_ACCURACY = "macro"
    PER_CLASS_ACCURACY = "none"

    def __str__(self):
        return self.value


def build_metric(metric_type: MetricType, *, num_classes: int, ks: Optional[tuple] = None):
    if metric_type.accuracy_averaging is not None:
        return build_topk_accuracy_metric(
            average_type=metric_type.accuracy_averaging,
            num_classes=num_classes,
            ks=(1, 5) if ks is None else ks,
        )
    elif metric_type == MetricType.IMAGENET_REAL_ACCURACY:
        return build_topk_imagenet_real_accuracy_metric(
            num_classes=num_classes,
            ks=(1, 5) if ks is None else ks,
        )

    raise ValueError(f"Unknown metric type {metric_type}")


def build_topk_accuracy_metric(average_type: AccuracyAveraging, num_classes: int, ks: tuple = (1, 5)):
    metrics: Dict[str, Metric] = {
        f"top-{k}": MulticlassAccuracy(top_k=k, num_classes=int(num_classes), average=average_type.value) for k in ks
    }
    return MetricCollection(metrics)


def build_topk_imagenet_real_accuracy_metric(num_classes: int, ks: tuple = (1, 5)):
    metrics: Dict[str, Metric] = {f"top-{k}": ImageNetReaLAccuracy(top_k=k, num_classes=int(num_classes)) for k in ks}
    return MetricCollection(metrics)


class ImageNetReaLAccuracy(Metric):
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def __init__(
        self,
        num_classes: int,
        top_k: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.top_k = top_k
        self.add_state("tp", [], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        # preds [B, D]
        # target [B, A]
        # preds_oh [B, D] with 0 and 1
        # select top K highest probabilities, use one hot representation
        preds_oh = select_topk(preds, self.top_k)
        # target_oh [B, D + 1] with 0 and 1
        target_oh = torch.zeros((preds_oh.shape[0], preds_oh.shape[1] + 1), device=target.device, dtype=torch.int32)
        target = target.long()
        # for undefined targets (-1) use a fake value `num_classes`
        target[target == -1] = self.num_classes
        # fill targets, use one hot representation
        target_oh.scatter_(1, target, 1)
        # target_oh [B, D] (remove the fake target at index `num_classes`)
        target_oh = target_oh[:, :-1]
        # tp [B] with 0 and 1
        tp = (preds_oh * target_oh == 1).sum(dim=1)
        # at least one match between prediction and target
        tp.clip_(max=1)
        # ignore instances where no targets are defined
        mask = target_oh.sum(dim=1) > 0
        tp = tp[mask]
        self.tp.append(tp)  # type: ignore

    def compute(self) -> Tensor:
        tp = dim_zero_cat(self.tp)  # type: ignore
        return tp.float().mean()


def rankme(embeddings: Union[torch.Tensor, np.ndarray]) -> float:
    """
    RankMe: Effective rank metric based on SVD entropy.
    
    Measures the effective number of learned dimensions in the embedding space.
    Stable estimation requires: num_samples >> embedding_dimension.
    
    Args:
        embeddings: Tensor of shape (N, D) where N is num_samples, D is embedding_dim.
                   N should be >= D for stable estimation .
    
    Returns:
        rankme_score (float): Exponential of normalized entropy.
                             Range: [1, embedding_dim]
                             Higher is better (more diverse, full-rank representations).
                             Value of D means perfectly rank-full embedding space.
        """
    with torch.no_grad():
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.tensor(embeddings, dtype=torch.float32)
        else:
            embeddings = embeddings.float()
        
        # Ensure 2D
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
        
        # Center the embeddings
        embeddings = embeddings - embeddings.mean(dim=0, keepdim=True)
        
        # Compute singular values
        s = torch.linalg.svdvals(embeddings)
        
        # Normalize to probability distribution (L2 norm for SVD)
        s = s[s > 1e-10]  # Remove near-zero singular values
        p = s / (torch.sum(s) + 1e-10)
        
        # Compute entropy
        epsilon = 1e-10
        entropy = -torch.sum(p * torch.log(p + epsilon))
        
        # RankMe score: exponential of entropy
        rankme_score = torch.exp(entropy).item()
    
    return rankme_score