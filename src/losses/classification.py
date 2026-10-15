"""Loss functions for dermatology diagnostic classification."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing, matching PanDerm official implementation."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        assert 0.0 <= smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate loss.

        Args:
            x: Logits tensor of shape (batch_size, num_classes).
            target: Ground truth indices of shape (batch_size).
        """
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class LogitAdjustedCrossEntropy(nn.Module):
    """Logit-Adjusted Cross-Entropy Loss (Menon et al., NeurIPS 2020).

    Enforces large margins on rare classes during training:
        loss = -log [ exp(x_y + tau * log pi_y) / sum_k exp(x_k + tau * log pi_k) ]

    Optionally combined with label smoothing epsilon:
        loss = (1 - eps) * NLL(x_adj, y) + eps * SmoothLoss(x_adj)

    Args:
        priors: Tensor, array, or sequence of training class priors pi_k or class counts N_k.
        tau: Adjustment strength hyperparameter (default: 1.0).
        smoothing: Label smoothing factor in [0.0, 1.0).
        weight: Optional class weights tensor.
    """

    def __init__(
        self,
        priors: Union[torch.Tensor, np.ndarray, List[float]],
        tau: float = 1.0,
        smoothing: float = 0.0,
        weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        assert tau >= 0.0, f"tau must be non-negative, got {tau}"
        assert 0.0 <= smoothing < 1.0, f"smoothing must be in [0, 1), got {smoothing}"
        self.tau = float(tau)
        self.smoothing = float(smoothing)
        self.confidence = 1.0 - self.smoothing

        if not isinstance(priors, torch.Tensor):
            priors_tensor = torch.as_tensor(priors, dtype=torch.float32)
        else:
            priors_tensor = priors.float()

        if priors_tensor.sum() > 1.05 or priors_tensor.sum() < 0.95:
            priors_tensor = priors_tensor / priors_tensor.sum()

        eps = 1e-12
        safe_priors = torch.clamp(priors_tensor, min=eps)
        log_priors = torch.log(safe_priors)
        self.register_buffer("log_priors", log_priors)

        if weight is not None:
            if not isinstance(weight, torch.Tensor):
                weight = torch.as_tensor(weight, dtype=torch.float32)
            else:
                weight = weight.float()
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate loss.

        Args:
            x: Logits tensor of shape (batch_size, num_classes).
            target: Ground truth indices of shape (batch_size).
        """
        x_adj = x + self.tau * self.log_priors.unsqueeze(0)

        if self.smoothing > 0.0:
            logprobs = F.log_softmax(x_adj, dim=-1)
            nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
            smooth_loss = -logprobs.mean(dim=-1)
            loss = self.confidence * nll_loss + self.smoothing * smooth_loss
            if self.weight is not None:
                loss = loss * self.weight[target]
            return loss.mean()
        else:
            return F.cross_entropy(x_adj, target, weight=self.weight)


def compute_class_weights(
    targets_or_counts: Union[List[int], np.ndarray, Dict[int, int]],
    num_classes: int,
    norm: bool = True,
) -> torch.Tensor:
    """Compute inverse class frequency weights: w_c = N / (C * N_c).

    Args:
        targets_or_counts: List/array of integer labels, or dict mapping class_idx -> count.
        num_classes: Total number of classes.
        norm: Whether to normalize weights so their mean is 1.0.

    Returns:
        torch.FloatTensor of shape (num_classes,).
    """
    if isinstance(targets_or_counts, dict):
        counts = np.array([targets_or_counts.get(i, 0) for i in range(num_classes)], dtype=np.float32)
    elif isinstance(targets_or_counts, (list, np.ndarray, torch.Tensor)):
        arr = np.asarray(targets_or_counts)
        counts = np.bincount(arr.astype(int), minlength=num_classes).astype(np.float32)
    else:
        raise TypeError(f"Unsupported type for targets_or_counts: {type(targets_or_counts)}")

    total_samples = float(np.sum(counts))
    weights = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        if counts[c] > 0:
            weights[c] = total_samples / (num_classes * counts[c])
        else:
            weights[c] = 1.0

    if norm:
        weights = weights / np.mean(weights)

    return torch.from_numpy(weights).float()


def get_loss_function(
    loss_name: str = "cross_entropy",
    label_smoothing: float = 0.0,
    weight: Optional[torch.Tensor] = None,
    priors: Optional[Union[torch.Tensor, np.ndarray, List[float]]] = None,
    tau: float = 1.0,
    cost_matrix: Optional[Union[torch.Tensor, np.ndarray]] = None,
    alpha: float = 1.0,
    c_fn: float = 5.0,
    c_fp: float = 1.5,
    c_within_mal: float = 1.0,
    c_within_ben: float = 0.5,
    num_classes: int = 11,
    malignant_indices: Optional[Sequence[int]] = None,
) -> nn.Module:
    """Build classification loss function.

    Args:
        loss_name: 'cross_entropy', 'ce', 'class_balanced_ce', 'weighted_ce',
                   'label_smoothing', 'ls_ce', 'logit_adjusted' / 'la_ce',
                   or 'hierarchical' / 'asymmetric_hierarchical'.
        label_smoothing: Smoothing factor epsilon in [0, 1).
        weight: Optional class weight tensor.
        priors: Class frequency priors (required for logit_adjusted loss).
        tau: Adjustment factor for logit adjustment.
        cost_matrix: Optional custom (K, K) cost matrix for hierarchical loss.
        alpha: Weight multiplier for expected hierarchical cost.
        c_fn: Cost of false negative error (malignant -> benign).
        c_fp: Cost of false positive error (benign -> malignant).
        c_within_mal: Cost of error within malignant superclass.
        c_within_ben: Cost of error within benign superclass.
        num_classes: Total number of classes.
        malignant_indices: Sequence of malignant class indices.

    Returns:
        PyTorch nn.Module.
    """
    name_clean = loss_name.strip().lower()
    if name_clean in ("logit_adjusted", "logit_adjustment", "la", "la_ce"):
        if priors is None:
            raise ValueError("Class priors must be provided for logit_adjusted loss")
        return LogitAdjustedCrossEntropy(
            priors=priors,
            tau=tau,
            smoothing=label_smoothing,
            weight=weight,
        )

    if name_clean in ("hierarchical", "hierarchical_cost", "asymmetric_hierarchical", "ahc"):
        from src.losses.hierarchical import AsymmetricHierarchicalCostLoss
        return AsymmetricHierarchicalCostLoss(
            cost_matrix=cost_matrix,
            num_classes=num_classes,
            alpha=alpha,
            c_fn=c_fn,
            c_fp=c_fp,
            c_within_mal=c_within_mal,
            c_within_ben=c_within_ben,
            malignant_indices=malignant_indices,
            label_smoothing=label_smoothing,
            weight=weight,
        )

    if name_clean in ("label_smoothing", "ls_ce") or (label_smoothing > 0 and name_clean not in ("class_balanced_ce", "weighted_ce")):
        smoothing = label_smoothing if label_smoothing > 0 else 0.1
        return LabelSmoothingCrossEntropy(smoothing=smoothing)

    return nn.CrossEntropyLoss(weight=weight)


build_loss = get_loss_function
