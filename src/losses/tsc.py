"""Targeted Supervised Contrastive Learning (TSC) Loss and Target Generation.

Paper: Li et al., "Targeted Supervised Contrastive Learning for Long-Tailed Recognition",
CVPR 2022 (arXiv:2111.13998).
Official Codebase: https://github.com/LTH14/targeted-supcon (MIT License).
"""

from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_uniform_hypersphere_targets(
    num_classes: int = 11,
    dim: int = 128,
    temperature: float = 0.07,
    num_iters: int = 2000,
    lr: float = 0.05,
    seed: int = 42,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """Generate maximally separated, uniform target vectors on an M-dimensional unit sphere.
    
    Optimizes the Gaussian potential pairwise uniformity loss from Li et al. (CVPR 2022):
        L_uniform(T) = (1 / K) * sum_i log( sum_j exp( (t_i . t_j) / temperature ) )
    subject to ||t_i||_2 = 1.
    
    Args:
        num_classes: Number of diagnostic classes (K=11 for MILK10k).
        dim: Dimension of hypersphere embedding (M=128 default).
        temperature: Bandwidth parameter for pairwise potential kernel (default 0.07).
        num_iters: Number of optimization steps (default 2000).
        lr: Learning rate for Adam optimizer (default 0.05).
        seed: Random seed for initialization reproducibility.
        device: Target compute device (defaults to CPU).
        
    Returns:
        torch.Tensor of shape (num_classes, dim), L2-normalized along dim=-1.
    """
    if device is None:
        device = torch.device("cpu")
    else:
        device = torch.device(device)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    raw_targets = torch.randn(num_classes, dim, generator=generator, device="cpu", dtype=torch.float32).to(device)
    raw_targets.requires_grad_(True)

    optimizer = torch.optim.Adam([raw_targets], lr=lr)

    for _ in range(num_iters):
        norm_targets = F.normalize(raw_targets, p=2, dim=-1)
        sim = torch.mm(norm_targets, norm_targets.t()) / temperature
        loss = torch.log(torch.exp(sim).sum(dim=-1)).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_targets = F.normalize(raw_targets.detach(), p=2, dim=-1)

    return final_targets


class TargetedSupervisedContrastiveLoss(nn.Module):
    """Targeted Supervised Contrastive (TSC) Loss.
    
    Paper: Li et al., CVPR 2022 (arXiv:2111.13998).
    
    For an anchor feature vector z_i and class label y_i:
        L_TSC(z_i, y_i) = -log [ exp( (z_i . t_{y_i}) / tau ) / sum_c exp( (z_i . t_c) / tau ) ]
    where ||z_i||_2 = 1 and ||t_c||_2 = 1.
    
    Corresponds mathematically to CrossEntropyLoss applied to scaled
    cosine similarities (z @ T.T / tau) with target labels y.
    
    Args:
        num_classes: Number of classes (default 11).
        dim: Feature embedding dimension (default 128).
        temperature: Contrastive temperature tau (default 0.1).
        targets: Pre-computed targets tensor of shape (num_classes, dim). If None,
            uniform targets are generated automatically.
        reduction: 'mean', 'sum', or 'none'.
        target_temp: Kernel temperature used for target generation (default 0.07).
        target_iters: Optimization iterations for target generation (default 2000).
        seed: Random seed for reproducible target generation.
    """
    def __init__(
        self,
        num_classes: int = 11,
        dim: int = 128,
        temperature: float = 0.1,
        targets: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        target_temp: float = 0.07,
        target_iters: int = 2000,
        seed: int = 42,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.temperature = float(temperature)
        self.reduction = reduction

        if targets is None:
            targets = generate_uniform_hypersphere_targets(
                num_classes=num_classes,
                dim=dim,
                temperature=target_temp,
                num_iters=target_iters,
                seed=seed,
            )
        else:
            if targets.shape != (num_classes, dim):
                raise ValueError(
                    f"Expected targets shape ({num_classes}, {dim}), got {tuple(targets.shape)}"
                )
            targets = F.normalize(targets.detach().clone(), p=2, dim=-1)

        # Register targets as persistent non-trainable buffer
        self.register_buffer("targets", targets)

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute TSC loss.
        
        Args:
            features: Tensor of shape (B, dim) or (B, ...).
            labels: Integer tensor of ground-truth class labels (B,).
            
        Returns:
            Scalar loss (or unreduced tensor if reduction='none').
        """
        if features.dim() > 2:
            features = features.flatten(1)

        features_norm = F.normalize(features, p=2, dim=-1)
        logits = torch.mm(features_norm, self.targets.t()) / self.temperature

        return F.cross_entropy(logits, labels, reduction=self.reduction)
