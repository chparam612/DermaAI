"""Unit tests for Targeted Supervised Contrastive Learning (TSC) loss and target generation.

Paper: Li et al., CVPR 2022 (arXiv:2111.13998).
"""

import pytest
import torch
import torch.nn.functional as F

from src.losses.tsc import (
    TargetedSupervisedContrastiveLoss,
    generate_uniform_hypersphere_targets,
)


def test_target_generation_shape_and_norm():
    """Verify generated targets have correct shape and unit L2 norms."""
    num_classes = 11
    dim = 128
    targets = generate_uniform_hypersphere_targets(
        num_classes=num_classes,
        dim=dim,
        temperature=0.07,
        num_iters=500,
        seed=42,
    )
    assert targets.shape == (num_classes, dim)
    assert targets.dtype == torch.float32

    # Verify unit L2 norm
    norms = torch.norm(targets, p=2, dim=-1)
    torch.testing.assert_close(norms, torch.ones(num_classes), rtol=1e-5, atol=1e-5)


def test_target_generation_spherical_uniformity():
    """Verify that generated targets form a near-regular simplex on the hypersphere.
    
    For K=11 classes in M=128 dimensions (M >> K), the pairwise dot product between
    any two distinct targets should approach -1/(K-1) = -1/10 = -0.10.
    """
    num_classes = 11
    dim = 128
    targets = generate_uniform_hypersphere_targets(
        num_classes=num_classes,
        dim=dim,
        temperature=0.07,
        num_iters=1000,
        lr=0.05,
        seed=42,
    )

    # Gram matrix of pairwise similarities
    gram = torch.mm(targets, targets.t())

    # Diagonal must be 1.0
    diag = torch.diag(gram)
    torch.testing.assert_close(diag, torch.ones(num_classes), rtol=1e-5, atol=1e-5)

    # Off-diagonal elements
    off_diag = gram[~torch.eye(num_classes, dtype=bool)]

    # Expected value for regular simplex in high dimensions is -1/(K-1) = -0.10
    mean_off_diag = off_diag.mean().item()
    std_off_diag = off_diag.std().item()

    assert -0.12 <= mean_off_diag <= -0.08, f"Mean off-diagonal similarity {mean_off_diag} deviates from -0.10"
    assert std_off_diag < 0.02, f"Off-diagonal standard deviation {std_off_diag} too high (uneven spacing)"
    assert off_diag.max().item() < 0.0, f"Max off-diagonal similarity {off_diag.max().item()} should be negative"
    assert off_diag.min().item() > -0.20, f"Min off-diagonal similarity {off_diag.min().item()} too negative"


def test_target_generation_reproducibility():
    """Verify exact reproducibility when fixed seed is provided."""
    t1 = generate_uniform_hypersphere_targets(num_classes=11, dim=128, num_iters=200, seed=123)
    t2 = generate_uniform_hypersphere_targets(num_classes=11, dim=128, num_iters=200, seed=123)
    torch.testing.assert_close(t1, t2)


def test_tsc_loss_forward_shapes():
    """Verify TSC loss output shapes for different reduction modes."""
    batch_size = 16
    dim = 128
    num_classes = 11

    loss_fn_mean = TargetedSupervisedContrastiveLoss(
        num_classes=num_classes, dim=dim, temperature=0.1, target_iters=200, reduction="mean"
    )
    loss_fn_sum = TargetedSupervisedContrastiveLoss(
        num_classes=num_classes, dim=dim, temperature=0.1, target_iters=200, reduction="sum"
    )
    loss_fn_none = TargetedSupervisedContrastiveLoss(
        num_classes=num_classes, dim=dim, temperature=0.1, target_iters=200, reduction="none"
    )

    features = torch.randn(batch_size, dim)
    labels = torch.randint(0, num_classes, (batch_size,))

    loss_mean = loss_fn_mean(features, labels)
    assert loss_mean.ndim == 0
    assert loss_mean.item() > 0.0

    loss_sum = loss_fn_sum(features, labels)
    assert loss_sum.ndim == 0
    torch.testing.assert_close(loss_sum, loss_mean * batch_size)

    loss_none = loss_fn_none(features, labels)
    assert loss_none.shape == (batch_size,)
    torch.testing.assert_close(loss_none.mean(), loss_mean)


def test_tsc_loss_gradient_flow():
    """Verify that gradients propagate back to input feature representations."""
    batch_size = 8
    dim = 128
    num_classes = 11

    loss_fn = TargetedSupervisedContrastiveLoss(
        num_classes=num_classes, dim=dim, temperature=0.1, target_iters=200
    )

    features = torch.randn(batch_size, dim, requires_grad=True)
    labels = torch.randint(0, num_classes, (batch_size,))

    loss = loss_fn(features, labels)
    loss.backward()

    assert features.grad is not None
    assert features.grad.shape == (batch_size, dim)
    assert not torch.isnan(features.grad).any()
    assert not torch.isinf(features.grad).any()
    assert (features.grad.abs().sum() > 0.0).item()

    # Targets buffer should NOT have gradients (frozen target anchors)
    assert loss_fn.targets.grad is None


def test_tsc_loss_with_custom_targets():
    """Verify that user-provided precomputed targets are accepted and validated."""
    num_classes = 5
    dim = 64
    custom_targets = F.normalize(torch.randn(num_classes, dim), p=2, dim=-1)

    loss_fn = TargetedSupervisedContrastiveLoss(
        num_classes=num_classes,
        dim=dim,
        targets=custom_targets,
    )
    torch.testing.assert_close(loss_fn.targets, custom_targets)

    # Invalid target shape should raise ValueError
    with pytest.raises(ValueError, match="Expected targets shape"):
        TargetedSupervisedContrastiveLoss(
            num_classes=num_classes,
            dim=dim,
            targets=torch.randn(num_classes, dim + 10),
        )


def test_tsc_loss_aligns_with_targets():
    """Verify that features matching target exactly achieve minimal loss."""
    num_classes = 4
    dim = 32
    loss_fn = TargetedSupervisedContrastiveLoss(
        num_classes=num_classes, dim=dim, temperature=0.1, target_iters=200
    )

    # Features set to exact matching target vectors
    perfect_features = loss_fn.targets.clone()
    perfect_labels = torch.arange(num_classes)
    loss_perfect = loss_fn(perfect_features, perfect_labels)

    # Features set to OPPOSITE (wrong) target vectors
    wrong_labels = (torch.arange(num_classes) + 1) % num_classes
    loss_wrong = loss_fn(perfect_features, wrong_labels)

    assert loss_perfect.item() < loss_wrong.item()
