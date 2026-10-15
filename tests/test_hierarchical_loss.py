"""Unit tests for Asymmetric Hierarchical Cost Loss."""

import pytest
import torch
import torch.nn.functional as F

from src.losses.classification import LabelSmoothingCrossEntropy, get_loss_function
from src.losses.hierarchical import (
    DEFAULT_BENIGN_CLASSES,
    DEFAULT_CLASSES,
    DEFAULT_MALIGNANT_CLASSES,
    AsymmetricHierarchicalCostLoss,
    build_asymmetric_cost_matrix,
)


def test_build_asymmetric_cost_matrix():
    """Verify cost matrix construction according to clinical error hierarchy."""
    num_classes = len(DEFAULT_CLASSES)
    cost = build_asymmetric_cost_matrix(
        num_classes=num_classes,
        class_names=DEFAULT_CLASSES,
        c_fn=5.0,
        c_fp=1.5,
        c_within_mal=1.0,
        c_within_ben=0.5,
    )
    assert cost.shape == (num_classes, num_classes)

    mal_indices = set(i for i, c in enumerate(DEFAULT_CLASSES) if c in DEFAULT_MALIGNANT_CLASSES)
    ben_indices = set(i for i, c in enumerate(DEFAULT_CLASSES) if c in DEFAULT_BENIGN_CLASSES)

    for i in range(num_classes):
        for j in range(num_classes):
            val = cost[i, j].item()
            if i == j:
                assert val == 0.0, f"Diagonal entry C[{i},{j}] must be 0, got {val}"
            elif i in mal_indices and j in ben_indices:
                assert val == 5.0, f"FN entry C[{i},{j}] must be 5.0, got {val}"
            elif i in ben_indices and j in mal_indices:
                assert val == 1.5, f"FP entry C[{i},{j}] must be 1.5, got {val}"
            elif i in mal_indices and j in mal_indices:
                assert val == 1.0, f"Intra-mal entry C[{i},{j}] must be 1.0, got {val}"
            elif i in ben_indices and j in ben_indices:
                assert val == 0.5, f"Intra-ben entry C[{i},{j}] must be 0.5, got {val}"


def test_hierarchical_loss_alpha_zero_equivalence():
    """When alpha=0, loss must be strictly identical to base Cross-Entropy."""
    logits = torch.randn(8, 11)
    targets = torch.tensor([0, 1, 2, 7, 8, 9, 3, 4], dtype=torch.long)

    # 1. No smoothing, alpha=0 vs standard cross_entropy
    ahc_loss_zero = AsymmetricHierarchicalCostLoss(alpha=0.0, label_smoothing=0.0)
    loss_ahc = ahc_loss_zero(logits, targets)
    loss_ce = F.cross_entropy(logits, targets)
    assert torch.isclose(loss_ahc, loss_ce, atol=1e-5)

    # 2. Smoothing 0.1, alpha=0 vs LabelSmoothingCrossEntropy
    ahc_loss_smooth = AsymmetricHierarchicalCostLoss(alpha=0.0, label_smoothing=0.1)
    ls_loss = LabelSmoothingCrossEntropy(smoothing=0.1)
    assert torch.isclose(ahc_loss_smooth(logits, targets), ls_loss(logits, targets), atol=1e-5)


def test_hierarchical_loss_asymmetric_penalties():
    """Prove that Malignant -> Benign error incurs significantly higher loss than Benign -> Malignant."""
    # Class 0: Malignant (e.g., MEL), Class 1: Benign (e.g., NV)
    cost_matrix = torch.tensor([
        [0.0, 5.0],  # True Malignant (0): misclassifying as Benign (1) costs 5.0
        [1.5, 0.0],  # True Benign (1): misclassifying as Malignant (0) costs 1.5
    ])
    loss_fn = AsymmetricHierarchicalCostLoss(cost_matrix=cost_matrix, alpha=1.0, label_smoothing=0.0)

    # Consider two symmetrical errors:
    # Scenario A: True is Malignant (0), model wrongly predicts Benign logits [0.0, 3.0]
    logits_fn = torch.tensor([[0.0, 3.0]], dtype=torch.float32)
    target_mal = torch.tensor([0], dtype=torch.long)
    loss_fn_val = loss_fn(logits_fn, target_mal)

    # Scenario B: True is Benign (1), model wrongly predicts Malignant logits [3.0, 0.0]
    logits_fp = torch.tensor([[3.0, 0.0]], dtype=torch.float32)
    target_ben = torch.tensor([1], dtype=torch.long)
    loss_fp_val = loss_fn(logits_fp, target_ben)

    # Both have the same base Cross-Entropy loss!
    base_ce_fn = F.cross_entropy(logits_fn, target_mal)
    base_ce_fp = F.cross_entropy(logits_fp, target_ben)
    assert torch.isclose(base_ce_fn, base_ce_fp, atol=1e-5)

    # BUT Asymmetric Hierarchical Loss penalizes the FN (Malignant missed) far more severely!
    assert loss_fn_val.item() > loss_fp_val.item()
    expected_diff = (5.0 - 1.5) * torch.softmax(logits_fn, dim=-1)[0, 1].item()
    assert torch.isclose(loss_fn_val - loss_fp_val, torch.tensor(expected_diff), atol=1e-4)


def test_hierarchical_loss_perfect_prediction():
    """When model prediction is perfectly confident and correct, cost risk is 0."""
    cost_matrix = torch.tensor([[0.0, 5.0], [1.5, 0.0]])
    loss_fn = AsymmetricHierarchicalCostLoss(cost_matrix=cost_matrix, alpha=1.0)

    # Extreme confident correct prediction for class 0
    logits = torch.tensor([[100.0, -100.0]], dtype=torch.float32)
    targets = torch.tensor([0], dtype=torch.long)
    loss = loss_fn(logits, targets)

    # Base CE is ~0, expected cost is 0 * 1.0 + 5.0 * 0.0 = 0.0
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-4)


def test_hierarchical_loss_gradient_flow():
    """Verify gradients propagate backwards through logits to all classes."""
    loss_fn = AsymmetricHierarchicalCostLoss(num_classes=11, alpha=1.0, label_smoothing=0.1)
    logits = torch.randn(4, 11, requires_grad=True)
    targets = torch.tensor([0, 1, 7, 8], dtype=torch.long)

    loss = loss_fn(logits, targets)
    assert loss.requires_grad
    loss.backward()

    assert logits.grad is not None
    assert torch.all(torch.isfinite(logits.grad))
    assert not torch.all(logits.grad == 0.0)


def test_hierarchical_loss_reductions():
    """Verify mean, sum, and none reductions."""
    loss_none = AsymmetricHierarchicalCostLoss(num_classes=11, alpha=1.0, reduction="none")
    loss_mean = AsymmetricHierarchicalCostLoss(num_classes=11, alpha=1.0, reduction="mean")
    loss_sum = AsymmetricHierarchicalCostLoss(num_classes=11, alpha=1.0, reduction="sum")

    logits = torch.randn(6, 11)
    targets = torch.tensor([0, 1, 2, 7, 8, 10], dtype=torch.long)

    l_none = loss_none(logits, targets)
    l_mean = loss_mean(logits, targets)
    l_sum = loss_sum(logits, targets)

    assert l_none.shape == (6,)
    assert torch.isclose(l_mean, l_none.mean(), atol=1e-5)
    assert torch.isclose(l_sum, l_none.sum(), atol=1e-5)


def test_get_loss_function_hierarchical_factory():
    """Verify get_loss_function instantiates AsymmetricHierarchicalCostLoss properly."""
    loss = get_loss_function(
        loss_name="hierarchical",
        alpha=1.5,
        c_fn=6.0,
        c_fp=2.0,
        label_smoothing=0.1,
    )
    assert isinstance(loss, AsymmetricHierarchicalCostLoss)
    assert loss.alpha == 1.5
    assert loss.label_smoothing == 0.1
    # Check cost matrix entries
    cost_mat = loss.cost_matrix
    assert cost_mat[1, 8] == 6.0  # BCC (1) is mal -> NV (8) is ben: c_fn = 6.0
    assert cost_mat[8, 1] == 2.0  # NV (8) is ben -> BCC (1) is mal: c_fp = 2.0
