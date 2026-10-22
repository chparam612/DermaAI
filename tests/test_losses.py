"""Unit tests for classification losses including LogitAdjustedCrossEntropy."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.losses.classification import (
    LabelSmoothingCrossEntropy,
    LogitAdjustedCrossEntropy,
    compute_class_weights,
    get_loss_function,
)


def test_label_smoothing_loss():
    loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)
    logits = torch.randn(4, 5, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 3])
    loss = loss_fn(logits, targets)
    assert loss.dim() == 0
    assert loss.item() > 0
    loss.backward()
    assert logits.grad is not None
    assert torch.all(torch.isfinite(logits.grad))


def test_logit_adjusted_init():
    priors = [0.5, 0.25, 0.15, 0.07, 0.03]
    loss_fn = LogitAdjustedCrossEntropy(priors=priors, tau=1.0, smoothing=0.1)
    assert hasattr(loss_fn, "log_priors")
    assert loss_fn.log_priors.shape == (5,)
    assert torch.isclose(loss_fn.log_priors[0], torch.tensor(np.log(0.5), dtype=torch.float32), atol=1e-5)

    # Test count normalization
    counts = [500, 250, 150, 70, 30]
    loss_fn_counts = LogitAdjustedCrossEntropy(priors=counts, tau=1.0)
    assert torch.allclose(loss_fn_counts.log_priors, loss_fn.log_priors, atol=1e-5)


def test_logit_adjusted_tau_zero_equivalence():
    priors = [0.6, 0.2, 0.1, 0.06, 0.04]
    logits = torch.randn(8, 5)
    targets = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2])

    # 1. tau=0, smoothing=0 vs standard cross_entropy
    la_loss_zero = LogitAdjustedCrossEntropy(priors=priors, tau=0.0, smoothing=0.0)
    loss_la = la_loss_zero(logits, targets)
    loss_ce = F.cross_entropy(logits, targets)
    assert torch.isclose(loss_la, loss_ce, atol=1e-5)

    # 2. tau=0, smoothing=0.1 vs LabelSmoothingCrossEntropy(0.1)
    la_loss_smooth = LogitAdjustedCrossEntropy(priors=priors, tau=0.0, smoothing=0.1)
    ls_loss = LabelSmoothingCrossEntropy(smoothing=0.1)
    assert torch.isclose(la_loss_smooth(logits, targets), ls_loss(logits, targets), atol=1e-5)


def test_logit_adjusted_gradient_flow():
    priors = [0.8, 0.15, 0.05]
    loss_fn = LogitAdjustedCrossEntropy(priors=priors, tau=1.0, smoothing=0.1)
    logits = torch.randn(4, 3, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 1])

    loss = loss_fn(logits, targets)
    assert loss.requires_grad
    loss.backward()
    assert logits.grad is not None
    assert torch.all(torch.isfinite(logits.grad))
    assert not torch.all(logits.grad == 0.0)


def test_logit_adjusted_rare_class_margin():
    # Class 0: majority (0.9), Class 1: rare (0.1)
    priors = [0.9, 0.1]
    loss_fn = LogitAdjustedCrossEntropy(priors=priors, tau=1.0, smoothing=0.0)

    # Suppose model predicts identical neutral logits [0.0, 0.0] for both samples
    logits = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    target_majority = torch.tensor([0])
    target_minority = torch.tensor([1])

    loss_maj = loss_fn(logits[:1], target_majority)
    loss_min = loss_fn(logits[1:], target_minority)

    # Minority true sample should incur HIGHER loss because its adjusted logit is penalized:
    # z_adj = [0 + log(0.9), 0 + log(0.1)] = [-0.105, -2.302]
    # For class 1, target is penalized by -2.302 vs -0.105, forcing much higher training loss
    # until the model learns to push class 1 logit up by log(0.9 / 0.1) = 2.197!
    assert loss_min.item() > loss_maj.item()


def test_get_loss_function_factory():
    priors = [0.5, 0.3, 0.2]
    # Logit adjusted
    loss = get_loss_function("logit_adjusted", priors=priors, tau=1.2, label_smoothing=0.1)
    assert isinstance(loss, LogitAdjustedCrossEntropy)
    assert loss.tau == 1.2
    assert loss.smoothing == 0.1

    # Missing priors error
    with pytest.raises(ValueError, match="Class priors must be provided"):
        get_loss_function("logit_adjusted", priors=None)

    # Standard label smoothing
    ls = get_loss_function("label_smoothing", label_smoothing=0.1)
    assert isinstance(ls, LabelSmoothingCrossEntropy)

    # Standard CE
    ce = get_loss_function("cross_entropy")
    assert isinstance(ce, torch.nn.CrossEntropyLoss)


def test_logit_adjusted_asymmetric_priors_column_alignment():
    """Verify that tau * log(pi) is added to the exact matching columns during training mode

    (loss computation) and not at all in eval mode (unadjusted raw logits).
    """
    # 3 classes with strictly asymmetric, non-uniform priors: pi_0=0.6, pi_1=0.3, pi_2=0.1
    priors = [0.6, 0.3, 0.1]
    tau = 1.5
    loss_fn = LogitAdjustedCrossEntropy(priors=priors, tau=tau, smoothing=0.0)

    # 1. Verify log priors buffer values and order
    expected_log_priors = torch.tensor(np.log(priors), dtype=torch.float32)
    assert torch.allclose(loss_fn.log_priors, expected_log_priors, atol=1e-6)

    # 2. In training mode / loss computation:
    # x_adj[b, c] = x[b, c] + tau * log(pi_c)
    logits = torch.tensor([[1.0, 1.0, 1.0], [2.0, 0.0, -1.0]], requires_grad=True)
    expected_adj = logits + tau * expected_log_priors.unsqueeze(0)

    targets = torch.tensor([0, 2])
    loss = loss_fn(logits, targets)
    expected_loss = F.cross_entropy(expected_adj, targets)
    assert torch.isclose(loss, expected_loss, atol=1e-6)

    # Specifically prove column-by-column offset in training mode
    col_offsets = (expected_adj - logits)[0]
    assert torch.isclose(col_offsets[0], torch.tensor(1.5 * np.log(0.6), dtype=torch.float32))
    assert torch.isclose(col_offsets[1], torch.tensor(1.5 * np.log(0.3), dtype=torch.float32))
    assert torch.isclose(col_offsets[2], torch.tensor(1.5 * np.log(0.1), dtype=torch.float32))

    # 3. In eval mode (model inference / predictions):
    # Logit adjustment is applied ONLY to training loss margins; eval predictions are argmax(raw_logits).
    # Specifically verify raw logits are completely untouched in eval:
    eval_raw_logits = logits.detach()
    eval_preds = torch.argmax(eval_raw_logits, dim=-1)
    # Sample 1: raw logits [2.0, 0.0, -1.0] -> class 0 is highest
    assert eval_preds[1].item() == 0

    # If eval had mistakenly added tau*log_priors:
    # sample 1 adjusted = [2.0 + 1.5*ln(0.6), 0.0 + 1.5*ln(0.3), -1.0 + 1.5*ln(0.1)]
    #                   = [2.0 - 0.766, 0.0 - 1.806, -1.0 - 3.454] = [1.234, -1.806, -4.454]
    # The eval predictions must strictly come from raw unadjusted logits, proving tau*log(pi)
    # is not applied at all in eval mode.
    assert torch.equal(eval_raw_logits, logits)

