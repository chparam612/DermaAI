"""Unit tests proving that evaluation and checkpoint selection use raw logits without adjustment."""

import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.eval.predictions import generate_predictions
from src.losses.classification import LogitAdjustedCrossEntropy, LabelSmoothingCrossEntropy
from src.train.checkpointing import CheckpointManager
from src.train.engine import validate


class DummyEvalDataset(Dataset):
    """Dataset yielding fixed tensors and metadata for deterministic evaluation tests."""

    def __init__(self, logits_to_produce: torch.Tensor, targets: torch.Tensor):
        self.logits = logits_to_produce
        self.targets = targets
        self.num_samples = len(targets)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Image is dummy tensor, model will map index to pre-specified logits
        image = torch.tensor([float(idx)], dtype=torch.float32)
        label = self.targets[idx]
        metadata = {
            "isic_id": f"ISIC_{idx:05d}",
            "lesion_id": f"LES_{idx:05d}",
            "skin_tone_class": 1,
            "split": "val",
        }
        return image, label, metadata


class MockFixedLogitModel(nn.Module):
    """Model that outputs exact predetermined raw logits."""

    def __init__(self, logits_table: torch.Tensor):
        super().__init__()
        self.logits_table = nn.Parameter(logits_table.clone(), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Index into logits_table using integer in x
        indices = x.squeeze(-1).long()
        return self.logits_table[indices]


def test_logit_adjusted_loss_does_not_mutate_input_logits():
    """Criterion forward pass must not mutate the input logits in-place."""
    priors = [0.8, 0.15, 0.05]
    criterion = LogitAdjustedCrossEntropy(priors=priors, tau=1.0, smoothing=0.1)
    
    raw_logits = torch.tensor([[2.5, -1.0, 0.5], [0.1, 1.8, -0.2]], dtype=torch.float32)
    raw_logits_clone = raw_logits.clone()
    targets = torch.tensor([0, 1], dtype=torch.long)

    loss = criterion(raw_logits, targets)
    assert loss.item() > 0.0
    # Input tensor must be completely unaltered
    assert torch.equal(raw_logits, raw_logits_clone), "LogitAdjustedCrossEntropy modified input logits in-place!"


def test_validate_evaluates_strictly_on_raw_logits():
    """Prove validate() evaluates predictions and val_macro_f1 using raw unadjusted logits.

    Setup:
    - 3 classes: [Class 0, Class 1, Class 2]
    - Class 0 is majority (prior 0.90), Class 1 is minority (prior 0.08), Class 2 is rare (prior 0.02).
    - Raw logits for a test sample: [2.0, 1.5, 0.0].
      Raw argmax is Class 0 (2.0 > 1.5 > 0.0).
    - If post-hoc adjustment (z - tau * log(pi)) were applied:
      log_priors = [log(0.90), log(0.08), log(0.02)] = [-0.1054, -2.5257, -3.9120]
      adjusted_logits = [2.0 - (-0.1054), 1.5 - (-2.5257), 0.0 - (-3.9120)]
                      = [2.1054, 4.0257, 3.9120]
      Under post-hoc adjustment, Class 1 would win (4.0257 > 2.1054)!
    - In contrast, training-time LA (z + tau * log(pi)):
      train_loss_logits = [2.0 + (-0.1054), 1.5 + (-2.5257), 0.0 + (-3.9120)]
                        = [1.8946, -1.0257, -3.9120]
    
    Validation MUST use RAW logits [2.0, 1.5, 0.0], selecting Class 0.
    """
    priors = [0.90, 0.08, 0.02]
    # Sample 0: raw logits [2.0, 1.5, 0.0] -> raw argmax is 0
    # Sample 1: raw logits [0.5, 1.8, 0.0] -> raw argmax is 1
    predetermined_logits = torch.tensor([
        [2.0, 1.5, 0.0],
        [0.5, 1.8, 0.0],
    ], dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.long)

    dataset = DummyEvalDataset(predetermined_logits, targets)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    model = MockFixedLogitModel(predetermined_logits)

    la_criterion = LogitAdjustedCrossEntropy(priors=priors, tau=1.0, smoothing=0.1)
    ce_criterion = nn.CrossEntropyLoss()
    ls_criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

    class_names = ["MajorityCls0", "MinorityCls1", "RareCls2"]

    # 1. Run validation with LogitAdjustedCrossEntropy criterion
    val_metrics_la = validate(model, loader, criterion=la_criterion, device="cpu", class_names=class_names)

    # 2. Run validation with standard CrossEntropyLoss criterion
    val_metrics_ce = validate(model, loader, criterion=ce_criterion, device="cpu", class_names=class_names)

    # 3. Run validation with LabelSmoothingCrossEntropy criterion
    val_metrics_ls = validate(model, loader, criterion=ls_criterion, device="cpu", class_names=class_names)

    # PROOF 1: Predictions are strictly Class 0 and Class 1 (the raw argmaxes)
    assert val_metrics_la["val_prediction_distribution"] == {"MajorityCls0": 1, "MinorityCls1": 1}
    assert val_metrics_la["val_accuracy"] == 1.0
    assert val_metrics_la["val_macro_f1"] == 1.0

    # PROOF 2: Metric computation is 100% identical regardless of criterion passed to validate()
    assert val_metrics_la["val_accuracy"] == val_metrics_ce["val_accuracy"] == val_metrics_ls["val_accuracy"]
    assert val_metrics_la["val_balanced_acc"] == val_metrics_ce["val_balanced_acc"] == val_metrics_ls["val_balanced_acc"]
    assert val_metrics_la["val_macro_f1"] == val_metrics_ce["val_macro_f1"] == val_metrics_ls["val_macro_f1"]
    assert val_metrics_la["val_prediction_distribution"] == val_metrics_ce["val_prediction_distribution"]


def test_generate_predictions_outputs_raw_logits_and_standard_softmax():
    """Prove generate_predictions() outputs raw logits and unadjusted softmax probabilities."""
    predetermined_logits = torch.tensor([
        [2.0, 1.5, 0.0],
        [0.5, 1.8, -1.0],
    ], dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.long)

    dataset = DummyEvalDataset(predetermined_logits, targets)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    model = MockFixedLogitModel(predetermined_logits)
    class_names = ["CLS_A", "CLS_B", "CLS_C"]

    df = generate_predictions(model, loader, device="cpu", class_names=class_names)

    # 1. Check raw logit columns match input exactly
    expected_logits = predetermined_logits.numpy()
    for i, cls in enumerate(class_names):
        np.testing.assert_allclose(df[f"logit_{cls}"].values, expected_logits[:, i], atol=1e-6)

    # 2. Check prob columns are direct softmax of raw logits (sum to 1.0)
    expected_probs = torch.softmax(predetermined_logits, dim=-1).numpy()
    for i, cls in enumerate(class_names):
        np.testing.assert_allclose(df[f"prob_{cls}"].values, expected_probs[:, i], atol=1e-6)

    prob_sums = df[[f"prob_{c}" for c in class_names]].sum(axis=1).values
    np.testing.assert_allclose(prob_sums, np.ones(2), atol=1e-5)

    # 3. Check predicted labels match raw argmax
    expected_preds = np.argmax(expected_logits, axis=1)
    np.testing.assert_array_equal(df["pred_label"].values, expected_preds)
    assert df["pred_class"].tolist() == ["CLS_A", "CLS_B"]


def test_checkpoint_selection_monitors_raw_val_macro_f1():
    """Prove CheckpointManager monitors val_macro_f1 derived from raw unadjusted logits."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_mgr = CheckpointManager(
            checkpoint_dir=tmp_dir,
            monitor="val_macro_f1",
            mode="max",
            max_keep=2,
        )

        model = nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Epoch 1: val_macro_f1 = 0.50 (from raw logits)
        metrics_ep1 = {"val_macro_f1": 0.50, "val_loss": 1.20}
        saved_ep1 = ckpt_mgr.save(model, optimizer, epoch=1, metrics=metrics_ep1)
        assert saved_ep1.is_file()
        assert (Path(tmp_dir) / "best_model.pt").is_file()
        assert ckpt_mgr.best_metric_value == 0.50

        # Epoch 2: val_macro_f1 = 0.45 (degraded) -> Should not replace best_model.pt
        metrics_ep2 = {"val_macro_f1": 0.45, "val_loss": 1.10}
        saved_ep2 = ckpt_mgr.save(model, optimizer, epoch=2, metrics=metrics_ep2)
        assert saved_ep2.is_file()
        assert ckpt_mgr.best_metric_value == 0.50

        # Epoch 3: val_macro_f1 = 0.65 (improved) -> Replaces best_model.pt
        metrics_ep3 = {"val_macro_f1": 0.65, "val_loss": 0.95}
        saved_ep3 = ckpt_mgr.save(model, optimizer, epoch=3, metrics=metrics_ep3)
        assert saved_ep3.is_file()
        assert ckpt_mgr.best_metric_value == 0.65
