"""Regression test for PyTorch checkpoint loading with numpy scalar metrics."""

from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.train.checkpointing import CheckpointManager


def test_checkpointing_with_numpy_scalars(tmp_path: Path):
    """Test 1: save a checkpoint with numpy scalars via CheckpointManager and reload via load()."""
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Perform a dummy step so optimizer state is populated
    x = torch.randn(2, 4)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()

    metrics = {
        "val_acc": np.float64(0.85),
        "val_macro_f1": np.float32(0.72),
        "val_count": np.int64(100),
        "nested": {
            "sub_metric": np.float64(0.99),
        },
    }

    ckpt_mgr = CheckpointManager(checkpoint_dir=tmp_path)
    save_path = ckpt_mgr.save(
        model=model,
        optimizer=optimizer,
        epoch=1,
        metrics=metrics,
        filename="test_ckpt.pt",
    )

    # Reload into fresh model and optimizer
    eval_model = nn.Linear(4, 2)
    eval_opt = torch.optim.SGD(eval_model.parameters(), lr=0.01)
    loaded_state = ckpt_mgr.load(save_path, eval_model, eval_opt)

    # Assert model weights are equal
    for p1, p2 in zip(model.parameters(), eval_model.parameters()):
        assert torch.equal(p1, p2), "Model parameters mismatch after checkpoint reload"

    # Assert metric values match
    loaded_metrics = loaded_state["metrics"]
    assert pytest.approx(loaded_metrics["val_acc"]) == 0.85
    assert pytest.approx(loaded_metrics["val_macro_f1"]) == 0.72
    assert loaded_metrics["val_count"] == 100
    assert pytest.approx(loaded_metrics["nested"]["sub_metric"]) == 0.99

    # Assert future checkpoints are loadable even under weights_only=True
    direct_loaded = torch.load(save_path, map_location="cpu", weights_only=True)
    assert direct_loaded["metrics"]["val_count"] == 100
    assert pytest.approx(direct_loaded["metrics"]["val_acc"]) == 0.85


def test_checkpointing_legacy_raw_numpy_scalars(tmp_path: Path):
    """Test 2: write a checkpoint the OLD way (torch.save with raw numpy scalars in the dict) and assert load() succeeds."""
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    raw_metrics = {
        "legacy_f64": np.float64(0.6543),
        "legacy_f32": np.float32(0.4321),
        "legacy_i64": np.int64(42),
    }

    legacy_ckpt_path = tmp_path / "legacy_ckpt.pt"
    raw_state = {
        "epoch": 5,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": raw_metrics,
    }
    torch.save(raw_state, legacy_ckpt_path)

    # Verify that CheckpointManager.load() loads this legacy checkpoint without error
    eval_model = nn.Linear(4, 2)
    eval_opt = torch.optim.SGD(eval_model.parameters(), lr=0.01)
    loaded_state = CheckpointManager.load(legacy_ckpt_path, eval_model, eval_opt)

    for p1, p2 in zip(model.parameters(), eval_model.parameters()):
        assert torch.equal(p1, p2), "Model parameters mismatch after legacy checkpoint reload"

    assert pytest.approx(loaded_state["metrics"]["legacy_f64"]) == 0.6543
    assert pytest.approx(loaded_state["metrics"]["legacy_f32"]) == 0.4321
    assert loaded_state["metrics"]["legacy_i64"] == 42
