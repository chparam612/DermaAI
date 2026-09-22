"""Unit tests for reproducibility utilities."""

import json
from pathlib import Path
import numpy as np
import torch
import pytest

from src.utils.reproducibility import (
    set_seed,
    get_git_commit_hash,
    get_git_status_summary,
    get_hardware_info,
    get_environment_versions,
    capture_reproducibility_manifest,
)


def test_set_seed_reproducibility():
    """Verify that set_seed ensures deterministic random number sequences."""
    set_seed(1234)
    r1 = np.random.rand(5)
    t1 = torch.rand(5)

    set_seed(1234)
    r2 = np.random.rand(5)
    t2 = torch.rand(5)

    np.testing.assert_allclose(r1, r2)
    assert torch.equal(t1, t2)


def test_get_hardware_info():
    """Verify hardware info structure contains system and torch details."""
    hw = get_hardware_info()
    assert "os" in hw
    assert "python_version" in hw
    assert "cuda_available" in hw
    assert isinstance(hw["devices"], list)


def test_get_environment_versions():
    """Verify environment versions capture core dependencies."""
    vers = get_environment_versions()
    assert "torch" in vers
    assert "numpy" in vers
    assert "python" in vers


def test_get_git_status_summary():
    """Verify git status returns dictionary with commit info."""
    git_info = get_git_status_summary()
    assert "commit" in git_info
    assert "branch" in git_info
    assert "is_dirty" in git_info


def test_capture_reproducibility_manifest(tmp_path):
    """Verify manifest can be generated and written to disk as valid JSON."""
    out_file = tmp_path / "manifest.json"
    manifest = capture_reproducibility_manifest(
        seed=999,
        config={"test_key": "test_val"},
        output_path=out_file,
    )
    assert manifest["seed"] == 999
    assert out_file.is_file()

    with open(out_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["seed"] == 999
    assert loaded["config"]["test_key"] == "test_val"
