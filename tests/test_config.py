"""Unit tests for experiment configuration loader and validator."""

from pathlib import Path
import pytest
import yaml

from src.utils.config import (
    ConfigDict,
    ConfigValidationError,
    load_config,
    validate_config,
)


@pytest.fixture
def baseline_config_path():
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / "configs" / "baseline.yaml"


def test_load_baseline_config(baseline_config_path):
    """Ensure configs/baseline.yaml loads and validates successfully."""
    config = load_config(baseline_config_path)
    assert isinstance(config, ConfigDict)
    assert config.dataset.name == "isic2020"
    assert config.dataloader.batch_size > 0
    assert config.training.epochs > 0
    assert config.model.name is not None
    assert config.reproducibility.seed == 42
    assert config.wandb.project is not None


def test_config_dot_access(baseline_config_path):
    """Test attribute-style dot access and dict access equivalence."""
    config = load_config(baseline_config_path)
    assert config.dataset.image_size == config["dataset"]["image_size"]
    assert config.training.learning_rate == config["training"]["learning_rate"]


def test_validate_config_missing_section():
    """Ensure validation fails when a required top-level section is missing."""
    incomplete = {
        "dataset": {"name": "test", "root_dir": "data", "image_size": 224},
        # missing dataloader, model, training, loss, etc.
    }
    with pytest.raises(ConfigValidationError) as exc:
        validate_config(incomplete)
    assert "Missing required top-level section" in str(exc.value)


def test_validate_config_missing_field():
    """Ensure validation fails when a required nested key is missing."""
    invalid = {
        "dataset": {"name": "test"},  # missing root_dir and image_size
        "dataloader": {"batch_size": 32, "num_workers": 2},
        "model": {"name": "resnet18"},
        "training": {"epochs": 10, "learning_rate": 0.001, "optimizer": "adam"},
        "loss": {"name": "cross_entropy"},
        "reproducibility": {"seed": 42},
        "output": {"checkpoint_dir": "ckpt", "log_dir": "logs"},
        "wandb": {"enabled": False, "project": "test"},
    }
    with pytest.raises(ConfigValidationError) as exc:
        validate_config(invalid)
    assert "missing required field: 'root_dir'" in str(exc.value)


def test_validate_config_invalid_types():
    """Ensure validation rejects invalid epoch or batch size values."""
    invalid = {
        "dataset": {"name": "test", "root_dir": "data", "image_size": 224},
        "dataloader": {"batch_size": -5, "num_workers": 2},
        "model": {"name": "resnet18"},
        "training": {"epochs": 0, "learning_rate": -0.01, "optimizer": "adam"},
        "loss": {"name": "cross_entropy"},
        "reproducibility": {"seed": 42},
        "output": {"checkpoint_dir": "ckpt", "log_dir": "logs"},
        "wandb": {"enabled": False, "project": "test"},
    }
    with pytest.raises(ConfigValidationError) as exc:
        validate_config(invalid)
    assert "positive integer" in str(exc.value)


def test_load_nonexistent_config(tmp_path):
    """Ensure loading a nonexistent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does_not_exist.yaml")
