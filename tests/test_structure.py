"""Verify repository directory structure, essential files, and package imports."""

import os
from pathlib import Path
import pytest


def test_required_directories_exist():
    """Verify that all Phase 0 architectural directories exist."""
    repo_root = Path(__file__).resolve().parent.parent
    expected_dirs = [
        repo_root / "src",
        repo_root / "src" / "data",
        repo_root / "src" / "models",
        repo_root / "src" / "losses",
        repo_root / "src" / "train",
        repo_root / "src" / "eval",
        repo_root / "src" / "utils",
        repo_root / "notebooks",
        repo_root / "configs",
        repo_root / "scripts",
        repo_root / "tests",
        repo_root / "external",
    ]
    for d in expected_dirs:
        assert d.is_dir(), f"Expected directory missing: {d}"


def test_required_files_exist():
    """Verify that key project configuration and source files exist."""
    repo_root = Path(__file__).resolve().parent.parent
    expected_files = [
        repo_root / ".gitignore",
        repo_root / ".env.example",
        repo_root / "requirements.txt",
        repo_root / "environment.yml",
        repo_root / "pyproject.toml",
        repo_root / "README.md",
        repo_root / "setup_colab.ipynb",
        repo_root / "configs" / "baseline.yaml",
        repo_root / "configs" / "data.yaml",
        repo_root / "notebooks" / "01_exploratory_data_analysis.ipynb",
        repo_root / "src" / "__init__.py",
        repo_root / "src" / "data" / "__init__.py",
        repo_root / "src" / "data" / "datasets.py",
        repo_root / "src" / "data" / "transforms.py",
        repo_root / "src" / "data" / "samplers.py",
        repo_root / "src" / "data" / "validator.py",
        repo_root / "src" / "data" / "eda.py",
        repo_root / "src" / "data" / "splitting.py",
        repo_root / "src" / "models" / "__init__.py",
        repo_root / "src" / "losses" / "__init__.py",
        repo_root / "src" / "train" / "__init__.py",
        repo_root / "src" / "eval" / "__init__.py",
        repo_root / "src" / "eval" / "wandb_logger.py",
        repo_root / "src" / "utils" / "config.py",
        repo_root / "src" / "utils" / "reproducibility.py",
    ]
    for f in expected_files:
        assert f.is_file(), f"Expected file missing: {f}"


def test_submodules_configured():
    """Verify that gitmodules references panderm and dermfm-zero."""
    repo_root = Path(__file__).resolve().parent.parent
    gitmodules_path = repo_root / ".gitmodules"
    assert gitmodules_path.is_file(), ".gitmodules must exist"
    content = gitmodules_path.read_text(encoding="utf-8")
    assert "external/panderm" in content
    assert "external/dermfm-zero" in content


def test_gitignore_protects_sensitive_files():
    """Verify that .gitignore excludes data, checkpoints, and secrets."""
    repo_root = Path(__file__).resolve().parent.parent
    gitignore_path = repo_root / ".gitignore"
    content = gitignore_path.read_text(encoding="utf-8")

    patterns = ["/data/", "checkpoints/", "*.pt", "*.pth", ".env", "wandb/"]
    for pattern in patterns:
        assert pattern in content, f"Missing {pattern} in .gitignore"


def test_package_imports():
    """Verify that src and its subpackages can be imported cleanly."""
    import src
    import src.data
    import src.models
    import src.losses
    import src.train
    import src.eval
    import src.utils

    assert hasattr(src, "__version__")
    assert callable(src.data.SkinLesionDataset)
    assert callable(src.data.DatasetValidator)
    assert callable(src.data.get_dataset_overview)
    assert callable(src.data.create_lesion_level_splits)
    assert callable(src.models.build_model)
    assert callable(src.losses.build_loss)
    assert callable(src.eval.WandbLogger)
    assert callable(src.utils.load_config)
