"""Reproducibility and Environment Tracking Utility for SkinCancerTracker.

Ensures fully deterministic execution where supported, and captures complete
hardware, software, Git, and configuration provenance for auditability.
"""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
import torch
import yaml


def set_seed(seed: int = 42, deterministic_cudnn: bool = True) -> int:
    """Set random seeds across Python, NumPy, and PyTorch for reproducible runs.

    Args:
        seed: Integer seed value.
        deterministic_cudnn: If True, configures CuDNN backends for determinism.

    Returns:
        The seed integer that was set.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic_cudnn:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True

    return seed


def get_git_commit_hash(repo_dir: Optional[Union[str, Path]] = None) -> Optional[str]:
    """Retrieve current Git commit hash if in a Git repository.

    Args:
        repo_dir: Path to repository root. Defaults to current working directory.

    Returns:
        Hexadecimal commit hash string, or None if Git is unavailable / not a repo.
    """
    cwd = Path(repo_dir).resolve() if repo_dir else Path.cwd()
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return output
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


def get_git_status_summary(repo_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Retrieve Git commit hash and dirty status.

    Args:
        repo_dir: Path to repository root.

    Returns:
        Dictionary with commit hash, branch, and is_dirty flag.
    """
    cwd = Path(repo_dir).resolve() if repo_dir else Path.cwd()
    info: Dict[str, Any] = {
        "commit": get_git_commit_hash(cwd),
        "branch": None,
        "is_dirty": False,
    }
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        info["branch"] = branch

        dirty_check = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        info["is_dirty"] = len(dirty_check) > 0
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        pass

    return info


def get_hardware_info() -> Dict[str, Any]:
    """Capture host system architecture, CPU, and GPU devices.

    Returns:
        Dictionary detailing available compute hardware and memory.
    """
    info: Dict[str, Any] = {
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": [],
    }

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(i)
            info["devices"].append({
                "index": i,
                "name": prop.name,
                "total_memory_mb": round(prop.total_memory / (1024 ** 2), 2),
                "major_capability": prop.major,
                "minor_capability": prop.minor,
            })
    return info


def get_environment_versions() -> Dict[str, str]:
    """Retrieve versions of key scientific and deep learning packages."""
    versions: Dict[str, str] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda if hasattr(torch.version, "cuda") else "none",
        "numpy": np.__version__,
    }

    # Dynamically query optional packages
    for pkg in ["torchvision", "pandas", "scipy", "sklearn", "PIL", "yaml", "wandb", "timm", "transformers"]:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "unknown")
            versions[pkg] = ver
        except ImportError:
            versions[pkg] = "not_installed"

    return versions


def capture_reproducibility_manifest(
    seed: int,
    config: Optional[Dict[str, Any]] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Build a comprehensive reproducibility manifest capturing software, hardware, and Git state.

    Args:
        seed: Random seed used in the run.
        config: Optional experiment configuration dictionary.
        output_path: If provided, writes the manifest as a JSON file to disk.

    Returns:
        Dictionary containing the full reproducibility state.
    """
    manifest: Dict[str, Any] = {
        "seed": seed,
        "git": get_git_status_summary(),
        "hardware": get_hardware_info(),
        "dependencies": get_environment_versions(),
        "config": config,
    }

    if output_path is not None:
        out = Path(output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, default=str)

    return manifest
