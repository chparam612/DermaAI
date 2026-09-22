"""Experiment Configuration Loader and Validator for SkinCancerTracker.

Provides YAML configuration loading, structured schema validation,
and CLI parameter parsing for reproducible research experiments.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml


class ConfigValidationError(ValueError):
    """Raised when an experiment configuration file fails validation."""
    pass


class ConfigDict(dict):
    """Dictionary that allows attribute-style access (e.g. config.model.name)."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
            if isinstance(value, dict) and not isinstance(value, ConfigDict):
                value = ConfigDict(value)
                self[key] = value
            return value
        except KeyError:
            raise AttributeError(f"Configuration key '{key}' not found.")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"Configuration key '{key}' not found.")


REQUIRED_CONFIG_SCHEMA: Dict[str, List[str]] = {
    "dataset": ["name", "root_dir", "image_size"],
    "dataloader": ["batch_size", "num_workers"],
    "model": ["name"],
    "training": ["epochs", "learning_rate", "optimizer"],
    "loss": ["name"],
    "reproducibility": ["seed"],
    "output": ["checkpoint_dir", "log_dir"],
    "wandb": ["enabled", "project"],
}


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load raw dictionary from a YAML file.

    Args:
        path: Path to YAML configuration file.

    Returns:
        Loaded configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ConfigValidationError: If the YAML syntax is invalid or file is empty.
    """
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigValidationError(f"Malformed YAML in '{config_path}': {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigValidationError(
            f"Configuration at '{config_path}' must evaluate to a dictionary, got {type(data).__name__}"
        )

    return data


def validate_config(config: Dict[str, Any], schema: Optional[Dict[str, List[str]]] = None) -> None:
    """Validate that required top-level and nested sections exist.

    Args:
        config: Configuration dictionary to validate.
        schema: Optional custom schema mapping section names to required keys.

    Raises:
        ConfigValidationError: If any required section or field is missing.
    """
    validation_schema = schema or REQUIRED_CONFIG_SCHEMA
    missing_errors: List[str] = []

    for section, required_keys in validation_schema.items():
        if section not in config:
            missing_errors.append(f"Missing required top-level section: '{section}'")
            continue

        section_data = config[section]
        if not isinstance(section_data, dict):
            missing_errors.append(
                f"Section '{section}' must be a mapping/dict, got {type(section_data).__name__}"
            )
            continue

        for key in required_keys:
            if key not in section_data:
                missing_errors.append(
                    f"Section '{section}' is missing required field: '{key}'"
                )

    # Validate value constraints
    if "training" in config and isinstance(config["training"], dict):
        epochs = config["training"].get("epochs")
        if epochs is not None and (not isinstance(epochs, int) or epochs <= 0):
            missing_errors.append(f"training.epochs must be a positive integer, got: {epochs}")

        lr = config["training"].get("learning_rate")
        if lr is not None and (not isinstance(lr, (int, float)) or lr <= 0):
            missing_errors.append(f"training.learning_rate must be a positive number, got: {lr}")

    if "dataloader" in config and isinstance(config["dataloader"], dict):
        batch_size = config["dataloader"].get("batch_size")
        if batch_size is not None and (not isinstance(batch_size, int) or batch_size <= 0):
            missing_errors.append(f"dataloader.batch_size must be a positive integer, got: {batch_size}")

    if missing_errors:
        error_msg = "Configuration validation failed:\n  - " + "\n  - ".join(missing_errors)
        raise ConfigValidationError(error_msg)


def load_config(
    path: Union[str, Path],
    validate: bool = True,
    return_dictconfig: bool = True,
) -> Union[ConfigDict, Dict[str, Any]]:
    """Load and optionally validate an experiment YAML configuration.

    Args:
        path: Path to the YAML file.
        validate: Whether to validate the configuration against required schema.
        return_dictconfig: If True, returns a ConfigDict for dot-accessible access.

    Returns:
        ConfigDict or standard dictionary containing configuration settings.
    """
    raw_config = load_yaml(path)
    if validate:
        validate_config(raw_config)

    return ConfigDict(raw_config) if return_dictconfig else raw_config


def parse_cli_args() -> argparse.Namespace:
    """Parse command-line arguments for experiment configuration selection."""
    parser = argparse.ArgumentParser(
        description="SkinCancerTracker Experiment Configuration Runner"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to YAML experiment configuration file (default: configs/baseline.yaml)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Override W&B experiment name",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override dataloader batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Disable online W&B logging (runs in offline/disabled mode)",
    )
    return parser.parse_args()


def get_config_from_cli() -> ConfigDict:
    """Load configuration specified by CLI arguments with command-line overrides applied."""
    args = parse_cli_args()
    config = load_config(args.config)

    # Apply CLI overrides if provided
    if args.experiment_name is not None:
        config.setdefault("wandb", {})["experiment_name"] = args.experiment_name
    if args.seed is not None:
        config.setdefault("reproducibility", {})["seed"] = args.seed
    if args.batch_size is not None:
        config.setdefault("dataloader", {})["batch_size"] = args.batch_size
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.lr is not None:
        config.setdefault("training", {})["learning_rate"] = args.lr
    if args.offline:
        config.setdefault("wandb", {})["enabled"] = False

    return config


if __name__ == "__main__":
    cfg = get_config_from_cli()
    print("Configuration loaded and validated successfully:")
    print(f"  Experiment: {cfg.experiment_name}")
    print(f"  Dataset:    {cfg.dataset.name} ({cfg.dataset.image_size}x{cfg.dataset.image_size})")
    print(f"  Model:      {cfg.model.name}")
    print(f"  Batch Size: {cfg.dataloader.batch_size}")
    print(f"  Epochs:     {cfg.training.epochs}")
    print(f"  W&B:        {'Enabled' if cfg.wandb.enabled else 'Disabled'}")

