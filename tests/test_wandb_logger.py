"""Unit tests for WandbLogger offline capabilities and medical metric tables."""

import pytest
from src.eval.wandb_logger import (
    WandbLogger,
    PER_CLASS_TABLE_COLUMNS,
    PER_SKIN_TONE_TABLE_COLUMNS,
    SUBGROUP_ERROR_TABLE_COLUMNS,
)


@pytest.fixture
def offline_logger():
    """Returns a WandbLogger instance with W&B disabled for offline test isolation."""
    config = {
        "wandb": {
            "enabled": False,
            "project": "test-project",
            "experiment_name": "unit-test-run",
        }
    }
    return WandbLogger(config=config, enabled=False)


def test_offline_logger_init(offline_logger):
    """Verify offline logger initializes cleanly without requiring network or API key."""
    assert offline_logger.enabled is False
    assert offline_logger.run is None
    assert offline_logger.project == "test-project"


def test_log_metrics(offline_logger):
    """Verify scalar metrics are tracked in history."""
    metrics = {"train_loss": 0.452, "val_loss": 0.389, "val_macro_f1": 0.812}
    offline_logger.log_metrics(metrics, step=1)
    assert len(offline_logger.history) == 1
    assert offline_logger.history[0]["step"] == 1
    assert offline_logger.history[0]["train_loss"] == 0.452


def test_log_per_class_table(offline_logger):
    """Verify Table 1 (per-class metrics) construction and columns."""
    sample_class_metrics = [
        {
            "class_name": "melanoma",
            "support": 120,
            "precision": 0.88,
            "recall": 0.82,
            "f1_score": 0.85,
            "specificity": 0.94,
            "balanced_accuracy": 0.88,
        },
        {
            "class_name": "nevus",
            "support": 800,
            "precision": 0.96,
            "recall": 0.97,
            "f1_score": 0.965,
            "specificity": 0.82,
            "balanced_accuracy": 0.895,
        },
    ]

    table_dict = offline_logger.log_per_class_table(sample_class_metrics, step=1)
    assert table_dict["columns"] == PER_CLASS_TABLE_COLUMNS
    assert len(table_dict["data"]) == 2
    assert table_dict["data"][0][0] == "melanoma"
    assert table_dict["data"][0][1] == 120
    assert "eval/per_class_metrics" in offline_logger.logged_tables


def test_log_per_skin_tone_table(offline_logger):
    """Verify Table 2 (per-skin-tone metrics) handles missing subgroup metadata gracefully."""
    sample_skin_tone_metrics = [
        {
            "skin_tone_group": "Fitzpatrick I-II",
            "support": 450,
            "accuracy": 0.91,
            "balanced_accuracy": 0.89,
            "precision": 0.88,
            "recall": 0.85,
            "f1_score": 0.865,
            "auc": 0.95,
        },
        {
            "skin_tone_group": "Fitzpatrick V-VI",
            "support": 65,
            "accuracy": 0.83,
            "balanced_accuracy": 0.79,
            "precision": 0.76,
            "recall": 0.72,
            "f1_score": 0.74,
            "auc": 0.88,
        },
        {
            # Missing subgroup annotation: must NOT crash and must NOT fabricate a label
            "skin_tone_group": None,
            "support": 30,
            "accuracy": 0.80,
            "balanced_accuracy": 0.75,
            "precision": 0.70,
            "recall": 0.68,
            "f1_score": 0.69,
            "auc": None,
        },
    ]

    table_dict = offline_logger.log_per_skin_tone_table(sample_skin_tone_metrics, step=1)
    assert table_dict["columns"] == PER_SKIN_TONE_TABLE_COLUMNS
    assert len(table_dict["data"]) == 3
    # Check that None was converted to an explicit unannotated label rather than fabricated
    assert "Unannotated" in table_dict["data"][2][0]
    # Check that None AUC remains None, not fake 0.0
    assert table_dict["data"][2][7] is None


def test_log_subgroup_error_table(offline_logger):
    """Verify Table 3 (subgroup error analysis) structure."""
    sample_errors = [
        {
            "sample_id": "ISIC_0012345",
            "true_label": "melanoma",
            "predicted_label": "seborrheic keratosis",
            "confidence": 0.642,
            "skin_tone_group": "Fitzpatrick IV",
            "error_type": "false_negative",
        }
    ]
    table_dict = offline_logger.log_subgroup_error_table(sample_errors, step=1)
    assert table_dict["columns"] == SUBGROUP_ERROR_TABLE_COLUMNS
    assert len(table_dict["data"]) == 1
    assert table_dict["data"][0][0] == "ISIC_0012345"
    assert table_dict["data"][0][1] == "melanoma"


def test_logger_finish(offline_logger):
    """Verify finish runs cleanly in offline mode."""
    offline_logger.finish()
    assert offline_logger.run is None
