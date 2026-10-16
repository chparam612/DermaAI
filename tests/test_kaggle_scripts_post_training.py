"""Regression test for post-training prediction, metrics, and subgroup fairness pipeline across all Kaggle scripts."""

import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.eval.metrics import calculate_classification_metrics, calculate_subgroup_metrics
from src.eval.predictions import save_predictions
from scripts.audit_script_calls import audit_script


@pytest.fixture
def synthetic_predictions():
    """Generate realistic synthetic predictions DataFrame for 11 dermatology classes."""
    classes = [
        "AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF",
        "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"
    ]
    n_samples = 40
    np.random.seed(42)

    true_labels = np.random.randint(0, 11, size=n_samples)
    pred_labels = np.random.randint(0, 11, size=n_samples)
    skin_tones = np.random.choice([0, 1, 2, 3, 4, 5], size=n_samples)

    # Generate probabilities summing to 1
    raw_probs = np.random.exponential(scale=1.0, size=(n_samples, 11))
    probs = raw_probs / raw_probs.sum(axis=1, keepdims=True)

    data = {
        "isic_id": [f"ISIC_{i:07d}" for i in range(n_samples)],
        "lesion_id": [f"IL_{i:07d}" for i in range(n_samples)],
        "split": ["test"] * n_samples,
        "skin_tone_class": skin_tones,
        "true_label": true_labels,
        "true_class": [classes[i] for i in true_labels],
        "pred_label": pred_labels,
        "pred_class": [classes[i] for i in pred_labels],
    }
    for idx, cls in enumerate(classes):
        data[f"prob_{cls}"] = probs[:, idx]
        data[f"logit_{cls}"] = np.log(probs[:, idx] + 1e-12)

    return pd.DataFrame(data), classes


def test_exp1_post_training_pipeline(synthetic_predictions, tmp_path):
    """Verify Exp-1 post-training evaluation block executes without TypeError."""
    df_test, class_names = synthetic_predictions
    results_dir = tmp_path / "results_exp1"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Classification metrics
    test_prob_cols = [f"prob_{c}" for c in class_names]
    test_metrics = calculate_classification_metrics(
        y_true=df_test["true_label"].values,
        y_pred=df_test["pred_label"].values,
        y_prob=df_test[test_prob_cols].values,
        class_names=class_names,
    )
    assert "accuracy" in test_metrics
    assert "macro_f1" in test_metrics

    # 2. Subgroup metrics (strictly no class_names kwarg)
    subgroup_metrics = calculate_subgroup_metrics(
        df_predictions=df_test,
        subgroup_col="skin_tone_class",
    )
    assert isinstance(subgroup_metrics, dict)
    assert "per_subgroup" in subgroup_metrics or "error" not in subgroup_metrics

    # 3. File saving
    metrics_data = {
        "seed": 42,
        "val_overall": test_metrics,
        "test_overall": test_metrics,
        "subgroup_fairness": subgroup_metrics,
    }
    metrics_path = results_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    assert metrics_path.is_file()
    with open(metrics_path) as f:
        loaded = json.load(f)
    assert loaded["seed"] == 42


def test_exp2c_and_exp3b_post_training_pipeline(synthetic_predictions, tmp_path):
    """Verify Exp-2c and Exp-3b post-training evaluation block executes cleanly."""
    df_test, class_names = synthetic_predictions
    results_dir = tmp_path / "results_exp2c"
    results_dir.mkdir(parents=True, exist_ok=True)

    prob_cols = [f"prob_{c}" for c in class_names]
    test_metrics = calculate_classification_metrics(
        y_true=df_test["true_label"].values,
        y_pred=df_test["pred_label"].values,
        y_prob=df_test[prob_cols].values,
        class_names=class_names,
    )

    subgroup_metrics = calculate_subgroup_metrics(
        df_predictions=df_test,
        subgroup_col="skin_tone_class",
    )
    assert isinstance(subgroup_metrics, dict)

    summary_data = {
        "experiment_name": "exp2c-panderm-uniform",
        "seed": 42,
        "validation_overall": test_metrics,
        "test_overall": test_metrics,
        "subgroup_fairness": subgroup_metrics,
    }
    summary_path = results_dir / "summary_metrics.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    assert summary_path.is_file()


def test_all_kaggle_scripts_ast_no_signature_mismatches():
    """Statically verify every src/ call in all 3 Kaggle scripts has 0 signature mismatches."""
    scripts = [
        "kaggle/run_exp1_resnet50.py",
        "kaggle/run_exp2c_panderm_uniform.py",
        "kaggle/run_exp3b_panderm_logit_adj.py",
    ]
    for s in scripts:
        valid_calls, mismatches = audit_script(s)
        assert len(mismatches) == 0, f"Found signature mismatches in {s}: {mismatches}"
        assert len(valid_calls) > 0, f"No valid src calls found in {s}"
