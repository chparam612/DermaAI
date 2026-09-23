"""Evaluation metrics computation for dermatology diagnostic classification.

Calculates standard classification metrics, per-class performance, and
skin-tone subgroup fairness audits without synthetic interpolation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

DEFAULT_CLASSES = [
    "AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF",
    "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"
]


def calculate_classification_metrics(
    y_true: Union[np.ndarray, List[int]],
    y_pred: Union[np.ndarray, List[int]],
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Calculate aggregate and per-class classification metrics.

    Args:
        y_true: Array of true class integer labels.
        y_pred: Array of predicted class integer labels.
        class_names: List of class names ordered by integer index.

    Returns:
        Dictionary containing overall and per-class metrics.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = class_names or DEFAULT_CLASSES

    # Aggregate metrics
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    # Per-class metrics
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=list(range(len(classes))), zero_division=0)
    per_class_rec = recall_score(y_true, y_pred, average=None, labels=list(range(len(classes))), zero_division=0)
    per_class_prec = precision_score(y_true, y_pred, average=None, labels=list(range(len(classes))), zero_division=0)

    per_class_dict = {}
    for i, name in enumerate(classes):
        per_class_dict[name] = {
            "f1": float(per_class_f1[i]),
            "recall": float(per_class_rec[i]),
            "precision": float(per_class_prec[i]),
            "support": int(np.sum(y_true == i)),
        }

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "per_class": per_class_dict,
        "confusion_matrix": cm.tolist(),
        "total_samples": len(y_true),
    }


def calculate_subgroup_metrics(
    df_predictions: pd.DataFrame,
    subgroup_col: str = "skin_tone_class",
    true_col: str = "true_label",
    pred_col: str = "pred_label",
) -> Dict[str, Any]:
    """Audit performance disaggregated by skin-tone class.

    Args:
        df_predictions: DataFrame containing predictions and metadata.
        subgroup_col: Column representing demographic attribute (skin tone).
        true_col: Ground truth label column.
        pred_col: Model prediction label column.

    Returns:
        Dictionary with per-subgroup metrics and disparity measures.
    """
    if subgroup_col not in df_predictions.columns:
        return {"error": f"Subgroup column '{subgroup_col}' not found"}

    subgroup_results: Dict[str, Any] = {}
    subgroup_accs: List[float] = []
    subgroup_bal_accs: List[float] = []

    unique_subgroups = sorted(df_predictions[subgroup_col].dropna().unique())

    for val in unique_subgroups:
        sub_df = df_predictions[df_predictions[subgroup_col] == val]
        n_samples = len(sub_df)
        if n_samples == 0:
            continue

        y_true = sub_df[true_col].values
        y_pred = sub_df[pred_col].values

        sub_acc = float(accuracy_score(y_true, y_pred))
        # Balanced accuracy requires at least 2 distinct classes present
        try:
            sub_bal_acc = float(balanced_accuracy_score(y_true, y_pred))
        except Exception:
            sub_bal_acc = sub_acc

        sub_macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        subgroup_key = f"skin_tone_{val}"
        subgroup_results[subgroup_key] = {
            "sample_count": n_samples,
            "accuracy": sub_acc,
            "balanced_accuracy": sub_bal_acc,
            "macro_f1": sub_macro_f1,
        }

        # Track groups with at least 10 samples for robust disparity calculation
        if n_samples >= 10:
            subgroup_accs.append(sub_acc)
            subgroup_bal_accs.append(sub_bal_acc)

    # Compute disparities (max difference across represented groups)
    acc_disparity = float(max(subgroup_accs) - min(subgroup_accs)) if len(subgroup_accs) > 1 else 0.0
    bal_acc_disparity = float(max(subgroup_bal_accs) - min(subgroup_bal_accs)) if len(subgroup_bal_accs) > 1 else 0.0

    return {
        "subgroups": subgroup_results,
        "accuracy_disparity": acc_disparity,
        "balanced_accuracy_disparity": bal_acc_disparity,
        "evaluated_subgroups_count": len(subgroup_results),
    }
