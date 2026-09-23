"""Inference and prediction generation for dermatology classification models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.eval.metrics import DEFAULT_CLASSES

logger = logging.getLogger(__name__)


def generate_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: Union[str, torch.device] = "cpu",
    class_names: Optional[List[str]] = None,
    max_batches: Optional[int] = None,
) -> pd.DataFrame:
    """Run model inference over DataLoader and return structured predictions DataFrame.

    Args:
        model: Evaluated PyTorch model.
        dataloader: PyTorch DataLoader yielding (images, labels, metadata).
        device: Device to run evaluation on.
        class_names: List of class names.
        max_batches: Optional max batch limit for quick testing.

    Returns:
        pd.DataFrame containing isic_id, lesion_id, split, true_label, pred_label,
        true_class, pred_class, skin_tone_class, and per-class probability columns.
    """
    classes = class_names or DEFAULT_CLASSES
    model.eval()
    model.to(device)

    all_isic_ids = []
    all_lesion_ids = []
    all_skin_tones = []
    all_splits = []
    all_true_labels = []
    all_pred_labels = []
    all_probs = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images, labels, metadata = batch
            images = images.to(device)

            logits = model(images)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_true_labels.extend(labels.numpy().tolist())
            all_pred_labels.extend(preds.tolist())
            all_probs.append(probs)

            # Extract batch metadata
            isic_ids = metadata.get("isic_id", [""] * len(labels))
            lesion_ids = metadata.get("lesion_id", [""] * len(labels))
            skin_tones = metadata.get("skin_tone_class", [-1] * len(labels))
            splits = metadata.get("split", [""] * len(labels))

            all_isic_ids.extend([str(x) for x in isic_ids])
            all_lesion_ids.extend([str(x) for x in lesion_ids])
            all_skin_tones.extend(skin_tones if isinstance(skin_tones, list) else skin_tones.numpy().tolist() if hasattr(skin_tones, "numpy") else list(skin_tones))
            all_splits.extend([str(x) for x in splits])

    all_probs_mat = np.vstack(all_probs) if all_probs else np.empty((0, len(classes)))

    df = pd.DataFrame({
        "isic_id": all_isic_ids,
        "lesion_id": all_lesion_ids,
        "split": all_splits,
        "skin_tone_class": all_skin_tones,
        "true_label": all_true_labels,
        "true_class": [classes[i] if 0 <= i < len(classes) else "Unknown" for i in all_true_labels],
        "pred_label": all_pred_labels,
        "pred_class": [classes[i] if 0 <= i < len(classes) else "Unknown" for i in all_pred_labels],
    })

    # Add probability columns
    for i, cls_name in enumerate(classes):
        df[f"prob_{cls_name}"] = all_probs_mat[:, i]

    return df


def save_predictions(df_predictions: pd.DataFrame, output_path: Union[str, Path]) -> Path:
    """Save predictions DataFrame to CSV."""
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    df_predictions.to_csv(path, index=False)
    logger.info(f"Saved predictions to {path}")
    return path
