"""Custom samplers for class-imbalanced and subgroup-aware medical data loaders."""

from __future__ import annotations

from typing import Iterator, Sequence
import numpy as np
import torch
from torch.utils.data import Sampler, WeightedRandomSampler


def create_balanced_class_sampler(labels: Sequence[int]) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler that balances sampling frequencies across classes.

    Essential for dermatology datasets (e.g. HAM10000, ISIC) where benign classes
    heavily outnumber rare malignancies like melanoma and dermatofibroma.

    Args:
        labels: Sequence of integer class labels for the training set.

    Returns:
        WeightedRandomSampler configured with inverse class frequency weights.
    """
    labels_arr = np.array(labels)
    unique_classes, counts = np.unique(labels_arr, return_counts=True)
    class_weights = {cls: 1.0 / count for cls, count in zip(unique_classes, counts)}

    sample_weights = [class_weights[label] for label in labels_arr]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler
