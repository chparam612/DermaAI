"""Data loader factory for training, validation, and testing pipelines."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.datasets import SkinLesionDataset
from src.data.transforms import get_transforms


def create_dataloaders(
    metadata_path: Union[str, Path] = "data/splits/lesion_level_splits.csv",
    image_dir: Union[str, Path] = "data/images",
    pipeline_type: str = "baseline",
    batch_size: int = 32,
    eval_batch_size: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    image_size: int = 224,
    label_mapping: Optional[Union[str, Path, Dict[str, int]]] = None,
    modality: Optional[str] = "dermoscopic",
    use_weighted_sampler: Optional[bool] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch DataLoaders for train, validation, and test splits.

    Guarantees strict lesion-level separation via Phase 1 splits file.

    Args:
        metadata_path: Path to lesion_level_splits.csv.
        image_dir: Path to image directory.
        pipeline_type: 'baseline' (Exp-1) or 'panderm' (Exp-2).
        batch_size: Batch size for training.
        eval_batch_size: Batch size for validation and testing (defaults to batch_size).
        num_workers: Number of workers for DataLoader.
        pin_memory: Whether to pin memory for CUDA transfer.
        persistent_workers: Whether to keep worker processes alive across epochs.
        image_size: Input spatial dimension.
        label_mapping: Label mapping path or dictionary.
        modality: Modality filter ('dermoscopic', 'clinical', or None for all).

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    eval_bs = eval_batch_size or batch_size

    # Build transforms
    train_transform = get_transforms(pipeline_type=pipeline_type, mode="train", image_size=image_size)
    eval_transform = get_transforms(pipeline_type=pipeline_type, mode="val", image_size=image_size)

    # Instantiate datasets
    train_dataset = SkinLesionDataset(
        metadata=metadata_path,
        image_dir=image_dir,
        transform=train_transform,
        split="train",
        label_mapping=label_mapping,
        modality=modality,
    )

    val_dataset = SkinLesionDataset(
        metadata=metadata_path,
        image_dir=image_dir,
        transform=eval_transform,
        split="validation",
        label_mapping=label_mapping,
        modality=modality,
    )

    test_dataset = SkinLesionDataset(
        metadata=metadata_path,
        image_dir=image_dir,
        transform=eval_transform,
        split="test",
        label_mapping=label_mapping,
        modality=modality,
    )

    use_persistent = persistent_workers if num_workers > 0 else False

    # Imbalance handling: PanDerm uses WeightedRandomSampler (inverse class frequencies)
    apply_sampler = (pipeline_type.lower() == "panderm") if use_weighted_sampler is None else bool(use_weighted_sampler)
    train_sampler = None
    train_shuffle = True

    if apply_sampler:
        labels = train_dataset.get_labels()
        label_series = pd.Series(labels)
        label_counts = label_series.value_counts().sort_index()
        total_samples = len(labels)
        num_classes = len(label_counts)
        weight_dict = {
            cls: total_samples / (num_classes * count)
            for cls, count in label_counts.items()
        }
        sample_weights = torch.tensor([weight_dict[y] for y in labels], dtype=torch.double)
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
        )
        train_shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
