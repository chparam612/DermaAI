"""Dataset classes for Skin Lesion and Dermatology Image Analysis.

Provides modular PyTorch Dataset implementations supporting multi-class,
binary diagnosis, and subgroup/skin-tone metadata preservation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class SkinLesionDataset(Dataset):
    """Modular PyTorch Dataset for dermatology image classification.

    Expects a metadata DataFrame or CSV containing at least:
    - 'image' or 'image_path': filename or relative/absolute path to image
    - 'label' (for multiclass) or 'binary_label' (for binary)
    - Optional 'split': 'train', 'val', or 'test'
    - Optional demographic/fairness metadata (e.g. 'skin_tone', 'fitzpatrick', 'age', 'sex')
    """

    def __init__(
        self,
        metadata: Union[pd.DataFrame, str, Path],
        image_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        split: Optional[str] = None,
        label_col: str = "label",
        image_col: str = "image",
        subgroup_col: Optional[str] = "skin_tone",
    ):
        """Initialize SkinLesionDataset.

        Args:
            metadata: Path to metadata CSV or pre-loaded pandas DataFrame.
            image_dir: Base directory where image files are stored.
            transform: Albumentations or torchvision image transform pipeline.
            split: Filter dataset by split ('train', 'val', 'test') if split column exists.
            label_col: Column name containing integer class target.
            image_col: Column name containing image filename or relative path.
            subgroup_col: Column name for skin tone or demographic subgroup.
        """
        self.image_dir = Path(image_dir).resolve()
        self.transform = transform
        self.label_col = label_col
        self.image_col = image_col
        self.subgroup_col = subgroup_col

        if isinstance(metadata, (str, Path)):
            meta_path = Path(metadata).resolve()
            if not meta_path.is_file():
                raise FileNotFoundError(f"Metadata file not found: {meta_path}")
            self.df = pd.read_csv(meta_path)
        elif isinstance(metadata, pd.DataFrame):
            self.df = metadata.copy()
        else:
            raise TypeError(f"metadata must be DataFrame or Path, got {type(metadata)}")

        # Filter by split if requested
        if split is not None and "split" in self.df.columns:
            self.df = self.df[self.df["split"].astype(str).str.lower() == str(split).lower()].reset_index(drop=True)

        self.df = self.df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        """Retrieve sample at given index.

        Returns:
            Tuple of (image_tensor, label, metadata_dict).
        """
        row = self.df.iloc[idx]
        img_name = str(row[self.image_col])

        # Resolve image path
        img_path = self.image_dir / img_name
        if not img_path.exists():
            # Try searching with common extensions if extension missing
            for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG"]:
                if (self.image_dir / f"{img_name}{ext}").exists():
                    img_path = self.image_dir / f"{img_name}{ext}"
                    break

        if not img_path.is_file():
            raise FileNotFoundError(f"Image file not found: {img_path}")

        # Load RGB image
        image = Image.open(img_path).convert("RGB")

        # Extract label
        label = int(row[self.label_col]) if self.label_col in row else -1

        # Extract subgroup metadata
        subgroup = str(row[self.subgroup_col]) if (self.subgroup_col and self.subgroup_col in row and pd.notna(row[self.subgroup_col])) else "Unknown"

        sample_metadata = {
            "image_id": img_name,
            "skin_tone_group": subgroup,
            "index": idx,
        }

        # Apply transformation
        if self.transform is not None:
            image = self.transform(image)
        else:
            # Fallback PIL to Tensor conversion
            from torchvision.transforms.functional import to_tensor
            image = to_tensor(image)

        return image, label, sample_metadata
