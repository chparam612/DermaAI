"""Dataset classes for Skin Lesion and Dermatology Image Analysis.

Provides modular PyTorch Dataset implementations supporting multi-class,
binary diagnosis, and subgroup/skin-tone metadata preservation for Phase 2
baseline experiments.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


DEFAULT_CLASSES = [
    "AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF",
    "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"
]
DEFAULT_CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(DEFAULT_CLASSES)}


def load_label_mapping(mapping_path: Optional[Union[str, Path]] = None) -> Dict[str, int]:
    """Load class-to-index mapping from json file or return defaults.

    Args:
        mapping_path: Path to label_mapping.json. If None, uses configs/label_mapping.json.

    Returns:
        Dictionary mapping class name to integer index.
    """
    if mapping_path is None:
        default_path = Path("configs/label_mapping.json")
        if default_path.is_file():
            mapping_path = default_path

    if mapping_path is not None and Path(mapping_path).is_file():
        with open(mapping_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("class_to_idx", DEFAULT_CLASS_TO_IDX)
    return DEFAULT_CLASS_TO_IDX


class SkinLesionDataset(Dataset):
    """Modular PyTorch Dataset for dermatology image classification.

    Features:
    - Reads from Phase 1 lesion-level splits (or arbitrary metadata).
    - Preserves lesion_id, isic_id, and skin_tone_class for fair subgroup evaluation.
    - Safely loads images in RGB format.
    - Supports train, validation, and test splits.
    """

    def __init__(
        self,
        metadata: Union[pd.DataFrame, str, Path] = "data/splits/lesion_level_splits.csv",
        image_dir: Union[str, Path] = "data/images",
        transform: Optional[Callable] = None,
        split: Optional[str] = None,
        label_col: str = "label",
        diagnosis_col: str = "diagnosis",
        image_col: str = "isic_id",
        subgroup_col: str = "skin_tone_class",
        label_mapping: Optional[Union[str, Path, Dict[str, int]]] = None,
    ):
        """Initialize SkinLesionDataset.

        Args:
            metadata: Path to metadata CSV or pre-loaded pandas DataFrame.
            image_dir: Base directory where image files are stored.
            transform: Image transform pipeline (torchvision/albumentations).
            split: Filter dataset by split ('train', 'val'/'validation', 'test').
            label_col: Column name containing integer class target.
            diagnosis_col: Column name containing diagnostic string label.
            image_col: Column name containing image identifier.
            subgroup_col: Column name for skin tone or demographic subgroup.
            label_mapping: Path to label mapping json or dictionary mapping class name to int.
        """
        self.image_dir = Path(image_dir).resolve()
        self.transform = transform
        self.label_col = label_col
        self.diagnosis_col = diagnosis_col
        self.image_col = image_col
        self.subgroup_col = subgroup_col

        if isinstance(label_mapping, dict):
            self.class_to_idx = label_mapping
        else:
            self.class_to_idx = load_label_mapping(label_mapping)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        if isinstance(metadata, (str, Path)):
            meta_path = Path(metadata).resolve()
            if not meta_path.is_file():
                raise FileNotFoundError(f"Metadata file not found: {meta_path}")
            self.df = pd.read_csv(meta_path)
        elif isinstance(metadata, pd.DataFrame):
            self.df = metadata.copy()
        else:
            raise TypeError(f"metadata must be DataFrame or Path, got {type(metadata)}")

        # Normalize split filtering ('val' matches 'validation')
        if split is not None and "split" in self.df.columns:
            target_split = str(split).strip().lower()
            if target_split in ("val", "validation"):
                mask = self.df["split"].astype(str).str.lower().isin(["val", "validation"])
            else:
                mask = self.df["split"].astype(str).str.lower() == target_split
            self.df = self.df[mask].reset_index(drop=True)

        # Ensure label column is populated
        if self.label_col not in self.df.columns:
            if self.diagnosis_col in self.df.columns:
                self.df[self.label_col] = self.df[self.diagnosis_col].map(self.class_to_idx).fillna(-1).astype(int)
            else:
                # Check for one-hot columns
                one_hot_cols = [c for c in self.class_to_idx.keys() if c in self.df.columns]
                if len(one_hot_cols) == len(self.class_to_idx):
                    self.df[self.label_col] = self.df[one_hot_cols].values.argmax(axis=1)
                else:
                    self.df[self.label_col] = -1

        self.df = self.df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def get_labels(self) -> List[int]:
        """Return list of integer labels for all samples in the dataset."""
        return self.df[self.label_col].tolist()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        """Retrieve sample at given index.

        Returns:
            Tuple of (image_tensor, label, metadata_dict).
        """
        row = self.df.iloc[idx]
        img_id = str(row[self.image_col])

        # Resolve image path (handle with or without .jpg extension)
        img_path = self.image_dir / img_id
        if not img_path.exists():
            for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG"]:
                candidate = self.image_dir / f"{img_id}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break

        if not img_path.is_file():
            raise FileNotFoundError(f"Image file not found: {img_path}")

        # Safely load RGB image
        try:
            with Image.open(img_path) as img:
                image = img.convert("RGB")
        except Exception as e:
            raise IOError(f"Failed to read image at {img_path}: {e}") from e

        # Extract label
        label = int(row[self.label_col])

        # Extract metadata
        lesion_id = str(row["lesion_id"]) if "lesion_id" in row else ""
        skin_tone = row[self.subgroup_col] if (self.subgroup_col in row and pd.notna(row[self.subgroup_col])) else -1
        try:
            skin_tone = int(skin_tone)
        except (ValueError, TypeError):
            pass

        sample_metadata = {
            "isic_id": img_id,
            "lesion_id": lesion_id,
            "skin_tone_class": skin_tone,
            "diagnosis": str(row.get(self.diagnosis_col, self.idx_to_class.get(label, "Unknown"))),
            "split": str(row.get("split", "")),
            "index": idx,
        }

        # Apply transformation
        if self.transform is not None:
            image = self.transform(image)
        else:
            from torchvision.transforms.functional import to_tensor
            image = to_tensor(image)

        return image, label, sample_metadata
