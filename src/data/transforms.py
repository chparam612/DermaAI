"""Data transformations and augmentations for dermatology medical imaging.

Supports ImageNet standard preprocessing for Exp-1 (CNN baseline) and
official PanDerm preprocessing for Exp-2.
"""

from __future__ import annotations

from typing import Tuple, Union
from torchvision import transforms


# Standard ImageNet normalization parameters (Exp-1)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# PanDerm official normalization parameters (Exp-2)
# Note: std[0] is 0.228 instead of 0.229 per external/panderm/classification/run_class_finetuning.py
PANDERM_MEAN = [0.485, 0.456, 0.406]
PANDERM_STD = [0.228, 0.224, 0.225]


def get_train_transforms(
    image_size: Union[int, Tuple[int, int]] = 224,
    mean: Tuple[float, float, float] = tuple(IMAGENET_MEAN),
    std: Tuple[float, float, float] = tuple(IMAGENET_STD),
) -> transforms.Compose:
    """Standard dermatology data augmentation pipeline for Exp-1 baseline model.

    Includes resizing, horizontal/vertical flips, affine rotation,
    color jitter (brightness/contrast/saturation), and normalization.

    Args:
        image_size: Output image dimension (int or (height, width)).
        mean: Normalization channel means.
        std: Normalization channel standard deviations.

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    size = (image_size, image_size) if isinstance(image_size, int) else image_size
    return transforms.Compose([
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_eval_transforms(
    image_size: Union[int, Tuple[int, int]] = 224,
    mean: Tuple[float, float, float] = tuple(IMAGENET_MEAN),
    std: Tuple[float, float, float] = tuple(IMAGENET_STD),
) -> transforms.Compose:
    """Deterministic preprocessing pipeline for validation and evaluation.

    Performs resizing, tensor conversion, and normalization.

    Args:
        image_size: Target image dimension.
        mean: Normalization channel means.
        std: Normalization channel standard deviations.

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    size = (image_size, image_size) if isinstance(image_size, int) else image_size
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_panderm_train_transforms(
    image_size: int = 224,
    resize_size: int = 256,
) -> transforms.Compose:
    """Official PanDerm training augmentation pipeline from run_class_finetuning.py.

    Includes Resize(256), RandomResizedCrop(224, scale=(0.75, 1.0)),
    RandomHorizontalFlip, RandomVerticalFlip, RandomRotation(45),
    ColorJitter(hue=0.2), and PanDerm normalization.
    """
    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=45),
        transforms.ColorJitter(hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=PANDERM_MEAN, std=PANDERM_STD),
    ])


def get_panderm_eval_transforms(
    image_size: int = 224,
    resize_size: int = 256,
) -> transforms.Compose:
    """Official PanDerm evaluation transform pipeline from run_class_finetuning.py.

    Includes Resize(256), CenterCrop(224), ToTensor, and PanDerm normalization.
    """
    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=PANDERM_MEAN, std=PANDERM_STD),
    ])


def get_transforms(
    pipeline_type: str = "baseline",
    mode: str = "train",
    image_size: int = 224,
) -> transforms.Compose:
    """Unified transform factory.

    Args:
        pipeline_type: 'baseline' / 'resnet' (Exp-1) or 'panderm' (Exp-2).
        mode: 'train', 'val', 'validation', or 'test'.
        image_size: Spatial target resolution.

    Returns:
        torchvision.transforms.Compose
    """
    is_train = mode.strip().lower() == "train"
    pipeline_type = pipeline_type.strip().lower()

    if pipeline_type in ("panderm", "exp2", "vit"):
        if is_train:
            return get_panderm_train_transforms(image_size=image_size)
        return get_panderm_eval_transforms(image_size=image_size)

    # Standard CNN baseline
    if is_train:
        return get_train_transforms(image_size=image_size)
    return get_eval_transforms(image_size=image_size)
