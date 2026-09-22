"""Data transformations and augmentations for dermatology medical imaging."""

from __future__ import annotations

from typing import Tuple, Union
from torchvision import transforms


# Standard ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(
    image_size: Union[int, Tuple[int, int]] = 224,
    mean: Tuple[float, float, float] = tuple(IMAGENET_MEAN),
    std: Tuple[float, float, float] = tuple(IMAGENET_STD),
) -> transforms.Compose:
    """Standard dermatology data augmentation pipeline for model training.

    Includes random resizing, random horizontal/vertical flips, affine rotation,
    color jitter (brightness/contrast adjustment), and normalization.

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
