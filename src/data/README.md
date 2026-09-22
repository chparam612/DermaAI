# Data Module (`src/data`)

This module manages dataset loading, metadata preservation, transformations, and sampling strategies for dermatology research.

## Files

- **`datasets.py`**:
  `SkinLesionDataset`: Modular PyTorch `Dataset` supporting binary and multiclass dermatology classification (e.g. melanoma vs nevus, HAM10000 7-class). Tracks sample IDs and fairness/subgroup annotations (`skin_tone`, `fitzpatrick`, `age`, `sex`) alongside images and labels.
- **`transforms.py`**:
  `get_train_transforms()` and `get_eval_transforms()`: Augmentation and preprocessing pipelines using `torchvision` (resizing, flips, rotations, color jitter, and ImageNet/dermatology normalization).
- **`samplers.py`**:
  `create_balanced_class_sampler()`: Weighted random sampler that balances sampling probability across severely skewed dermatology classes.

## Usage Example

```python
from src.data import SkinLesionDataset, get_train_transforms, create_balanced_class_sampler
from torch.utils.data import DataLoader

train_transform = get_train_transforms(image_size=224)
dataset = SkinLesionDataset(
    metadata="data/metadata.csv",
    image_dir="data/images",
    transform=train_transform,
    split="train",
    label_col="label",
    subgroup_col="skin_tone",
)

sampler = create_balanced_class_sampler(dataset.df["label"].values)
loader = DataLoader(dataset, batch_size=32, sampler=sampler, num_workers=4)
```
