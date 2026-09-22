# Losses Module (`src/losses`)

This module defines training objectives, loss functions, and penalty terms.

## Loss Functions

- **Cross Entropy (`cross_entropy`)**: Standard multi-class classification objective with optional class weighting.
- **BCE with Logits (`bce_with_logits`)**: Binary classification objective for malignant vs benign diagnosis.
- **Future Extensions (Phase 1+)**:
  - Class-balanced focal loss
  - Subgroup fairness penalty / minimax group loss to mitigate diagnostic disparities across Fitzpatrick skin types.

## Usage Example

```python
import torch
from src.losses import build_loss

criterion = build_loss("cross_entropy", class_weights=torch.tensor([1.0, 4.0]))
```
