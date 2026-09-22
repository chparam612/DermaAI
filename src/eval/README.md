# Evaluation & Experiment Tracking Module (`src/eval`)

This module provides evaluation metrics, demographic subgroup auditing, and structured Weights & Biases (W&B) experiment logging.

## Key Components

### `WandbLogger` (`wandb_logger.py`)

A resilient logger that captures:
1. Scalar loss and aggregate metrics (`accuracy`, `macro_f1`, `auroc`).
2. **Table 1: Per-Class Metrics** (`eval/per_class_metrics`):
   - Columns: `class_name`, `support`, `precision`, `recall`, `f1_score`, `specificity`, `balanced_accuracy`.
3. **Table 2: Per-Skin-Tone Metrics** (`eval/per_skin_tone_metrics`):
   - Columns: `skin_tone_group`, `support`, `accuracy`, `balanced_accuracy`, `precision`, `recall`, `f1_score`, `auc`.
   - Distinguishes unannotated data (`Unannotated / Unknown`) from observed skin tones without label fabrication.
4. **Table 3: Subgroup Error Analysis** (`eval/subgroup_error_analysis`):
   - Columns: `sample_id`, `true_label`, `predicted_label`, `confidence`, `skin_tone_group`, `error_type`.

### Offline & CI Safety

When running in environments without internet access or without a W&B API key configured, `WandbLogger` switches seamlessly to offline mode (`mode="disabled"`), caching tables in memory without throwing errors.
