# Experiment Configurations (`configs/`)

This directory contains YAML experiment configurations governing data splits, hyperparameters, models, and logging.

## Contents

- **`baseline.yaml`**: The foundational configuration file for standard training and evaluation pipelines.

## Specification Status: Section §9

The project specification mentions an experiment configuration table in §9. As this specification document was not present in the workspace, placeholder values with clear `[TODO §9: ...]` markers have been provided in `baseline.yaml`.

When §9 specifications are supplied:
1. Benchmark dataset names (e.g. HAM10000 vs ISIC2020 vs PAD-UFES-20) and target splits.
2. Architecture specifications (e.g. PanDerm ViT-B/16 vs ViT-L/16, DermFM-Zero).
3. Exact learning rates, batch sizes, and optimizer schedules.
