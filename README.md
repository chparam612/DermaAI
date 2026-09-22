# SkinCancerTracker

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Weights & Biases](https://img.shields.io/badge/Tracking-W%26B-orange)](https://wandb.ai/)

A modular, reproducible, research-grade deep learning pipeline for dermatology AI, skin lesion classification, and foundation model fine-tuning with integrated subgroup fairness evaluation.

---

## 1. Project Overview

`SkinCancerTracker` provides a standardized framework for training, evaluating, and probing dermatology computer vision models on benchmark datasets (such as ISIC, HAM10000, and PAD-UFES-20). The architecture integrates cutting-edge dermatology foundation models (**PanDerm** and **DermFM-Zero**) alongside traditional baselines, placing rigorous emphasis on:
- **Reproducibility**: Deterministic seeding, hardware profiling, and experiment manifest capture.
- **Fairness & Demographic Auditing**: Disaggregated per-skin-tone evaluation (Fitzpatrick phototypes) and sample-level error analysis.
- **Structured Experiment Tracking**: Multi-dimensional Weights & Biases (W&B) logging including custom per-class and per-subgroup tables.

---

## 2. Repository Structure

```text
SkinCancerTracker/
├── src/
│   ├── __init__.py               # Package metadata and version info
│   ├── data/
│   │   ├── __init__.py           # Data exports
│   │   ├── datasets.py           # SkinLesionDataset with subgroup tracking
│   │   ├── transforms.py         # Training augmentations & eval preprocessing
│   │   ├── samplers.py           # Class-balanced sampling utilities
│   │   └── README.md
│   ├── models/
│   │   ├── __init__.py           # Backbone model factory stubs
│   │   └── README.md
│   ├── losses/
│   │   ├── __init__.py           # Objective loss functions
│   │   └── README.md
│   ├── train/
│   │   ├── __init__.py           # Training loops & validation logic
│   │   └── README.md
│   ├── eval/
│   │   ├── __init__.py           # Evaluation modules
│   │   ├── wandb_logger.py       # W&B logger with Table 1, 2, and 3
│   │   └── README.md
│   └── utils/
│       ├── __init__.py           # Utility exports
│       ├── config.py             # YAML loader, schema validator, CLI parser
│       ├── reproducibility.py    # Seed setting, Git hash, hardware info
│       └── README.md
├── configs/
│   ├── README.md                 # Configuration documentation & §9 notes
│   └── baseline.yaml             # Baseline experiment configuration
├── notebooks/
│   └── README.md                 # Jupyter exploration & EDA directory
├── scripts/
│   └── README.md                 # CLI training & evaluation entry points
├── tests/
│   ├── __init__.py
│   ├── test_structure.py         # Directory & import verification
│   ├── test_config.py            # YAML schema & validation tests
│   ├── test_wandb_logger.py      # Offline W&B table unit tests
│   └── test_reproducibility.py   # Seeding and manifest tests
├── external/
│   ├── panderm/                  # Git Submodule: PanDerm (Nature Medicine 2025)
│   ├── dermfm-zero/              # Git Submodule: DermFM-Zero (Feb 2026)
│   └── README.md                 # External models documentation & licenses
├── .gitignore                    # Robust exclusion of datasets & checkpoints
├── .gitmodules                   # Submodule tracking for PanDerm & DermFM-Zero
├── .env.example                  # Environment variable template for secrets
├── requirements.txt              # Pinned Python package dependencies
├── environment.yml               # Conda environment definition
├── pyproject.toml                # Package configuration for editable installation
├── setup_colab.ipynb             # 10-step executable Google Colab setup
└── README.md                     # Project documentation
```

---

## 3. Installation Instructions

### Prerequisites
- Python 3.10 or higher (Python 3.10 recommended for Conda environments).
- Git with submodule support.
- NVIDIA GPU with CUDA 11.8+ or 12.1+ recommended for GPU acceleration.

### Option A: Local Conda Setup (Recommended)

```bash
# 1. Clone repository
git clone <repository-url>
cd SkinCancerTracker

# 2. Create and activate Conda environment
conda env create -f environment.yml
conda activate skincancertracker

# 3. Initialize Git submodules
git submodule update --init --recursive
```

### Option B: Local Pip / Virtualenv Setup

```bash
# 1. Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install PyTorch with CUDA acceleration (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install dependencies and editable package
pip install -r requirements.txt
pip install -e .

# 4. Initialize Git submodules
git submodule update --init --recursive
```

---

## 4. Local Development Setup

1. Copy the environment configuration template:
   ```bash
   cp .env.example .env
   ```
2. Configure credentials and paths in `.env` (e.g., `WANDB_API_KEY`, `DATASET_ROOT`).
3. Verify test suite execution:
   ```bash
   pytest
   ```

---

## 5. Google Colab Setup

For cloud training on free or premium Google Colab GPUs (T4, A100, V100):
1. Open `setup_colab.ipynb` in Google Colab.
2. Execute the notebook sequentially from Cell 1 through Cell 10.
3. The notebook automates:
   - Mounting Google Drive (`/content/drive`).
   - Repository access.
   - Hardware detection.
   - Dependency installation via `pip install -r requirements.txt`.
   - Editable package installation via `pip install -e .`.
   - Git submodule initialization (`git submodule update --init --recursive`).
   - Import validation and offline W&B verification.

---

## 6. Git Submodule Initialization

The repository tracks **PanDerm** and **DermFM-Zero** via Git submodules under `external/`:

```bash
git submodule update --init --recursive
```

Refer to [`external/README.md`](file:///external/README.md) for detailed descriptions of upstream papers, licenses, architectures, and weights download links.

---

## 7. Configuration System Usage

Experiments are configured via YAML files in `configs/`.

```python
from src.utils import load_config

# Load and validate configuration
config = load_config("configs/baseline.yaml")

print(config.dataset.name)           # Dot-accessible attribute
print(config["training"]["epochs"])  # Standard dict access
```

CLI argument parsing and hyperparameter overrides are supported:

```bash
python -m src.utils.config --config configs/baseline.yaml --epochs 30 --lr 0.0005 --offline
```

---

## 8. Weights & Biases (W&B) Setup

`SkinCancerTracker` includes a dedicated medical logger in `src/eval/wandb_logger.py`.

### Enabling W&B
1. Login to W&B:
   ```bash
   wandb login
   ```
2. In `configs/baseline.yaml`, set:
   ```yaml
   wandb:
     enabled: true
     project: "SkinCancerTracker"
     mode: "online"
   ```

### Offline & CI Mode
When running unit tests or working offline without an account, set:
```yaml
wandb:
  enabled: false
  mode: "disabled"
```
The logger will continue tracking metric history and table structures locally without throwing exceptions or requiring network calls.

### Custom Medical Evaluation Tables
The pipeline standardly generates three structured tables:
1. **Table 1: Per-Class Metrics** (`eval/per_class_metrics`):
   - Columns: `class_name`, `support`, `precision`, `recall`, `f1_score`, `specificity`, `balanced_accuracy`.
2. **Table 2: Per-Skin-Tone Metrics** (`eval/per_skin_tone_metrics`):
   - Columns: `skin_tone_group`, `support`, `accuracy`, `balanced_accuracy`, `precision`, `recall`, `f1_score`, `auc`.
   - Gracefully handles missing subgroup annotations by labelling them `Unannotated / Unknown` without fabricating labels or skewing denominators.
3. **Table 3: Subgroup Error Analysis** (`eval/subgroup_error_analysis`):
   - Columns: `sample_id`, `true_label`, `predicted_label`, `confidence`, `skin_tone_group`, `error_type`.

---

## 9. Reproducibility

To enforce determinism across PyTorch, NumPy, and Python:

```python
from src.utils import set_seed, capture_reproducibility_manifest

# Set deterministic seed
set_seed(42)

# Save snapshot of hardware, software dependencies, and Git state
manifest = capture_reproducibility_manifest(seed=42, output_path="outputs/manifest.json")
```

---

## 10. Running Tests

Run the complete test suite with `pytest`:

```bash
pytest
```

To run with verbose output:

```bash
pytest -v
```

---

## 11. Known Limitations & Unresolved Items

1. **Section §9 Specification Status**:
   - The formal experiment configuration table referenced in specification §9 was not present in the workspace.
   - Sensible baseline parameters are configured in `configs/baseline.yaml` and clearly designated with `[TODO §9: ...]`.
   - Exact benchmark splits, foundation model checkpoints, and target hyperparameter sweeps will be integrated once §9 is supplied.
2. **Section §14 Dependency Versions Status**:
   - Dependency versions from specification §14 were not present in the workspace.
   - Conservative, cross-compatible versions were established across `requirements.txt` and `environment.yml` based on verified upstream requirements for PanDerm and DermFM-Zero.
3. **Submodule Source Modifications**:
   - Neither PanDerm nor DermFM-Zero source code should be edited directly in `external/`. Upstream functionality will be wrapped via adapters in `src/models/` during Phase 1.
