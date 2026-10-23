# SkinCancerTracker

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests Passing](https://img.shields.io/badge/tests-77%20passed-brightgreen.svg)](tests/)

A modular, reproducible deep learning pipeline for skin lesion classification and foundation model evaluation on the MILK10k dataset, featuring lesion-level stratified splitting, demographic fairness auditing across skin tone subgroups (`skin_tone_class` 0–5), and paired bootstrap statistical testing.

---

## 1. Project Overview & Verified Benchmark Results

`SkinCancerTracker` benchmarks dermatology computer vision models on **MILK10k** under a patient- and lesion-level stratified partition (SHA-256: `a4346dcf4d91a8bf6382a26c42618be327975de7f17596ce1fddfbcf1e160c36`) evaluated across two seeds per arm (42, 43); a third seed was not run:
- **Total Dataset**: 5,240 unique lesions (10,480 images; paired dermoscopic + clinical close-up).
- **Split Distribution (verified against `data/splits/split_summary.json`)**:
  - Train: 3,668 lesions / 7,336 images (70.0%)
  - Validation: 786 lesions / 1,572 images (15.0%)
  - Test: 786 lesions / 1,572 images (15.0%)
- **Benchmark Modality**: Dermoscopic images (exactly 786 held-out test lesions evaluated, 1-to-1 bijection, zero leakage).

### Official Benchmark Ablation Table (Held-Out Test Split, $N=786$)

All numbers below are verifiable in [`results/ablation_table.csv`](results/ablation_table.csv), [`results/manifest.json`](results/manifest.json), and [`results/MANIFEST.md`](results/MANIFEST.md):

| Experiment | Model Architecture | Training Strategy | Accuracy | Macro-F1 | Malignant Recall | Malignant AUROC | Macro AUROC | Best Val Epoch |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Exp-1 (seed 42)** | ResNet-50 | Class-balanced CE, Uniform | 0.7023 | 0.4158 | 0.8834 | 0.8179 | 0.8440 | Epoch 29 |
| **Exp-1 (seed 43)** | ResNet-50 | Class-balanced CE, Uniform | 0.6730 | 0.4015 | 0.8731 | 0.8222 | 0.8249 | Epoch 19 |
| **Exp-1 mean ($n=2$)** | ResNet-50 | Class-balanced CE, Uniform | 0.6877 $\pm$ 0.0207 | 0.4086 $\pm$ 0.0101 | 0.8782 $\pm$ 0.0073 | 0.8200 $\pm$ 0.0030 | 0.8344 $\pm$ 0.0135 | — |
| **Exp-2 (v1)** | PanDerm ViT-Base | Uniform, Standard CE (collapsed) | 0.5344 | 0.1091 | 0.8165 | 0.6699 | 0.6253 | — |
| **Exp-2 (v2)** | PanDerm ViT-Base | WeightedRandomSampler, LS 0.1 | 0.6438 | 0.3925 | 0.8902 | 0.8022 | 0.7547 | Epoch 26 |
| **Exp-3a** | PanDerm ViT-Base | Post-hoc LA ($\tau^*=0.10$) on Exp-2 v2 | 0.6361 | 0.3883 | 0.8816 | 0.7964 | 0.7403 | — |
| **Exp-2c (seed 42)** | PanDerm ViT-Base | Uniform shuffle, LS 0.1 | 0.7214 | 0.4690 | 0.8885 | 0.8418 | 0.8368 | Epoch 27 |
| **Exp-2c (seed 43)** | PanDerm ViT-Base | Uniform shuffle, LS 0.1 | 0.6858 | 0.3636 | 0.8799 | 0.8677 | 0.8743 | Epoch 14 |
| **Exp-2c mean ($n=2$)** | PanDerm ViT-Base | Uniform shuffle, LS 0.1 | **0.7036 $\pm$ 0.0252** | **0.4163 $\pm$ 0.0746** | **0.8842 $\pm$ 0.0061** | **0.8547 $\pm$ 0.0183** | **0.8555 $\pm$ 0.0265** | — |
| **Exp-3b (seed 42)** | PanDerm ViT-Base | Train-Time LA ($\tau=1.0$), LS 0.1 | 0.2290 | 0.3667 | 0.9417 | 0.7596 | 0.7850 | Epoch 25 |
| **Exp-3b (seed 43)** | PanDerm ViT-Base | Train-Time LA ($\tau=1.0$), LS 0.1 | 0.2379 | 0.3653 | 0.9280 | 0.7753 | 0.8084 | Epoch 25 |
| **Exp-3b mean ($n=2$)** | PanDerm ViT-Base | Train-Time LA ($\tau=1.0$), LS 0.1 | 0.2335 $\pm$ 0.0063 | 0.3660 $\pm$ 0.0009 | 0.9348 $\pm$ 0.0097 | 0.7675 $\pm$ 0.0111 | 0.7967 $\pm$ 0.0165 | — |

*Malignant classes evaluated: AKIEC, BCC, MAL_OTH, MEL, SCCKA.*  
*Recipe selection rule*: Phase 4 base recipe selected strictly by validation Macro-F1 (**Exp-2c**, mean val Macro-F1 = `0.5064`).

---

## 2. Key Statistical Findings (Paired Bootstrap, 2,000 Resamples)

Evaluated in [`results/analysis/phase2_phase3_paired_bootstrap.json`](results/analysis/phase2_phase3_paired_bootstrap.json):

1. **PanDerm ViT-Base vs ResNet-50 (Exp-2c minus Exp-1)**:
   - Pooled Macro-F1 difference: **+0.0077** (95% CI `[-0.0525, 0.0627]`, $P(\Delta > 0) = 0.5925$).
   - The 95% confidence interval crosses zero, indicating that the 95% bootstrap CI does not exclude 0 for Macro-F1 between the two architectures under uniform sampling on $N=786$.
2. **Train-Time Logit Adjustment vs Uniform Control (Exp-3b minus Exp-2c)**:
   - Pooled Macro-F1 difference: **-0.0503** (95% CI `[-0.1118, -0.0061]`, $P(\Delta > 0) = 0.0140$). The Exp-3b minus Exp-2c pooled macro-F1 CI is `[-0.1118, -0.0061]` (95% bootstrap CI excludes 0), while seed 43 alone includes 0 (`[-0.0823, 0.0598]`).
   - Pooled Accuracy difference: **-0.4701** (95% CI `[-0.5064, -0.4338]`, $P(\Delta > 0) = 0.0000$, 95% bootstrap CI excludes 0).
   - Pooled Malignant Recall difference: **+0.0506** (95% CI `[0.0335, 0.0689]`, $P(\Delta > 0) = 1.0000$, 95% bootstrap CI excludes 0).
   - Train-time logit adjustment with fixed $\tau=1.0$ causes Basal Cell Carcinoma recall to drop to **0.0000** (0/391 correctly predicted), collapsing accuracy to ~23%.
3. **Post-Hoc Logit Adjustment Validation Sweep**:
   - Sweeping $\tau \in [0.0, 1.5]$ on the validation split yielded optimal $\tau^*=0.00$ (seed 42) and $\tau^*=0.10$ (seed 43), yielding negligible change on the test set (+0.0010 Macro-F1 on seed 43), confirming a null effect.

---

## 3. Repository Structure

```text
SkinCancerTracker/
├── src/
│   ├── data/                     # Dataset, DataLoaders, transforms, split validation
│   ├── models/                   # Architecture wrappers (PanDerm ViT, ResNet-50)
│   ├── losses/                   # Loss functions (CE, Label Smoothing, Logit Adjustment, Hierarchical, TSC)
│   ├── train/                    # Training loop, validation engine, checkpointing
│   ├── eval/                     # Metrics calculation, predictions export, W&B logger
│   └── utils/                    # Reproducibility manifest, seed control, YAML config loader
├── configs/
│   ├── label_mapping.json        # 11 diagnostic classes and index mapping
│   └── experiments/              # Experiment configurations for local and Kaggle runs
├── kaggle/                       # Standalone Kaggle execution scripts (Exp-1, Exp-2c, Exp-3b, Exp-4a)
├── notebooks/                    # Kaggle execution notebooks and exploratory data analysis
├── scripts/                      # Utility scripts (ablation table generator, bootstrap analysis, upload packaging)
├── results/
│   ├── ablation_table.csv        # Comprehensive benchmark ablation table
│   ├── ablation_table.md         # Markdown ablation table
│   ├── MANIFEST.md               # Artifact manifest, file sizes, SHA-256 checksums, and caveats
│   ├── manifest.json             # Machine-readable artifact manifest
│   └── analysis/                 # Detailed reports for paired bootstrap, post-hoc sweeps, and final summary
├── tests/                        # 77 unit tests covering data, models, losses, and Kaggle script AST calls
├── docs/                         # Phase technical reports and mathematical references
├── requirements.txt              # Pinned Python package dependencies
├── environment.yml               # Conda environment definition
├── pyproject.toml                # Package configuration for editable installation
└── README.md                     # Project documentation
```

---

## 4. Installation & Development

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

## 5. Running Tests

Run the complete 77-test suite with `pytest`:

```bash
pytest
```

To run with verbose output:

```bash
pytest -v
```

---

## 6. Primary Reports and Documentation

- **Final Summary Report**: [`results/analysis/FINAL_SUMMARY.md`](results/analysis/FINAL_SUMMARY.md)
- **Artifact Manifest & Caveats**: [`results/MANIFEST.md`](results/MANIFEST.md)
- **Paired Bootstrap Analysis**: [`results/analysis/phase2_phase3_paired_bootstrap_report.md`](results/analysis/phase2_phase3_paired_bootstrap_report.md)
- **Post-Hoc Logit Adjustment**: [`results/analysis/exp2c_posthoc_la_report.md`](results/analysis/exp2c_posthoc_la_report.md)
- **Phase 2 Baseline Experiments**: [`docs/phase2_baseline_experiments.md`](docs/phase2_baseline_experiments.md)
- **Phase 3 Logit Adjustment**: [`docs/phase3_logit_adjustment.md`](docs/phase3_logit_adjustment.md)
- **Mathematical Formulations Reference**: [`docs/technical_reference/ALL_FORMULAS_AND_MATHEMATICS.md`](docs/technical_reference/ALL_FORMULAS_AND_MATHEMATICS.md)

---

## 7. Known Caveats & Methodological Notes

1. **Recent PyTorch Checkpoint Loading**:
   Exp-2c seed 42 post-training evaluation was re-evaluated from `best_model.pt` with `weights_only=False` because recent PyTorch (weights_only=True default) rejected numpy scalar objects saved in checkpoint dictionaries in `torch.load()`.
2. **W&B Logging**:
   Batch training runs were offline because the WANDB_API_KEY secret was not attached to the Kaggle notebook and the scripts default to offline unless --online is passed.
3. **Random Seeds**:
   Due to GPU compute quota allocations, runs used two seeds per arm (42, 43); a third seed was not run.
4. **Significance Criteria (95% Bootstrap CI Excludes 0)**:
   Differences are reported as significant when the 95% bootstrap CI excludes 0. For paired bootstrap testing between Exp-3b and Exp-2c, the Exp-3b minus Exp-2c pooled macro-F1 CI is `[-0.1118, -0.0061]` (95% bootstrap CI excludes 0), while seed 43 alone includes 0 (`[-0.0823, 0.0598]`).
