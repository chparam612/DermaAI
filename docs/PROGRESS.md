# SkinCancerTracker Project Progress

## Session & Phase Overview
- **Project**: SkinCancerTracker (Multimodal Dermatology AI Benchmark on MILK10k)
- **Current State**: Phase 2 Close-Out Complete. Ready for Phase 3 (Logit Adjustment).
- **Environment**: Local PC (CPU-only for analysis, evaluation, docs, API, tests) + Kaggle GPU (T4/P100 16GB VRAM for heavy training).

---

## Phase Log

### Phase 0: Environment & Scaffold Setup
- **Status**: Completed
- **Deliverables**: Directory structure, virtual environment, `pyproject.toml`, `.gitignore`, configuration schemas.
- **Git Commit**: `106f4b8` ("Phase zero:Environment setup complete")

### Phase 1: Data Layer & Lesion-Level Stratified Splitting
- **Status**: Completed
- **Deliverables**:
  - Lesion-level stratified partition without patient/lesion leakage (`data/splits/lesion_level_splits.csv`).
  - Total: 5,240 lesions (10,480 images; paired dermoscopic + clinical close-up).
  - Train: 3,668 lesions (70%), Val: 786 lesions (15%), Test: 786 lesions (15%).
  - Splits file SHA-256: `a4346dcf4d91a8bf6382a26c42618be327975de7f17596ce1fddfbcf1e160c36`.
  - PyTorch Dataset, DataLoaders, transforms (`src/data/`).
  - 45 unit tests verified passing.
- **Git Commit**: `a83e965` ("Phase 1:Data Layer Implementation")

### Phase 2: Baseline Experiments (Exp-1 CNN & Exp-2 PanDerm ViT-Base)
- **Status**: Completed (Close-Out Documented)
- **Modality**: Dermoscopic image modality strictly isolated (1 image per lesion).
- **Runs Completed & Evaluated**:
  1. **Exp-1: ResNet-50 CNN Baseline**:
     - 1-epoch reference checkpoint locally (`checkpoints/exp1_baseline/best_model.pt`).
     - Test metrics (N=786): Acc: 0.5191, Balanced Acc: 0.3338, Macro-F1: 0.2869, Malignant Recall: 0.8868, Malignant AUROC: 0.7549.
     - Standalone 30-epoch Kaggle script created: `kaggle/run_exp1_resnet50.py` with config `configs/experiments/exp1_resnet50_kaggle.yaml`.
  2. **Exp-2 (v1): PanDerm ViT-Base (Initial Recipe)**:
     - Unweighted sampler, step LR schedule.
     - Test metrics (N=786): Acc: 0.5344, Balanced Acc: 0.2185, Macro-F1: 0.1091 (majority-class collapse onto BCC/NV).
  3. **Exp-2 (v2): PanDerm ViT-Base (Corrected Recipe)**:
     - Trained on Kaggle GPU (Tesla T4 16GB) in 29m 20s (1,760.5s).
     - Weighted random sampler, layer-wise LR decay (0.9), AdamW lr=2.5e-4, label smoothing 0.1, `val_macro_f1` checkpointing.
     - Best epoch: 26 / 30.
     - Test metrics (N=786): Top-1 Acc: 0.6438, Balanced Acc: 0.3934, Macro-F1: 0.3925, Weighted F1: 0.6469, Malignant Recall: 0.8902, Malignant AUROC: 0.8022.
- **Disaggregated Fairness & Safety Findings**:
  - Malignant Sensitivity: 89.02% (519 / 583 detected).
  - Malignant FNR (Critical Missed Cancers): 10.98% (64 false negatives). Missed malignancies collapse primarily into benign BKL (33 cases) and NV (20 cases).
  - 3-Bucket Fairness: Dark (Tones 0-1, N=12, LOW-CONFIDENCE, Mal FNR 16.67%), Medium (Tones 2-3, N=558, Mal FNR 10.60%), Light (Tones 4-5, N=216, Mal FNR 11.73%). 3-Bucket FNR Disparity: 6.06%.
  - Tone 3 Dissection (N=486, 61.8% of test set): Low Macro-F1 (0.2718) driven by extreme BCC dominance (53.50%) and 0.0000 recall across all 5 rare tail classes (`BEN_OTH`, `DF`, `INF`, `MAL_OTH`, `VASC`).
- **Overfitting Dynamics**:
  - Train Acc: 88.69% vs Val Acc: 62.09% (26.6% generalization gap).
  - Train Loss: 0.7589 vs Val Loss: 1.4394 (loss gap: +0.6805).
- **Deliverables**:
  - `docs/phase2_baseline_experiments.md`
  - `results/ablation_table.csv` & `results/ablation_table.md`
  - `scripts/analyze_exp2_results.py`
  - `kaggle/run_exp1_resnet50.py` & `configs/experiments/exp1_resnet50_kaggle.yaml`
- **Git Commit**: `b34c03e` ("Phase 2: Exp-2 PanDerm baseline evaluated and documented")
- **Next Action**: Phase 3 Logit Adjustment (Exp-3a post-hoc locally on CPU + Exp-3b train-time Kaggle script).
