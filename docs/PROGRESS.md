# SkinCancerTracker Project Progress

## Session & Phase Overview
- **Project**: SkinCancerTracker (Multimodal Dermatology AI Benchmark on MILK10k)
- **Current State**: Phase 3 Logit Adjustment (Exp-3a Complete & Exp-3b Kaggle script ready).
- **Environment**: Local PC (CPU/RTX 2050 GPU for evaluation, tests, docs) + Kaggle GPU (T4/P100 16GB VRAM for heavy training).

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
- **Git Commit**: `b34c03e` ("Phase 2: Exp-2 PanDerm baseline evaluated and documented")

### Phase 3: Logit Adjustment for Extreme Class Imbalance & Fairness
- **Status**: Completed (Exp-3a Evaluated as NULL result; Exp-3b Kaggle Script Redesigned & Smoke-Tested)
- **Theoretical Basis**: Menon et al., *"Long-Tail Learning via Logit Adjustment"*, ICLR 2021 (arXiv:2007.07314).
- **Runs Completed & Evaluated**:
  1. **Exp-3a: Post-Hoc Logit Adjustment**:
     - Evaluated with frozen PanDerm ViT-Base (`best_model.pt`).
     - Hyperparameter $\tau$ tuned exclusively on validation split ($N=786$) sweeping $[0.0, 2.0]$; optimal $\tau^* = 0.100$ selected by `val_macro_f1` ($0.4117$, $+1.28\%$ over baseline).
     - Test metrics ($N=786$): Top-1 Acc: 0.6361, Balanced Acc: 0.3921, Macro-F1: 0.3883, Weighted F1: 0.6426, Malignant Recall: 0.8816, Malignant AUROC: 0.7964.
     - Paired 95% bootstrap CIs ($B=2,000$):
       - $\Delta \text{Macro-F1}$: $-0.0042$ $[-0.010, +0.001]$ (**NULL result**, CI spans 0).
       - $\Delta \text{Balanced Acc}$: $-0.0014$ $[-0.006, +0.004]$ (**NULL result**, CI spans 0).
       - Melanoma (`MEL`) Recall: $70.15\%$ vs $68.66\%$ ($+1.49\%$, exactly **ONE** flipped lesion out of 67; 95% CI $[+0.000, +0.030]$).
       - 3-Bucket Malignant FNR Disparity: numerical gap shifted from $6.06\%$ to $4.94\%$; **not statistically reliable** because Dark bucket has $n=12$ where a single flipped lesion moves FNR by $\sim 8.3\%$.
     - Diagnostic: Exp-2 v2 was fine-tuned using `WeightedRandomSampler` (balanced batches $\approx 1/11$ per class). Double-correction is supported by mechanism (WeightedRandomSampler confirmed) and BCC under-prediction; to be tested by controlled Exp-2c vs Exp-3b. Latent feature representation collapse is not established.
  2. **Exp-3b: Train-Time Logit Adjustment (Loss-Level)**:
     - Implemented `LogitAdjustedCrossEntropy` with label smoothing in `src/losses/classification.py`.
     - 6 unit tests passing in `tests/test_losses.py` (51 total repository tests passing).
     - Standalone Kaggle execution script `kaggle/run_exp3b_panderm_logit_adj.py` and configuration `configs/experiments/exp3b_panderm_logit_adj_kaggle.yaml`.
     - 1-batch smoke test gate verified passing with 0 errors.
- **Deliverables**:
  - `docs/phase3_logit_adjustment.md`
  - `docs/technical_reference/ALL_FORMULAS_AND_MATHEMATICS.md` (Formula 1.8)
  - `results/exp3a_panderm_posthoc/` (predictions_val.csv, predictions_test.csv, metrics.json, summary_metrics.json, tau_sweep_validation.json, paired_bootstrap_comparison.json)
  - `results/ablation_table.csv` & `results/ablation_table.md`
  - `src/losses/classification.py` & `src/losses/__init__.py`
  - `tests/test_losses.py`, `tests/test_raw_logits_evaluation.py`
  - `kaggle/run_exp3b_panderm_logit_adj.py` & `configs/experiments/exp3b_panderm_logit_adj_kaggle.yaml`

---

### Phase 4: Hierarchical Loss & Targeted Supervised Contrastive Learning (TSC) (In Progress)
- **Master Plan Order Decision**: Explicit decision confirmed to retain original MASTER_PLAN order:
  - **Phase 4**: Hierarchical Loss + Targeted Supervised Contrastive Learning (TSC)
  - **Phase 5**: MONET Concept Bottleneck Head & Demographic Fairness Auditing
  - **Phase 6**: Multimodal Dual-Image Fusion (Dermoscopic + Clinical Close-up)
- **Calibration Note**: Calibration (Expected Calibration Error / ECE, reliability diagrams) is a cross-cutting evaluation metric reported in every phase, not an independent phase.
- **Rationale**:
  1. The hierarchical loss module is already implemented, computationally cheap, and directly targets the 58 out of 64 missed malignancies that leak into benign attractor states (BKL/NV).
  2. Dual-image fusion on MILK10k already exists in prior literature; the novel methodological contribution of this research is the cost-sensitive loss, demographic fairness auditing, and calibration triage framework.
- **Phase 4.1 Specification (Asymmetric Hierarchical Cost Loss)**:
  - Implemented `AsymmetricHierarchicalCostLoss` and `build_asymmetric_cost_matrix` in `src/losses/hierarchical.py`.
  - Cost matrix entries ($c_{\text{fn}}=5.0, c_{\text{fp}}=1.5, c_{\text{within\_mal}}=1.0, c_{\text{within\_ben}}=0.5$) are explicitly treated as **unvalidated clinical hypotheses**.
  - Planned hyperparameter $\alpha$ sweep: $\alpha \in \{0.25, 0.5, 1.0, 2.0\}$ tuned strictly on the VALIDATION split ($N=786$).
  - Mandatory reporting contract for all Phase 4 runs:
    1. Malignant False Negative Rate (FNR) with 95% bootstrap CI.
    2. Malignant Specificity & False-Positive lesion count.
    3. Macro-F1 with paired bootstrap CI against the base recipe.
  - **Base Recipe Selection Protocol**: Phase 4 will build on the winning base recipe among {Exp-2 v2 (weighted sampler), Exp-2c (uniform shuffle), Exp-3b (LA loss)}, selected exclusively by validation Macro-F1 from Kaggle results. If tied within bootstrap CI, the simpler recipe will be chosen. No Phase 4 Kaggle script will be written until this selection is complete.
  - 7 unit tests verified passing in `tests/test_hierarchical_loss.py`.
- **Phase 4.2 Specification (Targeted Supervised Contrastive Learning / TSC)**:
  - Paper: Li et al., *"Targeted Supervised Contrastive Learning for Long-Tailed Recognition"*, **CVPR 2022** (arXiv:2111.13998).
  - Official Codebase: `https://github.com/LTH14/targeted-supcon` (MIT License verified).
  - Planning document: `docs/phase4_tsc_plan.md` (documents hypersphere target generation, pre-head normalized embedding attachment, exact loss formulation, and research uncertainties).
  - Implementation: `generate_uniform_hypersphere_targets` and `TargetedSupervisedContrastiveLoss` in `src/losses/tsc.py`.
  - 7 unit tests verified passing in `tests/test_tsc_loss.py` (verifying shape, gradient flow, unit norms, off-diagonal dot product near regular simplex $-0.10$, and reproducibility).
- **Immediate Kaggle Execution Sequence**:
  1. **Exp-1**: ResNet-50 Baseline (class_balanced_ce, uniform shuffle, 30 epochs, seed 42) — Est: ~0.4 GPU-hours.
  2. **Exp-2c (Seed 42)**: PanDerm ViT Control Arm (uniform shuffling + label smoothing 0.1, 30 epochs, seed 42) — Est: ~0.5 GPU-hours.
  3. **Exp-2c (Seed 43)**: PanDerm ViT Control Arm (uniform shuffling + label smoothing 0.1, 30 epochs, seed 43) — Est: ~0.5 GPU-hours.
  4. **Exp-3b**: PanDerm ViT Train-Time Logit Adjustment (uniform shuffling + LA loss tau=1.0 + label smoothing 0.1, 30 epochs, seed 42) — Est: ~0.5 GPU-hours.
  - **Total Kaggle GPU Allocation**: ~1.9 GPU-hours.
  - **Required Download Artifacts After Each Run**:
    - `best_model.pt` (checkpoint containing model weights, epoch, and validation metrics)
    - `metrics.json` (detailed per-class metrics, fairness metrics, confusion matrices)
    - `summary_metrics.json` (compact headline summary with seed and split counts)
    - `predictions_val.csv` (validation per-lesion raw logits, probabilities, and predictions)
    - `predictions_test.csv` (held-out test per-lesion raw logits, probabilities, and predictions)
    - `predictions.csv` (test predictions duplicate for backward compatibility)
- **Kaggle Execution Notebooks (Generated & Validated)**:
  - `notebooks/kaggle_exp1_resnet50.ipynb`: Standard CNN baseline (ResNet-50 ImageNet-pretrained, uniform shuffle, class-balanced CE, 30 epochs).
  - `notebooks/kaggle_exp2c_panderm_uniform.ipynb`: PanDerm ViT-Base control arm (uniform shuffling + label smoothing 0.1, 30 epochs). Supports seed override (seed 42 and seed 43).
  - `notebooks/kaggle_exp3b_panderm_logit_adj.ipynb`: PanDerm ViT-Base train-time logit adjustment (tau=1.0, uniform shuffling, label smoothing 0.1, 30 epochs).
  - **Gates Implemented in Every Notebook**:
    1. GPU hardware availability gate (asserts CUDA present and VRAM >= 14GB).
    2. Split SHA-256 hash gate (asserts exact hash `a4346dcf4d91a8bf6382a26c42618be327975de7f17596ce1fddfbcf1e160c36`).
    3. Architecture and checkpoint gate (for PanDerm notebooks: official VisionTransformer, zero unexpected keys, only head keys missing, no fallback to timm; for Exp-1: clean timm ResNet-50 instantiation).
  - **Post-Training Seed & Packaging Verification**:
    - Reads back `metrics.json`, verifies recorded seed equals `SEED` variable.
    - Prints summary evaluation metrics and per-class performance breakdown.
    - Packages only evaluation artifacts into `{exp_name}_artifacts.zip` (`predictions_val.csv`, `predictions_test.csv`, `metrics.json`, `summary_metrics.json`, `training_curves.json`) without `best_model.pt`.
  - **Packaging Integrity**:
    - Updated `scripts/prepare_kaggle_upload.py` to enforce presence of all 3 Kaggle execution scripts and configs in `codebase.zip`.
    - Package integrity verified: all 72 repository unit tests pass inside the extracted package in an isolated temporary directory.


---

### Phase 6-lite: Demo Inference Backend & API Service
- **Status**: In Progress (Gate 1 Done)
- **Goal**: Demo-ready, honest FastAPI inference service for mentor/guide presentation.
- **Demo Model**: PanDerm ViT-Base (checkpoints/exp2c_panderm_uniform_seed42/best_model.pt).
- **Gate 1 Deliverables (Checkpoint Slimming & Parity Check)**:
  - Exported slim weights: checkpoints/final/demo_model.pt (343,331,341 bytes, SHA-256: 137c7991202448e3a2c45b7f93dca680d44e05f3b8321882840514219c613b3d).
  - Strict CPU loading with torch.load(..., weights_only=True, map_location='cpu') (load time: 4.0518s, missing keys: 0, unexpected keys: 0).
  - Architecture: PanDermClassifier (panderm_base, 11 diagnostic classes).
  - Preprocessing: src.data.transforms.get_panderm_eval_transforms (image_size=224, Resize(256), CenterCrop(224), Mean=[0.485, 0.456, 0.406], Std=[0.228, 0.224, 0.225]).
  - Parity verification on N=50 stratified test lesions across all 11 classes (results/audit/gate1_parity_report.json):
    - Max absolute probability difference: 0.004336 (Threshold <= 0.01)
    - Mean absolute probability difference: 0.000072
    - Argmax disagreements: 0 / 50 (Threshold <= 2)
    - Status: **PASSED**
  - CPU Benchmarking (20 timed forward passes after 5 warm-ups):
    - Per-image latency: Median 234.13 ms (Mean 290.02 ms, Min 210.12 ms, Max 479.44 ms)
    - Process RAM: 388.4 MB pre-load, 1454.1 MB post-load, 1479.7 MB peak RSS.
  - **Gate 2 Deliverables (Attention Rollout Explainability)**:
    - Module: api/explain.py (AttentionRollout class).
    - Architecture compatibility: PanDerm VisionTransformer without modifying external/panderm.
    - Captures per-layer attention across all 12 blocks via dynamic non-invasive forward interception; handles CLS token and layer-scale.
    - Properties: Output shape (14, 14) spatial patch grid, min-max normalized to [0.0, 1.0], deterministic (diff = 0.0).
    - CPU Latency (N=10): Median 216.11 ms, Mean 220.74 ms.
    - Output format: RGBA PNG Base64 Data URI (data:image/png;base64,...) with transparent background.
    - 5 visual audit examples rendered to results/audit/rollout_examples/ (BCC, MEL, SCCKA, NV, BKL).
    - Unit tests: 4 passing in tests/test_explain.py.
  - **Gate 3 Deliverables (FastAPI Backend Service)**:
    - Package: api/ (app.py, config.py, schemas.py, inference.py, explain.py).
    - Endpoints implemented:
      - GET /api/health -> status, model_version, device, checkpoint sha256.
      - GET /api/model-info -> model_version, 11 classes, malignant classes, provenance, limitations, disclaimer.
      - POST /api/predict -> predicted_class, confidence, malignant_probability, per_class_probs, concept_scores (null), explainability_overlay (data URI), model_version, disclaimer, inference_ms, input_note.
    - Requirements: In-memory image validation with PIL; max 10MB; JPEG/PNG strictly checked; CORSMiddleware with env-configured origins; disclaimer on every response; zero working-directory file leaks.
    - Test Suite: 9 tests passing in tests/test_api.py (all error cases, probability sum = 1, in-memory purity).
    - Smoke Test Script: scripts/demo_api_smoke.py verified on 3 test lesions (BCC, MEL, NV).
    - Contract Diff: Documented in results/audit/api_contract_diff.md vs frontend types.ts.
  - **Gate 4 Deliverables (Frontend F4 Panel Metrics Summary)**:
    - Generated: results/frontend_report_summary.json (source-traceable to results/exp2c_panderm_uniform_seed42/exp2c_panderm_uniform_artifacts/summary_metrics.json and predictions_test.csv).
    - Overall Test Metrics: Macro-F1: 0.4690, Top-1 Acc: 0.7214, Balanced Acc: 0.4724, Weighted F1: 0.7131, Macro AUROC: 0.8368.
    - Malignant vs Benign: Sensitivity: 0.8885, Specificity: 0.6552, FN: 65, FP: 70, TP: 518, TN: 133.
    - Per-class test recall/F1 across all 11 classes with support.
    - Skin-tone subgroups: Tone 1 (n=12, low_confidence=true), Tone 2 (n=72), Tone 3 (n=486), Tone 4 (n=153), Tone 5 (n=63).
    - Honest Caveats Embedded: Single-run provenance, seed 43 variance (|Delta F1| = 0.1054), internal split limitations, tail class supports ({'BEN_OTH': 8, 'DF': 5, 'INF': 5, 'MAL_OTH': 2, 'VASC': 6}), and benchmark incomparability with published 53.6%.
    - Test Suite: tests/test_frontend_report.py passing.
