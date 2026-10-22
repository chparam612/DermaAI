# SkinCancerTracker Backend Status Audit

**Audit Timestamp**: 2026-10-22T09:55:00+05:30  
**Repository HEAD**: `13b98873b310ed84321e11cb2a2fdcdfe6758e32`  
**Working Tree Status**: Clean (only untracked evaluation artifact directories from completed Kaggle runs preserved)  
**Test Suite Status**: 77 / 77 passing (0 failures, 12 warnings in 23.36s)  
**Audit Policy**: Strict read-only verification; all tables copied directly from generated artifact outputs under `results/audit/`.

---

## 1. Overall Verdict

1. **Last phase fully DONE**: Phase 3 (Logit Adjustment: Exp-3a post-hoc NULL result evaluated; Exp-3b train-time LA evaluated across seeds 42 and 43 with confirmed BCC recall collapse).
2. **Prior completed phases**: Phase 0 (Scaffold & environment), Phase 1 (Data pipeline with verified split SHA-256 `a4346dcf...`), and Phase 2 (Baselines: Exp-1 ResNet-50 seeds 42/43, Exp-2 v1/v2, Exp-2c PanDerm uniform control seeds 42/43).
3. **Phase currently in progress**: Phase 4 (Hierarchical Cost Loss & Targeted Supervised Contrastive Learning); loss mathematics and 14 unit tests pass, but multi-epoch Kaggle training runs are NOT YET EXECUTED.
4. **Phases not started**: Phase 5 (MONET Concept Bottleneck Head & Fairness Auditing), Phase 6 (Multimodal Dual-Image Fusion & FastAPI Inference Service), and Phase 7 (Paper Artifacts & Submission Tables).
5. **Backend Readiness**: Backend baseline infrastructure and benchmark ablation are 100% verified; next critical dependency is executing the Phase 4.1 Kaggle alpha sweep on the Exp-2c base recipe.

---

## 2. Phase Status Table

*(Copied verbatim from [`results/audit/phase_status_table.md`](../results/audit/phase_status_table.md))*

| Phase | Status | Evidence on Disk | Remaining Gaps |
| :--- | :---: | :--- | :--- |
| **Phase 0: Scaffold & Setup** | **DONE** | Commit 106f4b8; pyproject.toml, requirements.txt, environment.yml, .gitignore verified on disk. | None. |
| **Phase 1: Data Pipeline** | **DONE** | lesion_level_splits.csv (SHA-256: a4346dcf4d91a8bf6382a26c42618be327975de7f17596ce1fddfbcf1e160c36 matches contract); label_mapping.json (11 classes); 13 data integrity & class-order tests passing (tests/test_splitting.py, test_data_integrity_and_class_order.py). | None. |
| **Phase 2: Baselines** | **DONE** | Exp-1 (ResNet-50 30ep, seeds 42 & 43 in results/exp1_resnet50_seed42/ & seed43/); Exp-2 v1 (results/exp2_panderm/); Exp-2 v2 (results/exp2_panderm_v2/); Exp-2c (PanDerm uniform shuffle, seeds 42 & 43 in results/exp2c_panderm_uniform_seed42/ & seed43/); MANIFEST.md, ablation_table.csv, FINAL_SUMMARY.md. | None. All planned baseline arms completed, audited across 2 seeds, and documented. |
| **Phase 3: Logit Adjustment** | **DONE** | Exp-3a post-hoc LA evaluated on validation sweep [0.0, 2.0] & test set (results/exp3a_panderm_posthoc/, documented as NULL result); Exp-3b train-time LA tau=1.0 evaluated across seeds 42 & 43 (results/exp3b_panderm_logit_adj_seed42/ & seed43/); paired bootstrap report (phase2_phase3_paired_bootstrap.json). | None. Both post-hoc (3a) and train-time (3b) arms evaluated and compared to Exp-2c control. |
| **Phase 4: Hierarchical Loss + TSC** | **PARTIAL** | src/losses/hierarchical.py (AsymmetricHierarchicalCostLoss, 7 tests passing in test_hierarchical_loss.py); src/losses/tsc.py (TargetedSupervisedContrastiveLoss, 7 tests passing in test_tsc_loss.py); docs/phase4_tsc_plan.md; exp4a-wip branch contains run script, configs, and notebook. | On main: no training script/config wires hierarchical or TSC loss into training; Exp-4a alpha sweep (0.25, 0.5, 1.0, 2.0) not yet run on Kaggle; TSC faithfulness to official LTH14/targeted-supcon code unverified (requires Hungarian matching vs fixed target resolution & joint-training validation). |
| **Phase 5: MONET Concept Head & Fairness** | **NOT STARTED** | Skin-tone disaggregation metrics in Phase 2/3 (calculate_subgroup_metrics in src/eval/metrics.py); Tone 3 concentration documented in docs/technical_reference/DATASET_ANALYSIS.md. | Zero concept bottleneck head code; zero concept column loading; zero Expected Calibration Error (ECE) or reliability diagram code; zero demographic fairness mitigation code. |
| **Phase 6: Multimodal Fusion & Deployment** | **NOT STARTED** | Architecture diagram (docs/multimodal_dual_image_fusion_architecture.png); dermoscopic + clinical paired image filenames in dataset metadata. | No api/ directory; no FastAPI inference service; no dual-image fusion architecture implementation; no attention rollout explainability module; no official benchmark prediction export format. |
| **Phase 7: Paper Artifacts** | **NOT STARTED** | results/ablation_table.csv, results/MANIFEST.md, results/analysis/FINAL_SUMMARY.md, docs/technical_reference/ technical docs. | No REPRODUCIBILITY.md; no model_card.md; no publication-grade matplotlib/seaborn plotting scripts; final submission tables not compiled. |

---

## 3. Experiments Table

*(Copied verbatim from [`results/audit/experiments_table.md`](../results/audit/experiments_table.md))*

| Experiment | Configuration File | Execution Script | Results Folder Present? | Seed | Test Macro-F1 | Comparability Note |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Exp-1 (seed 42)** | `configs/experiments/exp1_resnet50_kaggle.yaml` | `kaggle/run_exp1_resnet50.py` | **YES** | 42 | **0.4158** | Standard CNN baseline (ResNet-50, ImageNet pretrained, class-balanced CE, uniform shuffle). Differs from PanDerm in backbone and training recipe. |
| **Exp-1 (seed 43)** | `configs/experiments/exp1_resnet50_kaggle.yaml` | `kaggle/run_exp1_resnet50.py` | **YES** | 43 | **0.4015** | Standard CNN baseline replicate (seed 43, best epoch 19). |
| **Exp-2 (v1)** | `configs/experiments/exp2_panderm.yaml` | `scripts/train_exp2_panderm.py` | **YES** | 42 | **0.1091** | Initial collapsed recipe (unweighted sampler, majority collapse onto BCC/NV, Macro-F1 0.1091). Not a valid baseline. |
| **Exp-2 (v2)** | `configs/experiments/exp2_panderm_kaggle.yaml` | `scripts/train_exp2_panderm.py` | **YES** | 42 | **0.3925** | Kaggle T4 run (WeightedRandomSampler, layer decay 0.9, AdamW). Batch-balanced, not uniform. |
| **Exp-2c (seed 42)** | `configs/experiments/exp2c_panderm_uniform_kaggle.yaml` | `kaggle/run_exp2c_panderm_uniform.py` | **YES** | 42 | **0.4690** | PanDerm ViT uniform shuffle control arm. Evaluated from best_model.pt (epoch 27) after weights_only fix. Phase 4 base recipe winner. |
| **Exp-2c (seed 43)** | `configs/experiments/exp2c_panderm_uniform_kaggle.yaml` | `kaggle/run_exp2c_panderm_uniform.py` | **YES** | 43 | **0.3636** | PanDerm ViT uniform shuffle control arm replicate (seed 43, best epoch 14). |
| **Exp-3a** | `configs/experiments/exp2_panderm_kaggle.yaml` | `scripts/run_exp3a_post_hoc.py` | **YES** | 42 | **0.3883** | Post-hoc logit adjustment applied to frozen Exp-2 v2 checkpoint (tau*=0.10 on val). NULL result (Delta F1 CI crosses 0). |
| **Exp-3b (seed 42)** | `configs/experiments/exp3b_panderm_logit_adj_kaggle.yaml` | `kaggle/run_exp3b_panderm_logit_adj.py` | **YES** | 42 | **0.3667** | Train-time logit adjustment with fixed tau=1.0. Destructive BCC recall collapse (0.0000), overall acc 22.90%. |
| **Exp-3b (seed 43)** | `configs/experiments/exp3b_panderm_logit_adj_kaggle.yaml` | `kaggle/run_exp3b_panderm_logit_adj.py` | **YES** | 43 | **0.3653** | Train-time logit adjustment replicate (seed 43, best epoch 25). BCC recall collapse confirmed. |
| **Exp-4a (alpha sweep)** | `configs/experiments/exp4a_panderm_hierarchical_kaggle.yaml (on exp4a-wip)` | `kaggle/run_exp4a_panderm_hierarchical.py (on exp4a-wip)` | **EMPTY (Smoke test only)** | 42 | **N/A** | Asymmetric hierarchical cost loss alpha in {0.25, 0.5, 1.0, 2.0}. Smoke tested locally (1 batch); 30-epoch Kaggle runs NOT YET RUN. |
| **Exp-4b (TSC)** | `UNVERIFIED (Not yet created)` | `UNVERIFIED (Not yet created)` | **NO** | 42 | **N/A** | Targeted Supervised Contrastive Learning loss. Unit tests pass (tests/test_tsc_loss.py); training integration script NOT YET CREATED. |

---

## 4. Remaining Work (Ordered by Dependency)

*(Copied verbatim from [`results/audit/remaining_work_table.md`](../results/audit/remaining_work_table.md))*

| Order | Planned Task | Primary Actor | Estimated Compute / GPU-Hours | Dependency |
| :---: | :--- | :---: | :--- | :--- |
| 1 | Merge/restore Exp-4a hierarchical loss execution code from branch exp4a-wip to main | **agy (local)** | 0.0 GPU-hours (local git & AST verify) | Status audit completion |
| 2 | Build validated Exp-4a Kaggle execution notebook with alpha sweep support | **agy (local)** | 0.0 GPU-hours (local generator script) | Task 1 |
| 3 | Package updated codebase.zip with Exp-4a scripts and configs and upload to Kaggle Dataset | **agy (local) / user (Kaggle upload)** | 0.0 GPU-hours | Task 2 |
| 4 | Execute Exp-4a validation alpha sweep on Kaggle GPU: alpha in {0.25, 0.5, 1.0, 2.0} on Exp-2c base recipe (seed 42) | **user (on Kaggle GPU)** | 2.0 GPU-hours (4 runs x ~0.50h measured wall-clock per 30-epoch PanDerm run) | Task 3 |
| 5 | Select optimal alpha* on validation Macro-F1; run Exp-4a confirm run on seed 43 | **user (on Kaggle GPU)** | 0.50 GPU-hours (1 run x 0.50h) | Task 4 |
| 6 | Verify TSC implementation against official LTH14/targeted-supcon repo and implement joint fine-tuning module (Exp-4b) | **agy (local)** | 0.0 GPU-hours (local unit tests & smoke tests) | Task 5 |
| 7 | Execute Exp-4b (TSC + hierarchical loss joint fine-tuning) on Kaggle GPU across seeds 42 and 43 | **user (on Kaggle GPU)** | 1.0 GPU-hours (2 runs x 0.50h) | Task 6 |
| 8 | Phase 5: Implement MONET concept bottleneck head, load concept annotations, implement ECE & calibration curve metrics | **agy (local)** | 0.0 GPU-hours | Task 7 |
| 9 | Train and evaluate Phase 5 Concept Bottleneck Head & demographic fairness mitigations | **user (on Kaggle GPU)** | 1.0 GPU-hours (estimated 2 runs x 0.50h) | Task 8 |
| 10 | Phase 6: Implement dual-image fusion architecture (dermoscopic + clinical close-up) & attention rollout explainability | **agy (local)** | 0.0 GPU-hours | Task 9 |
| 11 | Train dual-image fusion model on Kaggle GPU and evaluate test performance against single-modality baseline | **user (on Kaggle GPU)** | 1.5 GPU-hours (3 runs x 0.50h) | Task 10 |
| 12 | Phase 6: Build FastAPI backend inference service conforming to frontend contract, with unit and integration tests | **agy (local)** | 0.0 GPU-hours | Task 11 |
| 13 | Phase 7: Compile paper artifacts (REPRODUCIBILITY.md, model_card.md, matplotlib publication figures, camera-ready tables) | **agy (local)** | 0.0 GPU-hours | Task 12 |

*Note on GPU-hour derivations*: Measured training wall-clock on Kaggle Tesla T4 averages **1,808 seconds (~0.50 hours)** per 30-epoch PanDerm ViT-Base run (`exp2_panderm_v2`: 1,760s; `exp2c_seed43`: 1,615s; `exp3b_seed42`: 1,779s; `exp3b_seed43`: 2,077s). All multi-epoch ViT training estimates above are directly derived from these measured wall-clock figures. Tasks lacking measured precedent are marked as estimates or "estimate unavailable".

---

## 5. Open Risks

1. **Extreme Imbalance and Tail-Class Test Cohort Sizes**:
   The MILK10k dataset has a maximum class imbalance ratio of $280.2 : 1$. On the held-out test split ($N=786$), tail classes have minimal representation: `INF` ($n=2$), `BEN_OTH` ($n=7$), `MAL_OTH` ($n=12$), and `VASC` ($n=16$). Correctly classifying or misclassifying a single lesion in these classes swings the class recall by $8.3\%$ to $50.0\%$, making unaggregated tail metrics highly volatile.
2. **Two Random Seeds ($n=2$)**:
   Due to Kaggle GPU monthly quota allocations (30 GPU-hours per week), runs were constrained to two seeds ($42$ and $43$). While paired bootstrap resampling ($B=2,000$) provides lesion-level confidence intervals, inter-seed variation is noticeable (e.g. Exp-2c test Macro-F1 dropped from $0.4690$ on seed 42 to $0.3636$ on seed 43).
3. **TSC Faithfulness to Official Li et al. (CVPR 2022) Codebase**:
   The official `targeted-supcon` implementation uses 2-stage self-supervised pretraining with dynamic Hungarian matching on ImageNet-LT ($K=1000$). In our medical setting ($K=11$ on a pre-trained dermatology foundation model), dynamic Hungarian matching across epochs can destabilize supervised classification. Reconciling fixed hypersphere target binding with the paper's formulation remains a key technical risk for Exp-4b.
4. **Kaggle Environment and Module Shadowing Audit**:
   An `ImportError: cannot import name 'nn' from partially initialized module 'torch'` was observed in Cell 1 of a Kaggle notebook. A repository-wide audit confirmed that **zero** files or directories named `torch.py`, `torch/`, `nn.py`, or `nn/` exist in the repository or inside `codebase.zip`. The error was likely due to ephemeral out-of-order cell execution on Kaggle or a user-created file in `/kaggle/working`.
5. **Phase 4 Base Recipe Dependency**:
   Phase 4 builds on Exp-2c (uniform shuffling, label smoothing 0.1), which won the base recipe selection protocol with a validation Macro-F1 of $0.5064$ (+10.0% over weighted sampling and logit adjustment). The $\alpha$ hyperparameter must be selected strictly on validation Macro-F1 before touching the test set.

---

## 6. Decisions I Must Make

1. **Exp-4a Alpha Sweep Quota Allocation**:
   Do you approve allocating ~2.0 Kaggle GPU-hours (4 runs $\times$ ~0.50h on Tesla T4) to sweep $\alpha \in \{0.25, 0.5, 1.0, 2.0\}$ on seed 42, followed by ~0.50h for a seed 43 confirmation run?
2. **Target Binding Strategy for TSC (Exp-4b)**:
   Should we adopt deterministic fixed target binding on the 11-class unit hypersphere (recommended for supervised transfer fine-tuning), or attempt dynamic Hungarian matching per epoch as in the original 2-stage MoCo pretraining paper?
3. **Phase 5 MONET Concept Annotations**:
   How will the MONET concept annotations for the MILK10k dataset be provided (pre-computed concept score CSV/array, or running the MONET encoder directly during training)?
4. **Phase 6 Dual-Image Fusion Architecture**:
   Should the clinical close-up image stream use a separate PanDerm ViT backbone, a lighter CNN (ResNet-50), or cross-attention fusion into the dermoscopic PanDerm backbone?

---

## 7. Exactly What I Must Do by Hand Next, Step by Step

1. **Verify Kaggle GPU Quota**: Open your Kaggle Account page and ensure you have at least ~2.5 GPU-hours available for Phase 4.1.
2. **Authorize Merging `exp4a-wip` to `main`**: Tell the assistant to restore/merge `exp4a-wip` to `main`, regenerate `notebooks/kaggle_exp4a_panderm_hierarchical.ipynb`, and rebuild `codebase.zip`.
3. **Upload Codebase**: Upload the updated `codebase.zip` to your Kaggle Dataset (`skincancertracker-codebase`).
4. **Launch Alpha Sweep on Kaggle**: Import `notebooks/kaggle_exp4a_panderm_hierarchical.ipynb` into a Kaggle Notebook, attach GPU (T4 or P100), attach the MILK10k dataset and `skincancertracker-codebase`, and run the validation sweep for $\alpha \in \{0.25, 0.5, 1.0, 2.0\}$ on seed 42.
5. **Download Artifacts**: Download `exp4a_panderm_hierarchical_alpha{ALPHA}_seed42_artifacts.zip` for each completed run and place them in `results/`.
