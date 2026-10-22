| Benchmark Experiment | Seed | Results Folder on Disk | Test Macro-F1 | Best Val Epoch | Split Hash Verified | Status / Notes |
| :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| **Exp-1 (ResNet-50)** | 42 | `results/exp1_resnet50_seed42/` | **0.4158** | 29 | `a4346dcf...` | 30-epoch Kaggle run (Class-balanced CE, Uniform). |
| **Exp-1 (ResNet-50)** | 43 | `results/exp1_resnet50_seed43/exp1_resnet50_seed43_artifacts/` | **0.4015** | 19 | `a4346dcf...` | 30-epoch Kaggle run (Class-balanced CE, Uniform). |
| **Exp-2 (v1)** | 42 | `results/exp2_panderm/` | **0.1091** | 1 | `a4346dcf...` | Initial collapsed recipe (unweighted sampler, majority collapse). |
| **Exp-2 (v2)** | 42 | `results/exp2_panderm_v2/` | **0.3925** | 26 | `a4346dcf...` | Kaggle T4 run (WeightedRandomSampler, layer decay 0.9). |
| **Exp-2c (PanDerm Uniform)** | 42 | `results/exp2c_panderm_uniform_seed42/exp2c_panderm_uniform_artifacts/` | **0.4690** | 27 | `a4346dcf...` | PanDerm uniform control arm; best_model.pt re-evaluated post weights_only fix. Phase 4 base winner. |
| **Exp-2c (PanDerm Uniform)** | 43 | `results/exp2c_panderm_uniform_seed43/exp2c_panderm_uniform_seed43_artifacts/` | **0.3636** | 14 | `a4346dcf...` | PanDerm uniform control arm replicate. |
| **Exp-3a (Post-Hoc LA)** | 42 | `results/exp3a_panderm_posthoc/` | **0.3883** | N/A | `a4346dcf...` | Post-hoc LA applied to frozen Exp-2 v2 (tau*=0.10). NULL result (CI crosses 0). |
| **Exp-3b (Train-Time LA)** | 42 | `results/exp3b_panderm_logit_adj_seed42/` | **0.3667** | 25 | `a4346dcf...` | Train-time LA (tau=1.0). BCC recall collapsed to 0.0000; absorbed by MAL_OTH. |
| **Exp-3b (Train-Time LA)** | 43 | `results/exp3b_panderm_logit_adj_seed43/` | **0.3653** | 25 | `a4346dcf...` | Train-time LA replicate (tau=1.0). BCC recall collapsed to 0.0000; absorbed by MAL_OTH. |
| **Exp-4a (Hierarchical Cost)** | 42 | `results/exp4a_panderm_hierarchical_alpha1.0_seed42/` | **N/A** | N/A | `a4346dcf...` | Empty dir from 1-batch smoke test on exp4a-wip. 30-epoch runs NOT YET RUN. |
| **Exp-4b (TSC)** | 42 | `UNVERIFIED (Not yet created)` | **N/A** | N/A | UNVERIFIED | Script & run NOT YET CREATED. |
