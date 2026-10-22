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
