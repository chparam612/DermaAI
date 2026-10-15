# Phase 2 Baseline Experiments Ablation Table

| exp_id     | model            | modality    |   accuracy |   macro_f1 |   malignant_recall | auroc                | notes                                                                                                |
|:-----------|:-----------------|:------------|-----------:|-----------:|-------------------:|:---------------------|:-----------------------------------------------------------------------------------------------------|
| Exp-1      | ResNet-50        | Dermoscopic |     0.5191 |     0.2869 |             0.8868 | 0.7758 (mal: 0.7549) | 1 epoch, local, not comparable (RTX 2050 reference; full 30-epoch Kaggle baseline pending)           |
| Exp-2 (v1) | PanDerm ViT-Base | Dermoscopic |     0.5344 |     0.1091 |             0.8165 | 0.6253 (mal: 0.6699) | v1 recipe (unweighted sampler, majority-class collapse onto BCC/NV)                                  |
| Exp-2 (v2) | PanDerm ViT-Base | Dermoscopic |     0.6438 |     0.3925 |             0.8902 | 0.7547 (mal: 0.8022) | v2 corrected (weighted sampler, layer_decay=0.9, val_macro_f1 checkpointing; selection on val only)  |

*Evaluation performed on MILK10k held-out test split (786 dermoscopic images, zero lesion leakage).*
*Malignant classes evaluated: BCC, MEL, SCCKA, MAL_OTH, AKIEC.*
*Note on baseline comparisons: No backbone superiority claim is made between ResNet-50 and PanDerm ViT-Base until the full 30-epoch Kaggle Exp-1 run is executed.*
