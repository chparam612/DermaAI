# Per-Split skin_tone_class Distribution Table

Cross-tabulation of `skin_tone_class` by `split` in `data/splits/lesion_level_splits.csv`.

## 1. Unique Lesion Counts (N=5,240 lesions)

Each unique lesion has identical skin tone across paired clinical & dermoscopic modalities.

| skin_tone_class | Train (Lesions) | Validation (Lesions) | Test (Lesions) | Total (All Splits) |
|:---:|:---:|:---:|:---:|:---:|
| **Class 0** | 2 | 4 | 0 | 6 |
| **Class 1** | 82 | 11 | 12 | 105 |
| **Class 2** | 358 | 76 | 72 | 506 |
| **Class 3** | 2,198 | 490 | 486 | 3,174 |
| **Class 4** | 754 | 159 | 153 | 1,066 |
| **Class 5** | 274 | 46 | 63 | 383 |
| **NULL / Missing** | 0 | 0 | 0 | 0 |
| **Total** | 3,668 | 786 | 786 | 5,240 |

## 2. Dermoscopic Modality Counts (N=5,240 images, 1 per lesion)

This is the exact evaluation set used in baseline experiments (Exp-1, Exp-2, Exp-2c, Exp-3a, Exp-3b).

| skin_tone_class | Train (Derm) | Validation (Derm) | Test (Derm) | Total (All Splits) |
|:---:|:---:|:---:|:---:|:---:|
| **Class 0** | 2 | 4 | 0 | 6 |
| **Class 1** | 82 | 11 | 12 | 105 |
| **Class 2** | 358 | 76 | 72 | 506 |
| **Class 3** | 2,198 | 490 | 486 | 3,174 |
| **Class 4** | 754 | 159 | 153 | 1,066 |
| **Class 5** | 274 | 46 | 63 | 383 |
| **NULL / Missing** | 0 | 0 | 0 | 0 |
| **Total** | 3,668 | 786 | 786 | 5,240 |

## 3. All Images Counts (N=10,480 images, paired clinical + dermoscopic)

| skin_tone_class | Train (Images) | Validation (Images) | Test (Images) | Total (All Splits) |
|:---:|:---:|:---:|:---:|:---:|
| **Class 0** | 4 | 8 | 0 | 12 |
| **Class 1** | 164 | 22 | 24 | 210 |
| **Class 2** | 716 | 152 | 144 | 1,012 |
| **Class 3** | 4,396 | 980 | 972 | 6,348 |
| **Class 4** | 1,508 | 318 | 306 | 2,132 |
| **Class 5** | 548 | 92 | 126 | 766 |
| **NULL / Missing** | 0 | 0 | 0 | 0 |
| **Total** | 7,336 | 1,572 | 1,572 | 10,480 |

## 4. Comparison with Frontend F4 Panel and DATASET_ANALYSIS.md

| Skin Tone Class | Frontend F4 Panel (Test Dermoscopic N=786) | Test Unique Lesions (N=786) | Entire Dataset Unique Lesions (N=5,240) | Entire Dataset Total Images (N=10,480, DATASET_ANALYSIS.md) | User Prompt String |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Class 0** | *(Not shown / n=0)* | 0 | 6 | 12 | 0:12 |
| **Class 1** | Tone 1: **12** | 12 | 105 | 210 | 1:210 |
| **Class 2** | Tone 2: **72** | 72 | 506 | 1,012 | 2:1012 |
| **Class 3** | Tone 3: **486** | 486 | 3,174 | 6,348 | 3:48 |
| **Class 4** | Tone 4: **153** | 153 | 1,066 | 2,132 | 4:8 |
| **Class 5** | Tone 5: **63** | 63 | 383 | 766 | 5:1 |
| **Total** | **786** | **786** | **5,240** | **10,480** | N/A |

### Key Findings from Comparison:
1. **The Frontend shows the HELD-OUT TEST SPLIT ($N=786$ dermoscopic lesions)**: `Tone 1: 12`, `Tone 2: 72`, `Tone 3: 486`, `Tone 4: 153`, `Tone 5: 63` are the exact counts of `skin_tone_class == 1, 2, 3, 4, 5` in `df_test`. Class 0 has 0 test lesions, so it is omitted by `df_test['skin_tone_class'].dropna().unique()`.
2. **DATASET_ANALYSIS.md lines 134-142 shows the ENTIRE DATASET ($N=10,480$ total images)**: 12 images for Class 0, 210 for Class 1, 1,012 for Class 2, 6,348 for Class 3, 2,132 for Class 4, and 766 for Class 5.
3. **The User Prompt String `(0:12, 1:210, 2:1012, 3:48, 4:8, 5:1)` reflects truncated image counts**: `0:12`, `1:210`, `2:1012` match the total image counts; `3:48` truncates `6348`; `4:8` truncates/misreads `2132`; `5:1` truncates/misreads `766`.
4. **No Index Shift Occurred**: Class 1 in the test set happens to have exactly 12 lesions ($n=12$), which coincidentally equals the number of Class 0 images across the entire dataset (12 images across 6 lesions). Tone 1 in the frontend is NOT Class 0; it is literally `skin_tone_class == 1`.