# GATE 1: Checkpoint Export, Parity Verification, and Benchmarking Report

**Date/Time (UTC)**: 2026-10-22T10:08:54Z  
**Model Architecture**: `PanDermClassifier (panderm_base)`  
**Image Size / Normalization**: 224px (Resize 256, CenterCrop 224), mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]  

## 1. Checkpoint Export

- **Path**: `checkpoints\final\demo_model.pt`
- **File Size**: 343,331,341 bytes (327.43 MB)
- **SHA-256**: `137c7991202448e3a2c45b7f93dca680d44e05f3b8321882840514219c613b3d`
- **Load Mode**: `torch.load(..., weights_only=True, map_location='cpu')`
- **Model Load Time**: 4.0518 s

## 2. Parity Verification (N=50 Test Lesions)

- **Sample Size**: 50 test lesions (stratified across all test classes)
- **Max Absolute Probability Difference**: `4.335662e-03` (Threshold <= 0.01)
- **Mean Absolute Probability Difference**: `7.218315e-05`
- **Argmax Disagreements**: `0 / 50` (Threshold <= 2)
- **Parity Status**: `PASSED`

### Sampled Lesions Breakdown

| Lesion ID | ISIC ID | True Class | Saved Pred | API Pred | Max Abs Diff | Match |
|:---|:---|:---|:---|:---|:---|:---:|
| IL_0252336 | ISIC_6543284 | MAL_OTH | BCC | BCC | 0.000211 | YES |
| IL_0321969 | ISIC_9579888 | NV | MEL | MEL | 0.000245 | YES |
| IL_0424747 | ISIC_3642704 | NV | NV | NV | 0.001723 | YES |
| IL_0465605 | ISIC_3045480 | MEL | NV | NV | 0.000033 | YES |
| IL_0719888 | ISIC_2975261 | BCC | BCC | BCC | 0.000051 | YES |
| IL_0909755 | ISIC_9890254 | NV | NV | NV | 0.000260 | YES |
| IL_1192220 | ISIC_2321009 | BCC | BCC | BCC | 0.000065 | YES |
| IL_1364035 | ISIC_4875335 | BCC | BCC | BCC | 0.000311 | YES |
| IL_1557433 | ISIC_1826564 | BCC | BKL | BKL | 0.001709 | YES |
| IL_1673815 | ISIC_9927174 | BCC | AKIEC | AKIEC | 0.000233 | YES |
| IL_1694504 | ISIC_2328520 | BCC | BCC | BCC | 0.000086 | YES |
| IL_1734179 | ISIC_3103579 | MEL | BCC | BCC | 0.000099 | YES |
| IL_1936294 | ISIC_5044856 | BCC | BCC | BCC | 0.004336 | YES |
| IL_2096474 | ISIC_2872052 | SCCKA | SCCKA | SCCKA | 0.000089 | YES |
| IL_2704222 | ISIC_3235332 | BCC | BCC | BCC | 0.000086 | YES |
| IL_2929786 | ISIC_9977374 | SCCKA | BKL | BKL | 0.001110 | YES |
| IL_2962292 | ISIC_9348273 | BCC | BCC | BCC | 0.000464 | YES |
| IL_2989326 | ISIC_8955200 | MEL | BCC | BCC | 0.000315 | YES |
| IL_3102698 | ISIC_0084581 | SCCKA | AKIEC | AKIEC | 0.001376 | YES |
| IL_3102768 | ISIC_6909525 | BCC | BCC | BCC | 0.000083 | YES |
| IL_3146700 | ISIC_9643914 | BCC | BCC | BCC | 0.000145 | YES |
| IL_3360591 | ISIC_7656588 | SCCKA | SCCKA | SCCKA | 0.000182 | YES |
| IL_3565320 | ISIC_2753382 | MEL | INF | INF | 0.000614 | YES |
| IL_4308535 | ISIC_8138627 | AKIEC | BKL | BKL | 0.001525 | YES |
| IL_4396265 | ISIC_0497849 | BCC | BCC | BCC | 0.000251 | YES |
| IL_4504514 | ISIC_9232517 | BCC | BCC | BCC | 0.000156 | YES |
| IL_4896057 | ISIC_2378626 | BCC | BCC | BCC | 0.000048 | YES |
| IL_5267553 | ISIC_4189363 | BCC | BCC | BCC | 0.000240 | YES |
| IL_5277754 | ISIC_0655029 | BCC | BCC | BCC | 0.000099 | YES |
| IL_5283684 | ISIC_8052859 | BCC | BCC | BCC | 0.000089 | YES |
| IL_5572380 | ISIC_8586056 | BCC | BCC | BCC | 0.000908 | YES |
| IL_5959411 | ISIC_8439407 | BCC | BCC | BCC | 0.000167 | YES |
| IL_6130939 | ISIC_1793316 | BCC | BCC | BCC | 0.000037 | YES |
| IL_6192656 | ISIC_2036909 | BCC | BCC | BCC | 0.000124 | YES |
| IL_6767086 | ISIC_7991319 | BCC | BCC | BCC | 0.000083 | YES |
| IL_6841105 | ISIC_9390664 | BCC | BCC | BCC | 0.000059 | YES |
| IL_7385795 | ISIC_8825637 | INF | NV | NV | 0.000235 | YES |
| IL_7385851 | ISIC_7326842 | AKIEC | AKIEC | AKIEC | 0.000176 | YES |
| IL_7687935 | ISIC_6141128 | VASC | VASC | VASC | 0.000217 | YES |
| IL_7760285 | ISIC_1346867 | BCC | BCC | BCC | 0.000070 | YES |
| IL_8007348 | ISIC_3762115 | BCC | BCC | BCC | 0.000042 | YES |
| IL_8110471 | ISIC_4247695 | BKL | BKL | BKL | 0.000062 | YES |
| IL_8229749 | ISIC_6538645 | BCC | BCC | BCC | 0.000130 | YES |
| IL_8845602 | ISIC_8618352 | BCC | BCC | BCC | 0.000053 | YES |
| IL_9095433 | ISIC_9483507 | BCC | BCC | BCC | 0.000027 | YES |
| IL_9132121 | ISIC_0811596 | BEN_OTH | BKL | BKL | 0.000159 | YES |
| IL_9409547 | ISIC_9596173 | MEL | NV | NV | 0.000068 | YES |
| IL_9679718 | ISIC_6312345 | DF | DF | DF | 0.000484 | YES |
| IL_9701930 | ISIC_1417674 | BCC | BCC | BCC | 0.000505 | YES |
| IL_9725975 | ISIC_3706698 | BKL | SCCKA | SCCKA | 0.000114 | YES |

## 3. Latency & Resource Utilization on CPU

| Metric | Value |
|:---|:---|
| Median Per-Image Latency (N=20) | 234.13 ms |
| Mean Per-Image Latency (N=20) | 290.02 ms |
| Min Per-Image Latency (N=20) | 210.12 ms |
| Max Per-Image Latency (N=20) | 479.44 ms |
| Initial RAM (Process RSS) | 388.4 MB |
| RAM After Model Load | 1454.1 MB |
| Peak Process RAM | 1479.7 MB |
