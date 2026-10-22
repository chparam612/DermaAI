# GATE 2: Attention Rollout Explainability Report

**Timestamp (UTC)**: 2026-10-22T10:30:39Z  
**Method**: Attention Rollout (Abnar & Zuidema, 2020)  
**External PanDerm Modified**: `False` (zero edits to `external/panderm`)  
**Output Shape**: [14, 14] spatial patch grid, upsampled to input image dimensions  
**Value Bounds**: `[0.0, 1.0]` (min-max normalized)  
**Deterministic**: `True` (absolute diff between identical runs = 0.0)  
**CPU Latency (Median)**: 216.11 ms (Mean: 220.74 ms)  
**Overlay Format**: RGBA PNG Base64 Data URI with transparent background  

## Rendered Visual Examples

| # | True Class | Pred Class | ISIC ID | Original Image | Blended Overlay | Transparent Overlay |
|---|---|---|---|---|---|---|
| 1 | BCC | BCC | `ISIC_8325963` | `results\audit\rollout_examples\example_1_BCC_ISIC_8325963_orig.jpg` | `results\audit\rollout_examples\example_1_BCC_ISIC_8325963_blended.jpg` | `results\audit\rollout_examples\example_1_BCC_ISIC_8325963_overlay.png` |
| 2 | MEL | MEL | `ISIC_3704603` | `results\audit\rollout_examples\example_2_MEL_ISIC_3704603_orig.jpg` | `results\audit\rollout_examples\example_2_MEL_ISIC_3704603_blended.jpg` | `results\audit\rollout_examples\example_2_MEL_ISIC_3704603_overlay.png` |
| 3 | SCCKA | BKL | `ISIC_1230865` | `results\audit\rollout_examples\example_3_SCCKA_ISIC_1230865_orig.jpg` | `results\audit\rollout_examples\example_3_SCCKA_ISIC_1230865_blended.jpg` | `results\audit\rollout_examples\example_3_SCCKA_ISIC_1230865_overlay.png` |
| 4 | NV | NV | `ISIC_8015399` | `results\audit\rollout_examples\example_4_NV_ISIC_8015399_orig.jpg` | `results\audit\rollout_examples\example_4_NV_ISIC_8015399_blended.jpg` | `results\audit\rollout_examples\example_4_NV_ISIC_8015399_overlay.png` |
| 5 | BKL | MEL | `ISIC_1225678` | `results\audit\rollout_examples\example_5_BKL_ISIC_1225678_orig.jpg` | `results\audit\rollout_examples\example_5_BKL_ISIC_1225678_blended.jpg` | `results\audit\rollout_examples\example_5_BKL_ISIC_1225678_overlay.png` |
