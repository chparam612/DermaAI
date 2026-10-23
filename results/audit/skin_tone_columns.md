# Skin Tone Column Audit across data/ and data/supplements/

This table audits every column in `data/metadata.csv`, `data/supplements/*.csv`, and `data/splits/lesion_level_splits.csv`.

## File: `data/metadata.csv`
- Total Rows: 10,480
- Total Columns: 17
- Unique Lesions: 5,240
- Skin Tone Columns: **None** (No column in this file mentions skin tone)

## File: `data/supplements/training_input.csv`
- Total Rows: 10,480
- Total Columns: 17
- Unique Lesions: 5,240
- Matched Columns: `['skin_tone_class']`

### Column: `skin_tone_class`
- **Data Type**: `int64`
- **Null Count (All Rows)**: 0
- **Null Count (Over Lesions)**: 0
- **Non-Null Lesions**: 5240

**Value Counts Table:**

| Category / Value | All Rows Count | All Rows % | Unique Lesions Count | Unique Lesions % |
|:---:|:---:|:---:|:---:|:---:|
| 0 | 12 | 0.11% | 6 | 0.11% |
| 1 | 210 | 2.00% | 105 | 2.00% |
| 2 | 1,012 | 9.66% | 506 | 9.66% |
| 3 | 6,348 | 60.57% | 3,174 | 60.57% |
| 4 | 2,132 | 20.34% | 1,066 | 20.34% |
| 5 | 766 | 7.31% | 383 | 7.31% |

## File: `data/supplements/training_supp.csv`
- Total Rows: 10,480
- Total Columns: 4
- Skin Tone Columns: **None** (No column in this file mentions skin tone)

## File: `data/supplements/training_gt.csv`
- Total Rows: 5,240
- Total Columns: 12
- Unique Lesions: 5,240
- Skin Tone Columns: **None** (No column in this file mentions skin tone)

## File: `data/splits/lesion_level_splits.csv`
- Total Rows: 10,480
- Total Columns: 20
- Unique Lesions: 5,240
- Matched Columns: `['skin_tone_class']`

### Column: `skin_tone_class`
- **Data Type**: `int64`
- **Null Count (All Rows)**: 0
- **Null Count (Over Lesions)**: 0
- **Non-Null Lesions**: 5240

**Value Counts Table:**

| Category / Value | All Rows Count | All Rows % | Unique Lesions Count | Unique Lesions % |
|:---:|:---:|:---:|:---:|:---:|
| 0 | 12 | 0.11% | 6 | 0.11% |
| 1 | 210 | 2.00% | 105 | 2.00% |
| 2 | 1,012 | 9.66% | 506 | 9.66% |
| 3 | 6,348 | 60.57% | 3,174 | 60.57% |
| 4 | 2,132 | 20.34% | 1,066 | 20.34% |
| 5 | 766 | 7.31% | 383 | 7.31% |
