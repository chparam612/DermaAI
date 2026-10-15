"""Script to generate the unified, fully-merged master dataset CSV.

Merges:
1. training_input.csv (Image-level features, skin tone, MONET biomarkers)
2. training_gt.csv (Lesion-level ground-truth 11 one-hot diagnosis classes)
   --> KEY: lesion_id (Cardinality: 1 lesion to 2 images)
3. training_supp.csv (Supplementary diagnosis description, biopsy confirm type, invasion thickness)
   --> KEY: isic_id (Cardinality: 1 to 1)
4. metadata.csv (Anatomical site details, melanocytic flag, biopsy status)
   --> KEY: isic_id (Cardinality: 1 to 1)

Output:
data/processed/merged_dataset.csv (10,480 rows, fully integrated)
"""

from pathlib import Path
import pandas as pd
import numpy as np

def generate_master_merged_dataset(data_dir: Path = Path("data"), output_dir: Path = Path("data/processed")) -> Path:
    data_dir = data_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("[1/5] Loading source CSV tables...")
    df_input = pd.read_csv(data_dir / "training_input.csv")
    df_gt = pd.read_csv(data_dir / "training_gt.csv")
    df_supp = pd.read_csv(data_dir / "training_supp.csv")
    df_meta = pd.read_csv(data_dir / "metadata.csv")
    
    print(f"  - training_input: {df_input.shape[0]} rows, {df_input.shape[1]} cols")
    print(f"  - training_gt:    {df_gt.shape[0]} rows, {df_gt.shape[1]} cols")
    print(f"  - training_supp:  {df_supp.shape[0]} rows, {df_supp.shape[1]} cols")
    print(f"  - metadata:       {df_meta.shape[0]} rows, {df_meta.shape[1]} cols")
    
    # 11 diagnostic classes
    diagnostic_classes = [
        "AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF",
        "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"
    ]
    
    # Create categorical string diagnosis from one-hot columns in training_gt
    df_gt["diagnosis"] = df_gt[diagnostic_classes].idxmax(axis=1)
    
    print("\n[2/5] MERGE 1: Merging training_input with training_gt...")
    print("  --> MERGE KEY (Highlighted): [lesion_id]")
    print("  --> Nature: 1:2 Cardinality (1 lesion ground-truth maps to 2 image records)")
    merged = pd.merge(df_input, df_gt, on="lesion_id", how="left")
    print(f"  Resulting rows: {len(merged)} (Expected: 10480)")
    
    print("\n[3/5] MERGE 2: Merging with training_supp...")
    print("  --> MERGE KEY (Highlighted): [isic_id]")
    print("  --> Nature: 1:1 Cardinality (1 image maps to 1 supplementary record)")
    supp_cols_to_add = [c for c in df_supp.columns if c not in merged.columns or c == "isic_id"]
    merged = pd.merge(merged, df_supp[supp_cols_to_add], on="isic_id", how="left")
    print(f"  Resulting rows: {len(merged)}")
    
    print("\n[4/5] MERGE 3: Merging with metadata...")
    print("  --> MERGE KEY (Highlighted): [isic_id]")
    print("  --> Nature: 1:1 Cardinality (1 image maps to 1 clinical metadata record)")
    meta_cols_to_add = [
        "isic_id", "anatom_site_general", "anatom_site_special", 
        "concomitant_biopsy", "melanocytic", "diagnosis_confirm_type"
    ]
    # Filter only available columns
    meta_cols_available = [c for c in meta_cols_to_add if c in df_meta.columns and (c not in merged.columns or c == "isic_id")]
    merged = pd.merge(merged, df_meta[meta_cols_available], on="isic_id", how="left")
    print(f"  Resulting rows: {len(merged)}")
    
    # Reorder columns cleanly: IDs first, then Targets, then Features
    id_cols = ["isic_id", "lesion_id"]
    target_cols = ["diagnosis", "diagnosis_full"] + diagnostic_classes
    feature_cols = [c for c in merged.columns if c not in id_cols and c not in target_cols]
    final_cols = id_cols + target_cols + feature_cols
    merged = merged[final_cols]
    
    output_path = output_dir / "merged_dataset.csv"
    print(f"\n[5/5] Saving master merged CSV to: {output_path}...")
    merged.to_csv(output_path, index=False)
    print(f"SUCCESS: Saved {len(merged)} records with {len(merged.columns)} columns.")
    print(f"  - Total unique lesions: {merged['lesion_id'].nunique()}")
    print(f"  - Total unique images:  {merged['isic_id'].nunique()}")
    print(f"  - Nulls in lesion_id:   {merged['lesion_id'].isna().sum()}")
    print(f"  - Nulls in isic_id:     {merged['isic_id'].isna().sum()}")
    print(f"  - Nulls in diagnosis:   {merged['diagnosis'].isna().sum()}")
    
    return output_path

if __name__ == "__main__":
    generate_master_merged_dataset()
