import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score

df = pd.read_csv("results/exp2_panderm_v2/predictions.csv")
with open("configs/label_mapping.json") as f:
    label_map = json.load(f)
classes = label_map["classes"]

df_t3 = df[df["skin_tone_class"] == 3]
df_nt3 = df[df["skin_tone_class"] != 3]

np.random.seed(42)
n_boot = 2000

print(f"{'Class':<10} {'T3 Supp':<8} {'T3 Rec [95% CI]':<26} {'NT3 Supp':<8} {'NT3 Rec [95% CI]':<26} {'Diff [95% CI]':<24}")
print("-" * 105)

for idx, cls in enumerate(classes):
    t3_mask = (df_t3["true_label"] == idx).values
    nt3_mask = (df_nt3["true_label"] == idx).values
    t3_pred = (df_t3["pred_label"] == idx).values
    nt3_pred = (df_nt3["pred_label"] == idx).values

    t3_supp = np.sum(t3_mask)
    nt3_supp = np.sum(nt3_mask)

    t3_rec = np.sum(t3_mask & t3_pred) / t3_supp if t3_supp > 0 else np.nan
    nt3_rec = np.sum(nt3_mask & nt3_pred) / nt3_supp if nt3_supp > 0 else np.nan
    rec_diff = t3_rec - nt3_rec if (not np.isnan(t3_rec) and not np.isnan(nt3_rec)) else np.nan

    boot_t3_recs = []
    boot_nt3_recs = []
    boot_diffs = []

    for _ in range(n_boot):
        # Sample T3
        idx_t3 = np.random.randint(0, len(df_t3), size=len(df_t3))
        b_t3_m = t3_mask[idx_t3]
        b_t3_p = t3_pred[idx_t3]
        b_t3_supp = np.sum(b_t3_m)
        b_t3_rec = np.sum(b_t3_m & b_t3_p) / b_t3_supp if b_t3_supp > 0 else np.nan
        boot_t3_recs.append(b_t3_rec)

        # Sample NT3
        idx_nt3 = np.random.randint(0, len(df_nt3), size=len(df_nt3))
        b_nt3_m = nt3_mask[idx_nt3]
        b_nt3_p = nt3_pred[idx_nt3]
        b_nt3_supp = np.sum(b_nt3_m)
        b_nt3_rec = np.sum(b_nt3_m & b_nt3_p) / b_nt3_supp if b_nt3_supp > 0 else np.nan
        boot_nt3_recs.append(b_nt3_rec)

        if not np.isnan(b_t3_rec) and not np.isnan(b_nt3_rec):
            boot_diffs.append(b_t3_rec - b_nt3_rec)

    valid_t3 = [r for r in boot_t3_recs if not np.isnan(r)]
    valid_nt3 = [r for r in boot_nt3_recs if not np.isnan(r)]

    t3_ci = (np.percentile(valid_t3, 2.5), np.percentile(valid_t3, 97.5)) if valid_t3 else (np.nan, np.nan)
    nt3_ci = (np.percentile(valid_nt3, 2.5), np.percentile(valid_nt3, 97.5)) if valid_nt3 else (np.nan, np.nan)
    diff_ci = (np.percentile(boot_diffs, 2.5), np.percentile(boot_diffs, 97.5)) if boot_diffs else (np.nan, np.nan)

    t3_str = f"{t3_rec:.4f} [{t3_ci[0]:.3f}, {t3_ci[1]:.3f}]" if not np.isnan(t3_rec) else "N/A"
    nt3_str = f"{nt3_rec:.4f} [{nt3_ci[0]:.3f}, {nt3_ci[1]:.3f}]" if not np.isnan(nt3_rec) else "N/A"
    diff_str = f"{rec_diff:+.4f} [{diff_ci[0]:+.3f}, {diff_ci[1]:+.3f}]" if not np.isnan(rec_diff) else "N/A"

    print(f"{cls:<10} {t3_supp:<8} {t3_str:<26} {nt3_supp:<8} {nt3_str:<26} {diff_str:<24}")
