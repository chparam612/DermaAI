import json
import pandas as pd

with open("configs/label_mapping.json") as f:
    classes = json.load(f)["classes"]

df_val = pd.read_csv("results/exp2_panderm_v2/predictions_val.csv")
df_test = pd.read_csv("results/exp2_panderm_v2/predictions_test.csv")

def get_dist(df):
    true_cnt = df["true_label"].value_counts().sort_index().to_dict()
    pred_cnt = df["pred_label"].value_counts().sort_index().to_dict()
    res = []
    for i, c in enumerate(classes):
        t = true_cnt.get(i, 0)
        p = pred_cnt.get(i, 0)
        res.append((c, t, f"{t/len(df)*100:.1f}%", p, f"{p/len(df)*100:.1f}%"))
    return res

print("=== VALIDATION SET (N=786) ===")
print(f"{'Class':<10} {'True Count':<12} {'True %':<10} {'Pred Count':<12} {'Pred %':<10}")
print("-" * 54)
for row in get_dist(df_val):
    print(f"{row[0]:<10} {row[1]:<12} {row[2]:<10} {row[3]:<12} {row[4]:<10}")

print("\n=== TEST SET (N=786) ===")
print(f"{'Class':<10} {'True Count':<12} {'True %':<10} {'Pred Count':<12} {'Pred %':<10}")
print("-" * 54)
for row in get_dist(df_test):
    print(f"{row[0]:<10} {row[1]:<12} {row[2]:<10} {row[3]:<12} {row[4]:<10}")
