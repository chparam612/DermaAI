"""Gate 1 Checkpoint Export, Parity Verification, and Benchmarking."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from PIL import Image
import torch

from src.data.transforms import get_panderm_eval_transforms, PANDERM_MEAN, PANDERM_STD
from src.models.panderm import PanDermClassifier

class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ('cb', wintypes.DWORD),
        ('PageFaultCount', wintypes.DWORD),
        ('PeakWorkingSetSize', ctypes.c_size_t),
        ('WorkingSetSize', ctypes.c_size_t),
        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
        ('QuotaPagedPoolUsage', ctypes.c_size_t),
        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
        ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
        ('PagefileUsage', ctypes.c_size_t),
        ('PeakPagefileUsage', ctypes.c_size_t),
        ('PrivateUsage', ctypes.c_size_t),
    ]

_kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
_kernel32.GetCurrentProcess.restype = wintypes.HANDLE
_psapi = ctypes.WinDLL('psapi', use_last_error=True)
_psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
_psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def get_ram_usage_mb() -> float:
    try:
        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
        handle = _kernel32.GetCurrentProcess()
        if _psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return float(counters.WorkingSetSize / (1024 * 1024))
    except Exception:
        pass
    return 0.0

def main():
    print("=== Step 1a: Preprocessing and Model Class ===")
    print("Model Class: src.models.panderm.PanDermClassifier")
    print("Evaluation Preprocessing: src.data.transforms.get_panderm_eval_transforms")
    print(f"Image Size: 224 (Resize: 256, CenterCrop: 224)")
    print(f"Normalization: Mean = {PANDERM_MEAN}, Std = {PANDERM_STD}")

    # Step 1b: Export slim checkpoint
    src_ckpt = PROJECT_ROOT / "checkpoints" / "exp2c_panderm_uniform_seed42" / "best_model.pt"
    dest_ckpt = PROJECT_ROOT / "checkpoints" / "final" / "demo_model.pt"
    dest_ckpt.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Step 1b: Exporting slim checkpoint from {src_ckpt} ===")
    ram_before_load = get_ram_usage_mb()
    full_ckpt = torch.load(src_ckpt, map_location="cpu", weights_only=False)
    model_sd = full_ckpt["model_state_dict"]
    torch.save(model_sd, dest_ckpt)
    del full_ckpt

    ckpt_size_bytes = dest_ckpt.stat().st_size
    ckpt_sha256 = compute_sha256(dest_ckpt)
    print(f"Exported demo_model.pt size: {ckpt_size_bytes:,} bytes ({ckpt_size_bytes / (1024*1024):.2f} MB)")
    print(f"demo_model.pt SHA-256: {ckpt_sha256}")

    # Load via weights_only=True
    t0 = time.perf_counter()
    loaded_sd = torch.load(dest_ckpt, map_location="cpu", weights_only=True)
    model = PanDermClassifier(architecture="panderm_base", num_classes=11, pretrained=False)
    missing, unexpected = model.load_state_dict(loaded_sd, strict=True)
    model.eval()
    t_load = time.perf_counter() - t0
    ram_after_load = get_ram_usage_mb()
    print(f"Model load time: {t_load:.4f} s")
    print(f"Strict load keys: missing={len(missing)}, unexpected={len(unexpected)}")
    print(f"RAM before load: {ram_before_load:.1f} MB, after load: {ram_after_load:.1f} MB (Delta: {ram_after_load - ram_before_load:.1f} MB)")

    # Step 1c: Parity test on CPU
    print("\n=== Step 1c: Parity Test (N=50 test lesions) ===")
    preds_file = PROJECT_ROOT / "results" / "exp2c_panderm_uniform_seed42" / "exp2c_panderm_uniform_artifacts" / "predictions_test.csv"
    preds_df = pd.read_csv(preds_file)
    print(f"Total test predictions: {len(preds_df)}")

    # Load label mapping
    with open(PROJECT_ROOT / "configs" / "label_mapping.json", "r", encoding="utf-8") as f:
        label_map = json.load(f)
    classes = label_map["classes"]
    prob_cols = [f"prob_{c}" for c in classes]

    # Stratified sampling of N=50
    # Make sure every class with samples in test split appears
    # Using fixed seed=42
    rng = np.random.RandomState(42)
    sample_indices = []
    
    # First, pick 1 sample from each class present in test
    for c_idx in sorted(preds_df["true_label"].unique()):
        idx_pool = preds_df.index[preds_df["true_label"] == c_idx].tolist()
        chosen = rng.choice(idx_pool)
        sample_indices.append(chosen)

    # Remainder (50 - len(sample_indices)) proportionally sampled
    remaining_needed = 50 - len(sample_indices)
    remaining_pool = [i for i in preds_df.index if i not in sample_indices]
    remaining_probs = preds_df.loc[remaining_pool, "true_label"].map(preds_df["true_label"].value_counts(normalize=True))
    remaining_probs /= remaining_probs.sum()
    chosen_remaining = rng.choice(remaining_pool, size=remaining_needed, replace=False, p=remaining_probs)
    sample_indices.extend(chosen_remaining)
    sample_indices = sorted(sample_indices)

    sample_df = preds_df.loc[sample_indices].copy().reset_index(drop=True)
    print(f"Sampled N={len(sample_df)} test lesions across {sample_df['true_class'].nunique()} distinct classes.")
    print(f"Class distribution in sample:\n{sample_df['true_class'].value_counts().to_dict()}")

    transform = get_panderm_eval_transforms(image_size=224)
    image_dir = PROJECT_ROOT / "data" / "images"

    abs_diffs = []
    argmax_disagreements = 0
    lesion_results = []

    for idx, row in sample_df.iterrows():
        isic_id = row["isic_id"]
        lesion_id = row["lesion_id"]
        img_path = image_dir / f"{isic_id}.jpg"
        if not img_path.is_file():
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")
        tensor = transform(img).unsqueeze(0)  # Shape (1, 3, 224, 224)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        ref_probs = row[prob_cols].to_numpy(dtype=float)
        diff = np.abs(probs - ref_probs)
        abs_diffs.append(diff)

        pred_cls_idx = int(np.argmax(probs))
        ref_pred_idx = int(row["pred_label"])
        is_disagreement = (pred_cls_idx != ref_pred_idx)
        if is_disagreement:
            argmax_disagreements += 1

        lesion_results.append({
            "lesion_id": lesion_id,
            "isic_id": isic_id,
            "true_class": row["true_class"],
            "ref_pred_class": row["pred_class"],
            "api_pred_class": classes[pred_cls_idx],
            "max_abs_diff": float(np.max(diff)),
            "mean_abs_diff": float(np.mean(diff)),
            "argmax_match": not is_disagreement
        })

    all_diffs = np.concatenate(abs_diffs)
    max_abs_diff = float(np.max(all_diffs))
    mean_abs_diff = float(np.mean(all_diffs))

    print(f"\nParity Results on N=50:")
    print(f"  Max Absolute Probability Difference:  {max_abs_diff:.6e}")
    print(f"  Mean Absolute Probability Difference: {mean_abs_diff:.6e}")
    print(f"  Argmax Disagreements:                 {argmax_disagreements} / 50")

    if argmax_disagreements > 2 or max_abs_diff > 0.01:
        print("\n[CRITICAL FAILURE] Parity check failed gate threshold!")
        print(f"argmax_disagreements={argmax_disagreements} (> 2) or max_abs_diff={max_abs_diff:.4f} (> 0.01)")
    else:
        print("\n[VERIFIED] Parity check PASSED gate threshold (max_diff <= 0.01, disagreements <= 2).")

    # Step 1d: Benchmarking latency & RAM
    print("\n=== Step 1d: Benchmarking Latency & RAM on CPU ===")
    dummy_input = torch.randn(1, 3, 224, 224)

    # Warm-up (5 passes)
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)

    # 20 timed passes
    latencies = []
    with torch.no_grad():
        for _ in range(20):
            t_start = time.perf_counter()
            _ = model(dummy_input)
            latencies.append(time.perf_counter() - t_start)

    median_lat = float(np.median(latencies)) * 1000  # ms
    min_lat = float(np.min(latencies)) * 1000
    max_lat = float(np.max(latencies)) * 1000
    mean_lat = float(np.mean(latencies)) * 1000

    peak_ram = get_ram_usage_mb()
    print(f"Latency over 20 runs: Median={median_lat:.2f} ms, Mean={mean_lat:.2f} ms, Min={min_lat:.2f} ms, Max={max_lat:.2f} ms")
    print(f"Peak RSS RAM: {peak_ram:.1f} MB")

    # Save audit report
    audit_dir = PROJECT_ROOT / "results" / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    report_data = {
        "gate": "GATE 1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_architecture": "PanDermClassifier (panderm_base)",
        "image_size": 224,
        "resize_size": 256,
        "normalization_mean": PANDERM_MEAN,
        "normalization_std": PANDERM_STD,
        "demo_checkpoint_path": str(dest_ckpt.relative_to(PROJECT_ROOT)),
        "demo_checkpoint_size_bytes": ckpt_size_bytes,
        "demo_checkpoint_sha256": ckpt_sha256,
        "model_load_time_seconds": round(t_load, 4),
        "parity_sample_size": len(sample_df),
        "max_abs_prob_diff": max_abs_diff,
        "mean_abs_prob_diff": mean_abs_diff,
        "argmax_disagreements": argmax_disagreements,
        "parity_passed": bool(argmax_disagreements <= 2 and max_abs_diff <= 0.01),
        "latency_median_ms": round(median_lat, 2),
        "latency_mean_ms": round(mean_lat, 2),
        "latency_min_ms": round(min_lat, 2),
        "latency_max_ms": round(max_lat, 2),
        "ram_before_load_mb": round(ram_before_load, 1),
        "ram_after_load_mb": round(ram_after_load, 1),
        "peak_ram_mb": round(peak_ram, 1),
        "sampled_lesions": lesion_results,
    }

    report_json = audit_dir / "gate1_parity_report.json"
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"\nSaved report to {report_json}")

    # Generate Markdown Table
    md_lines = [
        "# GATE 1: Checkpoint Export, Parity Verification, and Benchmarking Report",
        "",
        f"**Date/Time (UTC)**: {report_data['timestamp']}  ",
        f"**Model Architecture**: `{report_data['model_architecture']}`  ",
        f"**Image Size / Normalization**: {report_data['image_size']}px (Resize {report_data['resize_size']}, CenterCrop {report_data['image_size']}), mean={report_data['normalization_mean']}, std={report_data['normalization_std']}  ",
        "",
        "## 1. Checkpoint Export",
        "",
        f"- **Path**: `{report_data['demo_checkpoint_path']}`",
        f"- **File Size**: {report_data['demo_checkpoint_size_bytes']:,} bytes ({report_data['demo_checkpoint_size_bytes'] / (1024*1024):.2f} MB)",
        f"- **SHA-256**: `{report_data['demo_checkpoint_sha256']}`",
        f"- **Load Mode**: `torch.load(..., weights_only=True, map_location='cpu')`",
        f"- **Model Load Time**: {report_data['model_load_time_seconds']:.4f} s",
        "",
        "## 2. Parity Verification (N=50 Test Lesions)",
        "",
        f"- **Sample Size**: {report_data['parity_sample_size']} test lesions (stratified across all test classes)",
        f"- **Max Absolute Probability Difference**: `{report_data['max_abs_prob_diff']:.6e}` (Threshold <= 0.01)",
        f"- **Mean Absolute Probability Difference**: `{report_data['mean_abs_prob_diff']:.6e}`",
        f"- **Argmax Disagreements**: `{report_data['argmax_disagreements']} / 50` (Threshold <= 2)",
        f"- **Parity Status**: `{'PASSED' if report_data['parity_passed'] else 'FAILED'}`",
        "",
        "### Sampled Lesions Breakdown",
        "",
        "| Lesion ID | ISIC ID | True Class | Saved Pred | API Pred | Max Abs Diff | Match |",
        "|:---|:---|:---|:---|:---|:---|:---:|",
    ]
    for item in lesion_results:
        md_lines.append(f"| {item['lesion_id']} | {item['isic_id']} | {item['true_class']} | {item['ref_pred_class']} | {item['api_pred_class']} | {item['max_abs_diff']:.6f} | {'YES' if item['argmax_match'] else 'NO'} |")

    md_lines.extend([
        "",
        "## 3. Latency & Resource Utilization on CPU",
        "",
        "| Metric | Value |",
        "|:---|:---|",
        f"| Median Per-Image Latency (N=20) | {report_data['latency_median_ms']} ms |",
        f"| Mean Per-Image Latency (N=20) | {report_data['latency_mean_ms']} ms |",
        f"| Min Per-Image Latency (N=20) | {report_data['latency_min_ms']} ms |",
        f"| Max Per-Image Latency (N=20) | {report_data['latency_max_ms']} ms |",
        f"| Initial RAM (Process RSS) | {report_data['ram_before_load_mb']} MB |",
        f"| RAM After Model Load | {report_data['ram_after_load_mb']} MB |",
        f"| Peak Process RAM | {report_data['peak_ram_mb']} MB |",
        "",
    ])

    report_md = audit_dir / "gate1_parity_report.md"
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"Saved Markdown report to {report_md}")

if __name__ == "__main__":
    main()
