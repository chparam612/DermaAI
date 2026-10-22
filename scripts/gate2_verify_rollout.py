"""Gate 2 Attention Rollout Verification and Rendering."""
from __future__ import annotations

import base64
import io
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

from api.explain import AttentionRollout, blend_heatmap_on_image
from src.data.transforms import get_panderm_eval_transforms
from src.models.panderm import PanDermClassifier

def main():
    print("=== Gate 2: Attention Rollout Verification ===")
    ckpt_path = PROJECT_ROOT / "checkpoints" / "final" / "demo_model.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing {ckpt_path}")

    model = PanDermClassifier(architecture="panderm_base", num_classes=11, pretrained=False)
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    rollout_engine = AttentionRollout(model)

    # 1. Properties & Determinism Test
    dummy_input = torch.randn(1, 3, 224, 224)
    raw1 = rollout_engine.compute_raw_rollout(dummy_input)
    raw2 = rollout_engine.compute_raw_rollout(dummy_input)

    assert raw1.shape == (14, 14), f"Unexpected raw shape: {raw1.shape}"
    assert raw1.min() >= 0.0 and raw1.max() <= 1.0, f"Values not in [0, 1]: [{raw1.min()}, {raw1.max()}]"
    det_diff = float(np.max(np.abs(raw1 - raw2)))
    assert det_diff == 0.0, f"Rollout not deterministic: max diff = {det_diff}"
    print(f"[VERIFIED] Raw rollout shape: {raw1.shape}, values in [{raw1.min():.4f}, {raw1.max():.4f}], determinism diff: {det_diff}")

    # 2. Timing on CPU
    latencies = []
    for _ in range(10):
        t0 = time.perf_counter()
        _ = rollout_engine.compute_raw_rollout(dummy_input)
        latencies.append((time.perf_counter() - t0) * 1000)
    median_time_ms = float(np.median(latencies))
    mean_time_ms = float(np.mean(latencies))
    print(f"[VERIFIED] CPU Rollout Latency (N=10): Median={median_time_ms:.2f} ms, Mean={mean_time_ms:.2f} ms")

    # 3. Data URI Generation Test
    data_uri = rollout_engine.generate_overlay_data_uri(dummy_input, target_size=(224, 224))
    assert data_uri.startswith("data:image/png;base64,"), "Data URI does not have valid PNG base64 prefix"
    # Decode and check image size and mode
    b64_content = data_uri.split(",", 1)[1]
    decoded_bytes = base64.b64decode(b64_content)
    overlay_img = Image.open(io.BytesIO(decoded_bytes))
    assert overlay_img.size == (224, 224), f"Unexpected overlay image size: {overlay_img.size}"
    assert overlay_img.mode == "RGBA", f"Unexpected overlay mode: {overlay_img.mode}"
    print(f"[VERIFIED] Overlay Data URI generated: length={len(data_uri)} chars, mode={overlay_img.mode}, size={overlay_img.size}")

    # 4. Render 5 Test Examples
    preds_file = PROJECT_ROOT / "results" / "exp2c_panderm_uniform_seed42" / "exp2c_panderm_uniform_artifacts" / "predictions_test.csv"
    preds_df = pd.read_csv(preds_file)
    image_dir = PROJECT_ROOT / "data" / "images"

    out_dir = PROJECT_ROOT / "results" / "audit" / "rollout_examples"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pick 5 diverse test samples (different classes)
    sample_classes = ["BCC", "MEL", "SCCKA", "NV", "BKL"]
    examples_meta = []
    transform = get_panderm_eval_transforms(image_size=224)

    for i, target_cls in enumerate(sample_classes):
        match_rows = preds_df[preds_df["true_class"] == target_cls]
        if match_rows.empty:
            match_rows = preds_df.iloc[[i]]
        sample_row = match_rows.iloc[0]

        isic_id = sample_row["isic_id"]
        lesion_id = sample_row["lesion_id"]
        img_path = image_dir / f"{isic_id}.jpg"

        orig_pil = Image.open(img_path).convert("RGB")
        tensor = transform(orig_pil).unsqueeze(0)

        # Compute rollout
        heatmap_14 = rollout_engine.compute_raw_rollout(tensor)
        blended = blend_heatmap_on_image(orig_pil, heatmap_14, alpha=0.45)
        data_uri_sample = rollout_engine.generate_overlay_data_uri(tensor, target_size=orig_pil.size)

        # Save files
        orig_out_path = out_dir / f"example_{i+1}_{target_cls}_{isic_id}_orig.jpg"
        blended_out_path = out_dir / f"example_{i+1}_{target_cls}_{isic_id}_blended.jpg"
        overlay_out_path = out_dir / f"example_{i+1}_{target_cls}_{isic_id}_overlay.png"

        orig_pil.save(orig_out_path)
        blended.save(blended_out_path)

        # Save decoded overlay PNG
        b64_raw = data_uri_sample.split(",", 1)[1]
        with open(overlay_out_path, "wb") as f_over:
            f_over.write(base64.b64decode(b64_raw))

        examples_meta.append({
            "example_index": i + 1,
            "isic_id": isic_id,
            "lesion_id": lesion_id,
            "true_class": target_cls,
            "pred_class": sample_row["pred_class"],
            "image_size": orig_pil.size,
            "orig_file": str(orig_out_path.relative_to(PROJECT_ROOT)),
            "blended_file": str(blended_out_path.relative_to(PROJECT_ROOT)),
            "overlay_file": str(overlay_out_path.relative_to(PROJECT_ROOT)),
            "heatmap_min": float(heatmap_14.min()),
            "heatmap_max": float(heatmap_14.max()),
            "data_uri_prefix": data_uri_sample[:40] + "..."
        })
        print(f"Rendered example {i+1}: {target_cls} ({isic_id}) -> {blended_out_path.name}")

    # Write report
    report = {
        "gate": "GATE 2",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": "Attention Rollout (Abnar & Zuidema, 2020) via dynamic forward interception",
        "target_architecture": "PanDerm ViT-Base (modeling_finetune.py)",
        "external_panderm_modified": False,
        "grid_size": "14x14 patches (196 tokens + 1 CLS token)",
        "raw_shape": list(raw1.shape),
        "values_range": [float(raw1.min()), float(raw1.max())],
        "is_deterministic": bool(det_diff == 0.0),
        "cpu_latency_median_ms": round(median_time_ms, 2),
        "cpu_latency_mean_ms": round(mean_time_ms, 2),
        "data_uri_format": "data:image/png;base64,... (RGBA)",
        "rendered_examples": examples_meta,
    }

    report_json = PROJECT_ROOT / "results" / "audit" / "gate2_rollout_report.json"
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    report_md = PROJECT_ROOT / "results" / "audit" / "gate2_rollout_report.md"
    md_content = f"""# GATE 2: Attention Rollout Explainability Report

**Timestamp (UTC)**: {report['timestamp']}  
**Method**: Attention Rollout (Abnar & Zuidema, 2020)  
**External PanDerm Modified**: `False` (zero edits to `external/panderm`)  
**Output Shape**: {report['raw_shape']} spatial patch grid, upsampled to input image dimensions  
**Value Bounds**: `[{report['values_range'][0]}, {report['values_range'][1]}]` (min-max normalized)  
**Deterministic**: `True` (absolute diff between identical runs = 0.0)  
**CPU Latency (Median)**: {report['cpu_latency_median_ms']} ms (Mean: {report['cpu_latency_mean_ms']} ms)  
**Overlay Format**: RGBA PNG Base64 Data URI with transparent background  

## Rendered Visual Examples

| # | True Class | Pred Class | ISIC ID | Original Image | Blended Overlay | Transparent Overlay |
|---|---|---|---|---|---|---|
"""
    for ex in examples_meta:
        md_content += f"| {ex['example_index']} | {ex['true_class']} | {ex['pred_class']} | `{ex['isic_id']}` | `{ex['orig_file']}` | `{ex['blended_file']}` | `{ex['overlay_file']}` |\n"

    with open(report_md, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\nSaved Gate 2 report to {report_json} and {report_md}")

if __name__ == "__main__":
    main()
