"""Smoke test script querying the SkinCancerTracker FastAPI service with sample images."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx
import pandas as pd


def get_client(base_url: str):
    """Attempt connecting to live server, fallback to in-process TestClient if offline."""
    try:
        probe = httpx.get(f"{base_url}/api/health", timeout=2.0)
        if probe.status_code == 200:
            print(f"Connected to live running server at {base_url}")
            return httpx.Client(base_url=base_url, timeout=30.0), False
    except Exception:
        pass

    print(f"No running server detected at {base_url}. Running in-process TestClient...")
    from fastapi.testclient import TestClient
    from api.app import app
    return TestClient(app), True


def run_smoke_test(base_url: str = "http://127.0.0.1:8000") -> bool:
    print(f"=== SkinCancerTracker API Smoke Test ===")
    print(f"Target URL: {base_url}\n")

    client, is_in_process = get_client(base_url)

    # 1. GET /api/health
    print("1. Querying GET /api/health ...")
    health_resp = client.get("/api/health")
    print(f"Status Code: {health_resp.status_code}")
    print(f"Payload: {json.dumps(health_resp.json(), indent=2)}\n")
    assert health_resp.status_code == 200, "Health check failed!"

    # 2. GET /api/model-info
    print("2. Querying GET /api/model-info ...")
    info_resp = client.get("/api/model-info")
    print(f"Status Code: {info_resp.status_code}")
    print(f"Payload: {json.dumps(info_resp.json(), indent=2)}\n")
    assert info_resp.status_code == 200, "Model-info check failed!"

    # 3. POST /api/predict with 3 sample images
    print("3. Querying POST /api/predict with 3 sample test lesions ...")
    preds_csv = PROJECT_ROOT / "results" / "exp2c_panderm_uniform_seed42" / "exp2c_panderm_uniform_artifacts" / "predictions_test.csv"
    if not preds_csv.is_file():
        raise FileNotFoundError(f"Missing {preds_csv}")

    preds_df = pd.read_csv(preds_csv)
    image_dir = PROJECT_ROOT / "data" / "images"

    target_classes = ["BCC", "MEL", "NV"]
    samples = []
    for cls in target_classes:
        row = preds_df[preds_df["true_class"] == cls].iloc[0]
        samples.append(row)

    print(f"Selected 3 diverse test samples:")
    for idx, s in enumerate(samples, start=1):
        isic_id = s["isic_id"]
        true_cls = s["true_class"]
        img_path = image_dir / f"{isic_id}.jpg"
        print(f"  Sample {idx}: {isic_id} (True: {true_cls}, File: {img_path.name})")

        with open(img_path, "rb") as f:
            img_bytes = f.read()

        files = {
            "dermoscopic_image": (f"{isic_id}.jpg", img_bytes, "image/jpeg"),
        }
        data = {
            "skin_tone": int(s["skin_tone_class"]) if pd.notna(s["skin_tone_class"]) else 3,
        }

        t_req = time.perf_counter()
        resp = client.post("/api/predict", files=files, data=data)
        req_latency_ms = (time.perf_counter() - t_req) * 1000

        print(f"  Response Status: {resp.status_code} (HTTP Roundtrip: {req_latency_ms:.1f} ms)")
        assert resp.status_code == 200, f"Predict failed with {resp.status_code}: {resp.text}"
        res_json = resp.json()

        print(f"    Predicted Class:      {res_json['predicted_class']}")
        print(f"    Confidence:           {res_json['confidence']:.4f}")
        print(f"    Malignant Prob:       {res_json['malignant_probability']:.4f}")
        print(f"    Server Latency:       {res_json['inference_ms']} ms")
        print(f"    Overlay Available:    {bool(res_json.get('explainability_overlay'))} (chars: {len(res_json.get('explainability_overlay') or '')})")
        print(f"    Input Note:           {res_json['input_note']}")
        print(f"    Top 3 Classes:")
        top3 = sorted(res_json["per_class_probs"].items(), key=lambda x: x[1], reverse=True)[:3]
        for c_name, c_prob in top3:
            print(f"      - {c_name:8s}: {c_prob:.4f}")
        print()

    print(f"=== Smoke Test Successfully Completed ({'In-Process TestClient' if is_in_process else 'Live HTTP Server'}) ===")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke test for SkinCancerTracker API")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Base URL of running service")
    args = parser.parse_args()

    success = run_smoke_test(args.url)
    sys.exit(0 if success else 1)
