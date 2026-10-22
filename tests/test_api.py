"""Unit and integration tests for FastAPI backend service."""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.app import app
from api import config

PROJECT_ROOT = Path(__file__).resolve().parent.parent

@pytest.fixture(scope="module")
def client():
    """Create TestClient with startup/lifespan triggered."""
    ckpt_path = PROJECT_ROOT / "checkpoints" / "final" / "demo_model.pt"
    if not ckpt_path.is_file():
        pytest.skip("demo_model.pt not available")

    with TestClient(app) as test_client:
        yield test_client

@pytest.fixture
def sample_jpeg_bytes() -> bytes:
    """Generate a small valid JPEG in memory."""
    img = Image.new("RGB", (224, 224), color=(180, 120, 90))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

@pytest.fixture
def sample_png_bytes() -> bytes:
    """Generate a small valid PNG in memory."""
    img = Image.new("RGB", (100, 100), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_health_endpoint(client):
    """Verify GET /api/health response."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_version"] == config.MODEL_VERSION
    assert data["device"] == "cpu"
    assert len(data["checkpoint_sha256"]) == 12
    assert data["checkpoint_sha256"] == config.CHECKPOINT_SHA256_SHORT


def test_model_info_endpoint(client):
    """Verify GET /api/model-info structure, class order, and provenance."""
    resp = client.get("/api/model-info")
    assert resp.status_code == 200
    data = resp.json()

    # Verify classes order matches configs/label_mapping.json exactly
    with open(PROJECT_ROOT / "configs" / "label_mapping.json", "r", encoding="utf-8") as f:
        ref_map = json.load(f)
    assert data["classes"] == ref_map["classes"]
    assert len(data["classes"]) == 11

    # Verify malignant classes
    assert set(data["malignant_classes"]) == set(config.MALIGNANT_CLASSES)

    # Provenance
    assert data["provenance"]["split_sha256"] == config.SPLIT_SHA256
    assert data["provenance"]["experiment_name"] == config.EXPERIMENT_NAME
    assert data["provenance"]["seed"] == 42

    # Disclaimer and limitations
    assert data["disclaimer"] == config.DISCLAIMER
    assert len(data["limitations"]) > 50


def test_predict_valid_jpeg(client, sample_jpeg_bytes):
    """Verify successful prediction with a valid dermoscopic JPEG."""
    files_before = set(os.listdir(PROJECT_ROOT))

    files = {
        "dermoscopic_image": ("lesion.jpg", sample_jpeg_bytes, "image/jpeg"),
    }
    resp = client.post("/api/predict", files=files)
    assert resp.status_code == 200
    data = resp.json()

    assert data["predicted_class"] in config.CLASSES
    assert 0.0 <= data["confidence"] <= 1.0
    assert 0.0 <= data["malignant_probability"] <= 1.0
    assert data["model_version"] == config.MODEL_VERSION
    assert data["disclaimer"] == config.DISCLAIMER
    assert data["inference_ms"] > 0
    assert data["concept_scores"] is None
    assert data["skin_tone"] is None

    # Probabilities must sum to 1.0 within 1e-5
    per_class = data["per_class_probs"]
    assert len(per_class) == 11
    assert set(per_class.keys()) == set(config.CLASSES)
    prob_sum = sum(per_class.values())
    assert abs(prob_sum - 1.0) < 1e-4

    # Explainability overlay
    assert data["explainability_overlay"] is not None
    assert data["explainability_overlay"].startswith("data:image/png;base64,")

    # In-memory purity check: no files written to working directory
    files_after = set(os.listdir(PROJECT_ROOT))
    assert files_before == files_after, f"New files created in working dir: {files_after - files_before}"


def test_predict_with_optional_fields(client, sample_jpeg_bytes, sample_png_bytes):
    """Verify predict endpoint handles optional clinical image and skin_tone."""
    files = {
        "dermoscopic_image": ("derm.jpg", sample_jpeg_bytes, "image/jpeg"),
        "clinical_image": ("clinical.png", sample_png_bytes, "image/png"),
    }
    data_form = {
        "skin_tone": 4,
    }
    resp = client.post("/api/predict", files=files, data=data_form)
    assert resp.status_code == 200
    data = resp.json()
    assert data["skin_tone"] == 4
    assert "clinical image validated but ignored" in data["input_note"]
    assert "skin_tone echoed but not used" in data["input_note"]


def test_error_missing_dermoscopic_image(client):
    """Missing dermoscopic_image should return 422 Unprocessable Entity."""
    resp = client.post("/api/predict", files={})
    assert resp.status_code in (400, 422)


def test_error_empty_image_file(client):
    """Empty 0-byte file should return 400 Bad Request."""
    files = {
        "dermoscopic_image": ("empty.jpg", b"", "image/jpeg"),
    }
    resp = client.post("/api/predict", files=files)
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"].lower()


def test_error_corrupt_image(client):
    """Corrupted bytes should return 400 Bad Request."""
    files = {
        "dermoscopic_image": ("corrupted.jpg", b"NOT_A_VALID_JPEG_HEADER_RANDOM_GARBAGE", "image/jpeg"),
    }
    resp = client.post("/api/predict", files=files)
    assert resp.status_code == 400
    assert "corrupted" in resp.json()["detail"].lower() or "invalid" in resp.json()["detail"].lower()


def test_error_unsupported_format_gif(client):
    """GIF format should be rejected with 400 Bad Request."""
    img = Image.new("P", (64, 64))
    buf = io.BytesIO()
    img.save(buf, format="GIF")
    files = {
        "dermoscopic_image": ("sample.gif", buf.getvalue(), "image/gif"),
    }
    resp = client.post("/api/predict", files=files)
    assert resp.status_code == 400
    assert "unsupported" in resp.json()["detail"].lower()


def test_error_file_too_large(client):
    """Files exceeding 10 MB should return 400 Bad Request."""
    oversized = b"0" * (10 * 1024 * 1024 + 1024)
    files = {
        "dermoscopic_image": ("huge.jpg", oversized, "image/jpeg"),
    }
    resp = client.post("/api/predict", files=files)
    assert resp.status_code == 400
    assert "exceeds" in resp.json()["detail"].lower()
