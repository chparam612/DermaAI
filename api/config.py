"""API configuration and environment variable management."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Core constant disclaimer
DISCLAIMER: str = "RESEARCH / DECISION-SUPPORT PROTOTYPE ONLY. Not a diagnostic device."

# Limitations text for /api/model-info
LIMITATIONS_TEXT: str = (
    "This model is a research prototype trained on a single internal partition of the MILK10k dataset. "
    "Tail diagnostic classes (e.g. MAL_OTH, INF, VASC, DF) have low test support (n <= 17) with high variance. "
    "Dermoscopic images only are utilized; clinical close-ups and skin tone metadata are ignored by this model arm. "
    "Not evaluated or approved for clinical diagnostic workflows."
)

# Configuration loaded with environment variable overrides
MODEL_CHECKPOINT_PATH: Path = Path(
    os.getenv("MODEL_CHECKPOINT_PATH", str(PROJECT_ROOT / "checkpoints" / "final" / "demo_model.pt"))
).resolve()

MODEL_VERSION: str = os.getenv("MODEL_VERSION", "exp2c-panderm-vit-base-v1")
DECISION_RULE_T: float = float(os.getenv("DECISION_RULE_T", "0.0"))
DEVICE: str = os.getenv("DEVICE", "cpu")
MAX_IMAGE_SIZE_BYTES: int = int(os.getenv("MAX_IMAGE_SIZE_BYTES", str(10 * 1024 * 1024)))  # 10 MB

# Provenance metadata
SPLIT_SHA256: str = "a4346dcf4d91a8bf6382a26c42618be327975de7f17596ce1fddfbcf1e160c36"
EXPERIMENT_NAME: str = "exp2c_panderm_uniform"
TRAIN_SEED: int = 42

# Parse CORS Origins from environment (comma-separated or single)
raw_cors = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
CORS_ORIGINS: List[str] = [orig.strip() for orig in raw_cors.split(",") if orig.strip()]

# Load class names from repository contract
LABEL_MAP_PATH: Path = PROJECT_ROOT / "configs" / "label_mapping.json"
if LABEL_MAP_PATH.is_file():
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        _label_data = json.load(f)
    CLASSES: List[str] = _label_data["classes"]
else:
    from src.eval.metrics import DEFAULT_CLASSES
    CLASSES: List[str] = DEFAULT_CLASSES

# Malignant classes directly from project evaluation constant
from src.eval.metrics import DEFAULT_MALIGNANT_CLASSES
MALIGNANT_CLASSES: List[str] = list(DEFAULT_MALIGNANT_CLASSES)


def get_checkpoint_sha256(path: Path) -> str:
    """Compute or retrieve full SHA-256 for checkpoint file."""
    if not path.is_file():
        return "UNKNOWN"
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


# Cache checkpoint sha256 at module load
_cached_sha256: str = get_checkpoint_sha256(MODEL_CHECKPOINT_PATH)
CHECKPOINT_SHA256_FULL: str = _cached_sha256
CHECKPOINT_SHA256_SHORT: str = _cached_sha256[:12] if _cached_sha256 != "UNKNOWN" else "UNKNOWN"
