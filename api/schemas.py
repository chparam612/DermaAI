"""Pydantic schemas for SkinCancerTracker FastAPI inference service."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response schema for GET /api/health."""
    status: str = Field(..., description="Operational status of the service (e.g., 'ok')")
    model_version: str = Field(..., description="Version identifier of the active model")
    device: str = Field(..., description="Compute device utilized for inference ('cpu' or 'cuda')")
    checkpoint_sha256: str = Field(..., description="First 12 characters of the model weights SHA-256")


class ProvenanceInfo(BaseModel):
    """Training and evaluation provenance."""
    split_sha256: str = Field(..., description="SHA-256 hash of the exact split partition CSV")
    experiment_name: str = Field(..., description="Baseline experiment arm name")
    seed: int = Field(..., description="Random seed used during training")


class ModelInfoResponse(BaseModel):
    """Response schema for GET /api/model-info."""
    model_version: str = Field(..., description="Version identifier of the active model")
    classes: List[str] = Field(..., description="List of all 11 diagnostic classes in index order")
    malignant_classes: List[str] = Field(..., description="Subset of classes classified as malignant")
    provenance: ProvenanceInfo = Field(..., description="Split and experiment provenance metadata")
    limitations: str = Field(..., description="Clinical limitations, tail-class notes, and dataset caveats")
    disclaimer: str = Field(..., description="Mandatory decision-support regulatory disclaimer")


class PredictResponse(BaseModel):
    """Response schema for POST /api/predict."""
    predicted_class: str = Field(..., description="Predicted diagnostic class name")
    confidence: float = Field(..., description="Maximum predicted class probability (in [0, 1])")
    malignant_probability: float = Field(..., description="Summed probability across all malignant classes")
    per_class_probs: Dict[str, float] = Field(..., description="Softmax probabilities for all 11 diagnostic classes")
    concept_scores: Optional[Dict[str, float]] = Field(None, description="MONET concept bottleneck scores (null in Phase 6-lite)")
    explainability_overlay: Optional[str] = Field(None, description="Base64 data URI of RGBA attention rollout overlay or null")
    model_version: str = Field(..., description="Model version string")
    disclaimer: str = Field(..., description="Mandatory decision-support regulatory disclaimer")
    inference_ms: float = Field(..., description="Total wall-clock inference time in milliseconds")
    input_note: str = Field(..., description="Explanatory note regarding optional/ignored input modalities")
    skin_tone: Optional[int] = Field(None, description="Skin tone category echoed back from input if provided")


class ErrorResponse(BaseModel):
    """Standardized error response payload."""
    detail: str = Field(..., description="Descriptive error explanation")
    disclaimer: str = Field(..., description="Regulatory decision-support disclaimer")
