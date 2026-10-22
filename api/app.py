"""FastAPI application for SkinCancerTracker inference backend."""
from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from typing import Optional
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api import config
from api.inference import ImageValidationError, InferenceEngine
from api.schemas import HealthResponse, ModelInfoResponse, PredictResponse, ProvenanceInfo

logger = logging.getLogger("api.app")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Global inference engine instance loaded once at startup
engine: Optional[InferenceEngine] = None


def get_engine() -> InferenceEngine:
    """Retrieve or lazily initialize global inference engine."""
    global engine
    if engine is None:
        logger.info("Initializing SkinCancerTracker inference backend...")
        engine = InferenceEngine()
        logger.info("SkinCancerTracker inference backend ready to receive requests.")
    return engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager ensuring model loads once at startup into memory."""
    get_engine()
    yield
    logger.info("Shutting down inference backend.")


app = FastAPI(
    title="SkinCancerTracker Inference Backend",
    version=config.MODEL_VERSION,
    description="Research & Decision-Support prototype for dermatological diagnostic classification.",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ImageValidationError)
async def image_validation_exception_handler(request: Request, exc: ImageValidationError):
    """Handle custom image validation errors cleanly with HTTP 400."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "detail": str(exc),
            "disclaimer": config.DISCLAIMER,
        },
    )


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Service health and model provenance endpoint."""
    return HealthResponse(
        status="ok",
        model_version=config.MODEL_VERSION,
        device=config.DEVICE,
        checkpoint_sha256=config.CHECKPOINT_SHA256_SHORT,
    )


@app.get("/api/model-info", response_model=ModelInfoResponse)
async def model_info():
    """Model information, diagnostic classes, training provenance, and clinical limitations."""
    return ModelInfoResponse(
        model_version=config.MODEL_VERSION,
        classes=config.CLASSES,
        malignant_classes=config.MALIGNANT_CLASSES,
        provenance=ProvenanceInfo(
            split_sha256=config.SPLIT_SHA256,
            experiment_name=config.EXPERIMENT_NAME,
            seed=config.TRAIN_SEED,
        ),
        limitations=config.LIMITATIONS_TEXT,
        disclaimer=config.DISCLAIMER,
    )


@app.post("/api/predict", response_model=PredictResponse)
async def predict_endpoint(
    dermoscopic_image: UploadFile = File(..., description="Required dermoscopic image file (JPEG or PNG, max 10MB)"),
    clinical_image: Optional[UploadFile] = File(None, description="Optional clinical close-up image (ignored by model)"),
    skin_tone: Optional[int] = Form(None, description="Optional skin tone index (echoed back, not used in prediction)"),
):
    """Diagnostic prediction endpoint accepting multipart form data."""
    inference_engine = get_engine()

    # 1. Read dermoscopic image bytes
    if not dermoscopic_image:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing required multipart field 'dermoscopic_image'.",
        )

    derm_bytes = await dermoscopic_image.read()
    if len(derm_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded 'dermoscopic_image' is empty (0 bytes).",
        )

    # 2. Read optional clinical image bytes
    clinical_bytes = None
    if clinical_image is not None and clinical_image.filename:
        clinical_bytes = await clinical_image.read()
        if len(clinical_bytes) == 0:
            clinical_bytes = None

    # 3. Perform in-memory inference and rollout explainability
    try:
        response = inference_engine.predict(
            dermoscopic_bytes=derm_bytes,
            clinical_bytes=clinical_bytes,
            skin_tone=skin_tone,
            include_overlay=True,
        )
        return response
    except ImageValidationError as ive:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ive))
    except Exception as exc:
        logger.error(f"Prediction failed with internal error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal inference error occurred during processing.",
        )
