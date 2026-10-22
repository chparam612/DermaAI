"""Inference engine managing model lifecycle, in-memory image validation, and predictions."""
from __future__ import annotations

import io
import logging
from pathlib import Path
import time
from typing import Dict, Optional, Tuple
import numpy as np
from PIL import Image, UnidentifiedImageError
import torch

from api import config
from api.explain import AttentionRollout
from api.schemas import PredictResponse
from src.data.transforms import get_panderm_eval_transforms
from src.models.panderm import PanDermClassifier

logger = logging.getLogger("api.inference")


class ImageValidationError(ValueError):
    """Raised when an uploaded image fails validation."""
    pass


class InferenceEngine:
    """Encapsulates the PyTorch model, evaluation pipeline, and attention rollout."""

    def __init__(self, checkpoint_path: Optional[Path] = None):
        """Initialize engine and load model weights strictly into CPU memory."""
        self.checkpoint_path = checkpoint_path or config.MODEL_CHECKPOINT_PATH
        self.device = config.DEVICE
        self.classes = config.CLASSES
        self.malignant_classes = set(config.MALIGNANT_CLASSES)
        self.transform = get_panderm_eval_transforms(image_size=224)

        logger.info(f"Loading PanDermClassifier from {self.checkpoint_path} onto {self.device}...")
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found at: {self.checkpoint_path}")

        self.model = PanDermClassifier(
            architecture="panderm_base",
            num_classes=len(self.classes),
            pretrained=False,
        )
        state_dict = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=True)
        if len(missing) > 0 or len(unexpected) > 0:
            raise RuntimeError(f"Checkpoint mismatch! Missing: {missing}, Unexpected: {unexpected}")

        self.model.eval()
        self.rollout_engine = AttentionRollout(self.model)
        logger.info("InferenceEngine initialized successfully and set to eval mode.")

    def validate_and_load_image(self, image_bytes: bytes, field_name: str) -> Image.Image:
        """Validate raw bytes in memory and convert to PIL Image.

        Rules:
        - Image size <= MAX_IMAGE_SIZE_BYTES (10 MB).
        - Verify valid JPEG or PNG header and body using PIL.verify().
        - Images are processed in-memory and never written to disk or logged.
        """
        if not image_bytes or len(image_bytes) == 0:
            raise ImageValidationError(f"Field '{field_name}' is empty.")

        if len(image_bytes) > config.MAX_IMAGE_SIZE_BYTES:
            size_mb = len(image_bytes) / (1024 * 1024)
            max_mb = config.MAX_IMAGE_SIZE_BYTES / (1024 * 1024)
            raise ImageValidationError(
                f"Image in '{field_name}' ({size_mb:.2f} MB) exceeds maximum allowed size ({max_mb:.1f} MB)."
            )

        # First pass: identify format and verify integrity
        try:
            temp_stream = io.BytesIO(image_bytes)
            with Image.open(temp_stream) as probe_img:
                img_format = probe_img.format
                if img_format not in ("JPEG", "PNG"):
                    raise ImageValidationError(
                        f"Unsupported format '{img_format}' in '{field_name}'. Only JPEG and PNG are accepted."
                    )
                probe_img.verify()
        except ImageValidationError:
            raise
        except (UnidentifiedImageError, OSError, Exception) as exc:
            raise ImageValidationError(f"Invalid or corrupted image in '{field_name}': {exc}") from exc

        # Second pass: decode pixels safely into RGB
        try:
            decode_stream = io.BytesIO(image_bytes)
            img = Image.open(decode_stream).convert("RGB")
            # Force pixel load into memory
            img.load()
            return img
        except Exception as exc:
            raise ImageValidationError(f"Failed to decode image pixels in '{field_name}': {exc}") from exc

    @torch.no_grad()
    def predict(
        self,
        dermoscopic_bytes: bytes,
        clinical_bytes: Optional[bytes] = None,
        skin_tone: Optional[int] = None,
        include_overlay: bool = True,
    ) -> PredictResponse:
        """Execute end-to-end inference on in-memory images.

        Args:
            dermoscopic_bytes: Raw bytes of the required dermoscopic image.
            clinical_bytes: Optional raw bytes of clinical close-up (validated if provided, then ignored).
            skin_tone: Optional skin tone category (echoed back in response).
            include_overlay: Whether to generate attention rollout overlay data URI.

        Returns:
            PredictResponse object.
        """
        t0 = time.perf_counter()

        # 1. Validate required dermoscopic image
        derm_img = self.validate_and_load_image(dermoscopic_bytes, "dermoscopic_image")

        # 2. Validate optional clinical image if provided (then ignore)
        clinical_provided = False
        if clinical_bytes is not None and len(clinical_bytes) > 0:
            _ = self.validate_and_load_image(clinical_bytes, "clinical_image")
            clinical_provided = True

        # 3. Preprocessing (deterministic PanDerm eval transform)
        input_tensor = self.transform(derm_img).unsqueeze(0).to(self.device)

        # 4. Model forward pass
        logits = self.model(input_tensor)

        # 5. Decision rule adjustment if t != 0.0 (default 0.0: standard raw logit argmax)
        if config.DECISION_RULE_T != 0.0:
            pass  # Extensible for post-hoc logit adjustment
        
        probs_tensor = torch.softmax(logits, dim=-1).squeeze(0).cpu()
        probs_np = probs_tensor.numpy()

        pred_idx = int(np.argmax(probs_np))
        predicted_class = self.classes[pred_idx]
        confidence = float(probs_np[pred_idx])

        # Per-class probabilities dictionary
        per_class_probs: Dict[str, float] = {
            cls_name: float(probs_np[i]) for i, cls_name in enumerate(self.classes)
        }

        # Malignant probability (sum of probabilities for classes in MALIGNANT_CLASSES set)
        malignant_prob = float(
            sum(probs_np[i] for i, cls_name in enumerate(self.classes) if cls_name in self.malignant_classes)
        )

        # 6. Attention rollout overlay
        overlay_data_uri: Optional[str] = None
        if include_overlay:
            try:
                overlay_data_uri = self.rollout_engine.generate_overlay_data_uri(
                    input_tensor,
                    target_size=derm_img.size,
                )
            except Exception as e:
                logger.warning(f"Attention rollout overlay generation failed: {e}")
                overlay_data_uri = None

        inference_ms = round((time.perf_counter() - t0) * 1000, 2)

        # Input modality notes
        notes = []
        if clinical_provided:
            notes.append("clinical image validated but ignored by this single-modality model")
        else:
            notes.append("clinical image not provided/not used")

        if skin_tone is not None:
            notes.append("skin_tone echoed but not used by model")
        else:
            notes.append("skin_tone not provided/not used")

        input_note = "; ".join(notes)

        return PredictResponse(
            predicted_class=predicted_class,
            confidence=round(confidence, 6),
            malignant_probability=round(malignant_prob, 6),
            per_class_probs={k: round(v, 6) for k, v in per_class_probs.items()},
            concept_scores=None,
            explainability_overlay=overlay_data_uri,
            model_version=config.MODEL_VERSION,
            disclaimer=config.DISCLAIMER,
            inference_ms=inference_ms,
            input_note=input_note,
            skin_tone=skin_tone,
        )
