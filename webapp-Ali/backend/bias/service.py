"""Backend-facing wrapper around the vendored bias classifier.

Loads a single lazily-initialised ProductionClassifier (the SigLIP/PaddleOCR models
are themselves loaded lazily on first use) and normalises its output to the
uppercase label space the frontend expects (LEFT / LEFT-CENTER / LEAST BIASED /
RIGHT-CENTER / RIGHT).
"""
from __future__ import annotations

import io
import threading
from pathlib import Path

from PIL import Image

from .config import load_settings
from .production_inference import ProductionClassifier

ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "mediascope_artifacts"

_classifier: ProductionClassifier | None = None
_lock = threading.Lock()


def get_classifier() -> ProductionClassifier:
    global _classifier
    if _classifier is None:
        with _lock:
            if _classifier is None:
                _classifier = ProductionClassifier(load_settings(), ARTIFACT_DIR)
    return _classifier


def analyze(png_bytes: bytes) -> dict:
    """png bytes -> {label, confidences, llm_label} with uppercased labels."""
    image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    result = get_classifier().analyze_bias(image, len(png_bytes))
    if result.get("error"):
        raise RuntimeError(result["error"])
    confidences = {str(k).upper(): float(v) for k, v in (result.get("confidences") or {}).items()}
    llm_label = result.get("llm_label")
    return {
        "label": (result.get("label") or "").upper(),
        "confidences": confidences,
        "llm_label": llm_label.upper() if llm_label else None,
    }


def analyze_factuality_evidence(png_bytes: bytes) -> dict:
    """png bytes -> {label, evidence} for the factuality evidence overlay.

    The label is the LLM's own pick from Oleg's 4-class FACTUALITY_LABELS,
    uppercased to match the frontend's factuality casing; it may differ from the
    MLP factuality label on the card. Each evidence item is Oleg's schema:
    {id, kind, role, bbox, text, importance, reason}, with bbox in original
    full-image pixel space [x1, y1, x2, y2].
    """
    image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    result = get_classifier().analyze_factuality_evidence(image)
    if result.get("error"):
        raise RuntimeError(result["error"])
    label = result.get("label")
    return {
        "label": label.upper() if label else None,
        "evidence": result.get("evidence") or [],
    }
