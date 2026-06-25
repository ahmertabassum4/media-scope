from __future__ import annotations

import asyncio
import base64
import binascii
import io
import json
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

from app.config import load_settings
from app.image_utils import ImageValidationError, crop_to_first_screen, load_image_from_bytes
from app.production_inference import ProductionClassifier
from app.schemas import AnalyzeResponse, ModelsHealthResponse, ReasonRequest, ReasonResponse


settings = load_settings()
model_dir = Path(__file__).resolve().parents[1] / "models"
classifier = ProductionClassifier(settings=settings, model_dir=model_dir)

app = FastAPI(title="Media Models API", version="1.1.0")


@app.get("/health", response_model=ModelsHealthResponse)
def health() -> ModelsHealthResponse:
    return ModelsHealthResponse(
        openrouter_configured=bool(settings.openrouter_api_key),
        bias_openrouter_model=settings.bias_openrouter_model,
        factuality_openrouter_model=settings.factuality_openrouter_model,
        bias_pipeline_loaded=True,
        factuality_pipeline_loaded=True,
    )


@app.post("/predict", response_model=AnalyzeResponse)
async def predict(
    file: UploadFile = File(...),
    provenance: str | None = Form(None),
) -> AnalyzeResponse:
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Image file is required")

    try:
        image, source_format = load_image_from_bytes(payload, filename=file.filename)
    except ImageValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    prov: dict | None = None
    if provenance:
        try:
            prov = json.loads(provenance)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid provenance JSON: {exc}") from exc
        if not isinstance(prov, dict):
            prov = None

    _, crop_info = crop_to_first_screen(image)
    bias, factuality = await asyncio.to_thread(classifier.predict, image, len(payload), prov)

    return AnalyzeResponse(
        image={
            "filename": file.filename or "uploaded-image",
            "source_format": source_format,
            "bytes": len(payload),
            "original_size": {"width": image.width, "height": image.height},
            "analyzed_crop": crop_info,
        },
        bias=bias,
        factuality=factuality,
    )


@app.post("/reason", response_model=ReasonResponse)
async def reason(req: ReasonRequest) -> ReasonResponse:
    try:
        png = base64.b64decode(req.screenshot_b64)
        image = Image.open(io.BytesIO(png))
        image.load()
    except (binascii.Error, ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid screenshot: {exc}") from exc

    try:
        text = await asyncio.to_thread(
            classifier.reason,
            image,
            req.factuality_label,
            req.bias_label,
            req.evidence,
        )
    except Exception as exc:  # noqa: BLE001 - surfaced to caller as 502
        raise HTTPException(status_code=502, detail=f"Reasoning failed: {exc}") from exc

    return ReasonResponse(reasoning=text)
