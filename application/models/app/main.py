from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.config import load_settings
from app.image_utils import ImageValidationError, crop_to_first_screen, load_image_from_bytes
from app.production_inference import ProductionClassifier
from app.schemas import AnalyzeResponse, ModelsHealthResponse


settings = load_settings()
model_dir = Path(__file__).resolve().parents[1] / "models"
classifier = ProductionClassifier(settings=settings, model_dir=model_dir)

app = FastAPI(title="Media Models API", version="1.0.0")


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
async def predict(file: UploadFile = File(...)) -> AnalyzeResponse:
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Image file is required")

    try:
        image, source_format = load_image_from_bytes(payload, filename=file.filename)
    except ImageValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _, crop_info = crop_to_first_screen(image)
    bias, factuality = await asyncio.to_thread(classifier.predict, image, len(payload))

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
