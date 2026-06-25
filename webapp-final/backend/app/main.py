from __future__ import annotations

import base64
import json
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app.config import load_settings
from app.game import GameDataset
from app.schemas import (
    AnalyzeResponse,
    GameAnswerRequest,
    GameAnswerResult,
    GameItemResponse,
    HealthResponse,
    ReasonRequest,
    ReasonResponse,
)


settings = load_settings()
game_dataset = GameDataset(Path(settings.data_dir))

app = FastAPI(title="Media Classification Backend", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


async def models_health() -> dict:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{settings.models_api_url}/health")
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        return {"status": "unavailable", "error": str(exc)}


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(models=await models_health(), game_items=game_dataset.count)


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_image(
    file: UploadFile | None = File(None),
    url: str | None = Form(None),
) -> JSONResponse:
    max_bytes = settings.max_upload_mb * 1024 * 1024
    provenance_json: str | None = None
    filename = "uploaded-image"
    content_type = "application/octet-stream"

    if url and url.strip():
        # URL path: capture a full-page screenshot and crawl provenance so the
        # models service can use the metadata factuality model.
        from app.provenance import build_provenance
        from app.screenshot import capture

        try:
            payload, html, final_url, host = await capture(url.strip())
        except Exception as exc:  # noqa: BLE001 - surfaced to caller as 502
            raise HTTPException(status_code=502, detail=f"Screenshot capture failed: {exc}") from exc
        filename = f"{host or 'url'}.png"
        content_type = "image/png"
        try:
            provenance_json = json.dumps(build_provenance(html, final_url, host))
        except Exception:  # noqa: BLE001 - degraded provenance falls back to the image-only model
            provenance_json = None
    elif file is not None:
        payload = await file.read()
        filename = file.filename or filename
        content_type = file.content_type or content_type
    else:
        raise HTTPException(status_code=400, detail="Provide an image file or a url")

    if not payload:
        raise HTTPException(status_code=400, detail="Image file is required")
    if len(payload) > max_bytes:
        raise HTTPException(status_code=413, detail=f"Image is larger than {settings.max_upload_mb} MB")

    files = {"file": (filename, payload, content_type)}
    data = {"provenance": provenance_json} if provenance_json else None
    try:
        async with httpx.AsyncClient(timeout=settings.models_request_timeout_seconds) as client:
            response = await client.post(f"{settings.models_api_url}/predict", files=files, data=data)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Models service unavailable: {exc}") from exc

    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise HTTPException(status_code=response.status_code, detail=detail)

    result = response.json()
    # Return the analyzed image so the UI can render URL captures and feed /api/reason.
    result["screenshot_b64"] = base64.b64encode(payload).decode("ascii")
    return JSONResponse(result, headers={"Cache-Control": "no-store"})


@app.post("/api/reason", response_model=ReasonResponse)
async def reason(req: ReasonRequest) -> JSONResponse:
    try:
        async with httpx.AsyncClient(timeout=settings.models_request_timeout_seconds) as client:
            response = await client.post(f"{settings.models_api_url}/reason", json=req.model_dump())
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Models service unavailable: {exc}") from exc

    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise HTTPException(status_code=response.status_code, detail=detail)

    return JSONResponse(response.json(), headers={"Cache-Control": "no-store"})


@app.get("/api/game/random", response_model=GameItemResponse)
def random_game_item() -> GameItemResponse:
    try:
        item = game_dataset.random_item()
    except LookupError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return GameItemResponse(id=item.id, image_url=f"/api/game/image/{item.id}")


@app.get("/api/game/image/{item_id}")
def game_image(item_id: str) -> FileResponse:
    try:
        path = game_dataset.image_path(item_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.post("/api/game/answer", response_model=GameAnswerResult)
def answer_game_item(answer: GameAnswerRequest) -> GameAnswerResult:
    try:
        result = game_dataset.grade(answer.id, answer.bias, answer.factuality)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return GameAnswerResult(**result)
