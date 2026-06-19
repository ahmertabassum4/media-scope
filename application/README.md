# Media bias and Factuality

Microservice web application for screenshot-only media bias and factuality classification.

## Services

- `frontend`: nginx static UI on port `8080`
- `backend`: FastAPI API on port `8000`
- `models`: FastAPI model inference service on port `8001`

## Features

- Upload an image and get bias/factuality labels from the production pipelines.
- Play a labeling game using screenshots and true labels from `data/manifest.json`.
- No request or image caching is used in the backend game endpoints.

## Run

```bash
docker compose up -d --build
```

Open:

- UI: `http://localhost:8080`
- Backend health: `http://localhost:8000/api/health`
- Models health: `http://localhost:8001/health`

## Configuration

The app reads `.env` from this folder. Use `.env.example` as a template.

Required for LLM inference:

```bash
OPENROUTER_API_KEY=...
```

Without the key, the containers can start, but the production hybrid inference path cannot complete LLM-backed predictions.

## Inference Methods

- Bias: `nested_stack::core_gpt55_ocr_text_siglip`
- Factuality: `baseline_plus_semantic_visual_extra_trees_cv`

## Game Data

The game uses `data/manifest.json` and `data/images/*.jpg`.

Current dataset:

- 65 screenshots
- bias labels: `left`, `left-center`, `least biased`, `right-center`, `right`
- factuality labels: `very low`, `low`, `high`, `very high`

Images are first-screen 16:9 viewport crops generated from the source screenshots to keep the web experience responsive.
