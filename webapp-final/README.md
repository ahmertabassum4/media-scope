---
title: MediaScope
emoji: 📰
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Media bias & factuality classifier + labeling game
---

# Media bias and Factuality

Microservice web application for media bias and factuality classification from a
screenshot **or** a live URL.

> **Deploying to Hugging Face Spaces?** See [DEPLOY.md](DEPLOY.md). On Spaces the three
> services are merged into one container (root `Dockerfile` + `deploy/`) served on port
> `7860`, GPU-accelerated on a T4, and game answers are saved to a private HF Dataset.

## Services

- `frontend`: nginx static UI on port `8080`
- `backend`: FastAPI API on port `8000` (also captures URL screenshots + provenance)
- `models`: FastAPI model inference service on port `8001`

## Features

- Analyze a homepage by **uploading an image** or **entering a URL**.
  - URL mode captures a full-page screenshot (Playwright/Chromium) and crawls provenance
    so the metadata factuality model can run; image mode uses the image-only model.
- Bias + factuality labels with **bounding-box evidence overlays** on the screenshot.
- On-demand **Reasoning**: an "Explain" button requests a combined factuality + bias
  rationale grounded in the screenshot and the highlighted evidence regions.
- Play a labeling game using screenshots and true labels from `data/manifest.json`.
- No request or image caching is used in the backend game endpoints.

## Factuality models (place before running)

The factuality head is a 5-class MLP trained in Colab (see the training notebook's export
cells). Drop these four files into `models/models/` next to the bias `.joblib`:

- `factuality_metadata_mlp.pt`  — full features incl. provenance (used for URL analyses)
- `factuality_image_mlp.pt`     — screenshot-only features (used for image uploads)
- `factuality_config.json`      — labels, dims, hidden/dropout, feature order
- `factuality_prov_stats.json`  — provenance keys + standardization stats

The `models` service loads them on startup, so they must be present before
`docker compose up`.

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

- Bias: `nested_stack::core_gpt55_ocr_text_siglip` (gpt-5.5 + OCR + SigLIP stack)
- Factuality: Ali's MLP over DINOv2 visual + joint-verdict one-hot + MiniLM rationale
  (+ provenance for the metadata variant). A joint-verdict LLM call supplies the verdict
  one-hot and rationale; a separate LLM call supplies the evidence bounding boxes.

## Game Data

The game uses `data/manifest.json` and `data/images/*.jpg`.

Current dataset:

- 65 screenshots
- bias labels: `left`, `left-center`, `least biased`, `right-center`, `right`
- factuality labels: `very low`, `low`, `mixed`, `high`, `very high` (slider is 5-class;
  the bundled ground-truth labels are 4-class, so `mixed` is never the correct answer
  until MIXED-labeled items are added to the manifest).

Images are first-screen 16:9 viewport crops generated from the source screenshots to keep the web experience responsive.
