# Single-container build for Hugging Face Spaces (Docker SDK).
# HF runs ONE container on ONE port, so the three compose services (frontend/backend/
# models) are merged here and supervised together behind nginx on port 7860.
#
# Base is a CUDA runtime so the models service can use the Space's T4 GPU. The image
# also runs fine on CPU-only hardware (the CUDA torch wheels fall back to CPU).
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

# Python 3.11 + nginx + supervisor + OpenCV/Paddle runtime libs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends software-properties-common ca-certificates curl \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-distutils \
        nginx supervisor \
        libgl1 libglib2.0-0 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- models service (heavy ML deps: CUDA torch, transformers, paddle, etc.) ---
COPY models/requirements.txt /app/models_service/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /app/models_service/requirements.txt

# --- backend service (FastAPI + Playwright/Chromium for URL screenshots) ---
COPY backend/requirements.txt /app/backend/requirements.txt
RUN python -m pip install -r /app/backend/requirements.txt \
    && python -m playwright install --with-deps chromium

# Application code + model weights + game data.
COPY models/app /app/models_service/app
COPY models/models /app/models_service/models
COPY backend/app /app/backend/app
COPY data /app/backend/data

# Static frontend served by nginx.
COPY frontend/index.html frontend/styles.css frontend/app.js /usr/share/nginx/html/
COPY frontend/memes /usr/share/nginx/html/memes

# Process manager + reverse proxy config.
COPY deploy/nginx-hf.conf /etc/nginx/conf.d/default.conf
COPY deploy/supervisord.conf /etc/supervisor/conf.d/app.conf
RUN rm -f /etc/nginx/sites-enabled/default 2>/dev/null || true

# Internal wiring for the backend -> models hop inside this one container.
ENV MODELS_API_URL=http://127.0.0.1:8001 \
    DATA_DIR=/app/backend/data \
    GAME_RESPONSES_PATH=/tmp/game_responses.csv \
    MODELS_REQUEST_TIMEOUT_SECONDS=360 \
    CORS_ORIGINS=* \
    HF_HOME=/tmp/hf-cache \
    PYTHONPATH=/app/backend:/app/models_service

EXPOSE 7860

CMD ["supervisord", "-c", "/etc/supervisor/conf.d/app.conf"]
