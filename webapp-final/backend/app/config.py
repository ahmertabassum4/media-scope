import os
from dataclasses import dataclass

from dotenv import load_dotenv


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    max_upload_mb: int
    models_api_url: str
    models_request_timeout_seconds: float
    data_dir: str
    cors_origins: list[str]
    # Game response logging -> private HF Dataset repo.
    responses_csv_path: str
    responses_repo_id: str | None
    responses_repo_private: bool
    responses_flush_every: int
    responses_flush_interval_seconds: float
    hf_token: str | None


def load_settings() -> Settings:
    load_dotenv()
    data_dir = os.getenv("DATA_DIR", "/data")
    return Settings(
        max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "20")),
        models_api_url=os.getenv("MODELS_API_URL", "http://models:8001"),
        models_request_timeout_seconds=float(os.getenv("MODELS_REQUEST_TIMEOUT_SECONDS", "360")),
        data_dir=data_dir,
        cors_origins=_split_csv(os.getenv("CORS_ORIGINS", "*")),
        responses_csv_path=os.getenv("GAME_RESPONSES_PATH", "/tmp/game_responses.csv"),
        responses_repo_id=os.getenv("GAME_DATASET_REPO") or None,
        responses_repo_private=_as_bool(os.getenv("GAME_DATASET_PRIVATE", "true")),
        responses_flush_every=int(os.getenv("GAME_RESPONSES_FLUSH_EVERY", "20")),
        responses_flush_interval_seconds=float(os.getenv("GAME_RESPONSES_FLUSH_INTERVAL", "300")),
        # HF_TOKEN is the conventional secret name on Spaces; accept HUGGINGFACE_HUB_TOKEN too.
        hf_token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or None,
    )
