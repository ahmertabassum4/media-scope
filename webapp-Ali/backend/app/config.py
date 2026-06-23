import os
from dataclasses import dataclass

from dotenv import load_dotenv


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


@dataclass(frozen=True)
class Settings:
    max_upload_mb: int
    models_api_url: str
    models_request_timeout_seconds: float
    data_dir: str
    cors_origins: list[str]


def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "20")),
        models_api_url=os.getenv("MODELS_API_URL", "http://models:8001"),
        models_request_timeout_seconds=float(os.getenv("MODELS_REQUEST_TIMEOUT_SECONDS", "360")),
        data_dir=os.getenv("DATA_DIR", "/data"),
        cors_origins=_split_csv(os.getenv("CORS_ORIGINS", "*")),
    )
