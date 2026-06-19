import os
from dataclasses import dataclass

from dotenv import load_dotenv


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str | None
    openrouter_base_url: str
    bias_openrouter_model: str
    factuality_openrouter_model: str
    request_timeout_seconds: float
    request_retries: int
    max_upload_mb: int
    max_llm_image_width: int
    cors_origins: list[str]


def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        bias_openrouter_model=os.getenv("BIAS_OPENROUTER_MODEL", os.getenv("BIAS_MODEL", "openai/gpt-5.5")),
        factuality_openrouter_model=os.getenv(
            "FACTUALITY_OPENROUTER_MODEL",
            os.getenv("FACTUALITY_MODEL", "openai/gpt-5.5"),
        ),
        request_timeout_seconds=float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "180")),
        request_retries=int(os.getenv("OPENROUTER_RETRIES", "2")),
        max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "20")),
        max_llm_image_width=int(os.getenv("MAX_LLM_IMAGE_WIDTH", "1600")),
        cors_origins=_split_csv(os.getenv("CORS_ORIGINS", "*")),
    )
