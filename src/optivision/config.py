# src/optivision/config.py
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, AliasChoices  # ← AliasChoices is the trick

PROJECT_DIR = Path(__file__).resolve().parents[2]  # …/optivision


class Settings(BaseSettings):
    # ─── General ────────────────────────────────────────────────────────────
    debug: bool = Field(False, env="DEBUG")
    log_level: str = Field("INFO", env="LOG_LEVEL")

    # ─── Paths / Models ─────────────────────────────────────────────────────
    yolo_weights: Path = PROJECT_DIR / "weights/yolov8s.pt"

    # ─── Third-party keys ───────────────────────────────────────────────────
    ocr_space_api_key: str = Field(
        ...,
        validation_alias=AliasChoices(
            "OCR_API_KEY",
            "OCR_SPACE_API_KEY",
            "ocr_api_key",
        ),
    )
    open_meteo_base_url: str = "https://api.open-meteo.com/v1/forecast"

    # ─── pydantic-settings meta ─────────────────────────────────────────────
    model_config = SettingsConfigDict(
        env_file=PROJECT_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # ignore unrelated env vars
    )


@lru_cache
def get_settings() -> Settings:
    """Singleton accessor compatible with FastAPI dependency injection."""
    return Settings()
