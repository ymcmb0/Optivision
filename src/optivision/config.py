from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, ValidationError, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict

# ──────────────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parents[2]      # …/optivision/
ENV_FILE    = PROJECT_DIR / ".env"

# ──────────────────────────────────────────────────────────────────────────────
class Settings(BaseSettings):
    """
    All application configuration lives here.  Load order:

    1. Environment variables      (highest priority)
    2. `.env` in repo root
    3. Field defaults (below)
    """

    # Pydantic-Settings meta
    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_prefix="",               # naked env vars allowed
        case_sensitive=False,
        extra="forbid",
        validate_default=True,
    )

    # ─── General ────────────────────────────────────────────────────────────
    debug: bool = Field(False, alias="DEBUG")
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    # ─── Paths / Models ─────────────────────────────────────────────────────
    yolo_weights: Path = PROJECT_DIR / "weights/yolov8s.pt"

    # ─── External services ─────────────────────────────────────────────────
    # Accept either OCR_API_KEY **or** OCR_SPACE_API_KEY
    ocr_space_api_key: str = Field(
        ...,
        validation_alias=AliasChoices("OCR_API_KEY", "OCR_SPACE_API_KEY"),
        description="Secret key issued by OCR.Space",
    )

    open_meteo_base_url: str = "https://api.open-meteo.com/v1/forecast"


@lru_cache
def get_settings() -> Settings:            # pragma: no cover
    """Singleton accessor used by FastAPI DI."""
    try:
        return Settings()                  # type: ignore[call-arg]
    except ValidationError as exc:
        # Fail fast & clearly on startup
        import sys, textwrap, json

        sys.stderr.write(
            "\n❌  CONFIG VALIDATION ERROR\n"
            + textwrap.indent(exc.json(), "   ")
            + "\n\nHint: populate `.env` or export OCR_API_KEY.\n"
        )
        raise
