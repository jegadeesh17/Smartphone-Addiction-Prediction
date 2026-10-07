"""Application configuration management using Pydantic v2 Settings.

Loads environment variables, defines default paths, application ports,
CORS permissions, and model toggles (M1-TASK-05).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Centralized service configuration settings loaded from environment or defaults."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    APP_HOST: str = "127.0.0.1"
    APP_PORT: int = 8000
    APP_DEBUG: bool = False
    MODEL_PATH: Path = ROOT_DIR / "models" / "lgb_fold_1.joblib"
    PRIORS_PATH: Path = ROOT_DIR / "data" / "priors.json"
    COHORT_CACHE_PATH: Path = ROOT_DIR / "data" / "cohort_summary.json"
    USE_MOCK_MODEL: bool = False
    CORS_ORIGINS: list[str] = ["*"]

    @field_validator("APP_HOST", mode="before")
    @classmethod
    def _parse_app_host(cls, v: Any) -> str:
        if v is None or (isinstance(v, str) and not v.strip()):
            return "127.0.0.1"
        return str(v).strip()

    @field_validator("APP_PORT", mode="before")
    @classmethod
    def _parse_app_port(cls, v: Any) -> int:
        if v is None or v == "":
            return 8000
        return int(v)

    @field_validator("APP_DEBUG", "USE_MOCK_MODEL", mode="before")
    @classmethod
    def _parse_bool(cls, v: Any) -> bool:
        if v is None or v == "":
            return False
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "t", "yes", "y")
        return bool(v)

    @field_validator("MODEL_PATH", mode="before")
    @classmethod
    def _parse_model_path(cls, v: Any) -> Path:
        if v is None or (isinstance(v, str) and not v.strip()):
            return ROOT_DIR / "models" / "lgb_fold_1.joblib"
        p = Path(v)
        return p if p.is_absolute() else ROOT_DIR / p

    @field_validator("PRIORS_PATH", mode="before")
    @classmethod
    def _parse_priors_path(cls, v: Any) -> Path:
        if v is None or (isinstance(v, str) and not v.strip()):
            return ROOT_DIR / "data" / "priors.json"
        p = Path(v)
        return p if p.is_absolute() else ROOT_DIR / p

    @field_validator("COHORT_CACHE_PATH", mode="before")
    @classmethod
    def _parse_cohort_cache_path(cls, v: Any) -> Path:
        if v is None or (isinstance(v, str) and not v.strip()):
            return ROOT_DIR / "data" / "cohort_summary.json"
        p = Path(v)
        return p if p.is_absolute() else ROOT_DIR / p

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def _parse_cors_origins(cls, v: Any) -> list[str]:
        if v is None or v == "":
            return ["*"]
        if isinstance(v, str):
            v_str = v.strip()
            if not v_str or v_str == "*":
                return ["*"]
            if v_str.startswith("[") and v_str.endswith("]"):
                try:
                    return json.loads(v_str)
                except Exception:
                    pass
            return [x.strip() for x in v_str.split(",") if x.strip()]
        return list(v)


@lru_cache()
def get_settings() -> Settings:
    """Return cached singleton instance of Settings."""
    return Settings()
