"""
backend/core/config.py
Application configuration using Pydantic BaseSettings.
All values can be overridden by environment variables or a .env file.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        str_strip_whitespace=True,
    )

    # ── Application ──────────────────────────────────────────────────────────
    app_name: str = "Horizon Exoplanet Platform"
    app_version: str = "1.0.0"
    app_env: str = "development"
    debug: bool = True
    secret_key: str = "change-me-in-production-32-chars-min"

    # ── Database ─────────────────────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://horizon:horizon@localhost:5432/horizondb"
    use_sqlite: bool = False
    sqlite_url: str = "sqlite+aiosqlite:///./horizon.db"

    @property
    def effective_database_url(self) -> str:
        return self.sqlite_url if self.use_sqlite else self.database_url

    # ── Redis ────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    use_redis: bool = True
    cache_ttl_seconds: int = 3600  # 1 hour default TTL

    # ── MAST / Lightkurve ────────────────────────────────────────────────────
    mast_timeout: int = 120
    mast_cache_dir: str = "./backend/datasets/cache"

    # ── ML Models ────────────────────────────────────────────────────────────
    model_dir: str = "./backend/ml_models"
    cnn_model_path: str = "./backend/ml_models/transit_classifier.h5"
    input_length: int = 201  # Fixed length for folded light curve input to CNN

    # ── Storage ──────────────────────────────────────────────────────────────
    datasets_dir: str = "./backend/datasets"
    reports_dir: str = "./backend/reports/output"
    max_upload_size_mb: int = 500

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    # ── CORS ─────────────────────────────────────────────────────────────────
    allowed_origins: str = "http://localhost:3000,http://localhost:5173,http://localhost:80"

    @property
    def allowed_origins_list(self) -> List[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]

    # ── Transit Detection ─────────────────────────────────────────────────────
    tls_min_period: float = 0.5    # days
    tls_max_period: float = 27.0   # days (TESS sector ~27 days)
    bls_min_period: float = 0.5
    bls_max_period: float = 27.0
    snr_threshold: float = 7.0     # minimum SNR to flag as candidate

    # ── Validation thresholds ─────────────────────────────────────────────────
    confidence_threshold: float = 0.5  # ML confidence above which → CANDIDATE


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()


# Convenience singleton
settings = get_settings()
