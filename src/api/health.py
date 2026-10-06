"""Health diagnostics, telemetry, and model readiness API endpoints."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status

from src.config import Settings, get_settings
from src.schemas import HealthResponse

router = APIRouter(tags=["Health"])

START_TIME = time.time()


@router.get("/health", response_model=HealthResponse)
@router.get("/api/health", response_model=HealthResponse)
def get_health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Return system health status, artifact availability, and uptime telemetry.

    Raises:
        HTTPException: 503 Service Unavailable if required model or priors are missing
                       and not running in mock mode (AC-1.9).
    """
    uptime = max(0.0, round(time.time() - START_TIME, 2))

    if settings.USE_MOCK_MODEL:
        cohort_cached = Path(settings.COHORT_CACHE_PATH).is_file()
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            version="1.0.0",
            model_family="MockHeuristic",
            fold=1,
            priors_loaded=True,
            cohort_priors_cached=True,
            cohort_cache_loaded=cohort_cached,
            mock_mode=True,
            uptime_seconds=uptime,
        )

    model_exists = Path(settings.MODEL_PATH).is_file()
    priors_exist = Path(settings.PRIORS_PATH).is_file()
    cohort_cached = Path(settings.COHORT_CACHE_PATH).is_file()

    if not model_exists or not priors_exist:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model checkpoint or priors artifact missing",
        )

    return HealthResponse(
        status="healthy",
        model_loaded=model_exists,
        version="1.0.0",
        model_family="LightGBM",
        fold=1,
        priors_loaded=priors_exist,
        cohort_priors_cached=priors_exist,
        cohort_cache_loaded=cohort_cached,
        mock_mode=False,
        uptime_seconds=uptime,
    )
