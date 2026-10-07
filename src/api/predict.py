"""Inference API endpoints for live tabular risk evaluation, batch scoring, and export diagnostics."""

from __future__ import annotations

import io
import time
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, Response, UploadFile
import pandas as pd

from src.config import Settings, get_settings
from src.inference import get_inference_engine
from src.schemas import (
    BatchPredictionResponse,
    BehavioralProfileInput,
    PredictionResponse,
)

router = APIRouter(tags=["Inference"])

# In-memory export cache: download_token -> (timestamp, csv_bytes)
_batch_export_cache: dict[str, tuple[float, bytes]] = {}
CACHE_TTL_SECONDS = 3600
MAX_CACHE_ENTRIES = 500


def store_batch_export(token: str, content: bytes) -> None:
    """Store generated CSV diagnostic report bytes keyed by download token."""
    now = time.time()
    if len(_batch_export_cache) >= MAX_CACHE_ENTRIES:
        expired = [k for k, (t, _) in _batch_export_cache.items() if now - t > CACHE_TTL_SECONDS]
        for k in expired:
            _batch_export_cache.pop(k, None)
        if len(_batch_export_cache) >= MAX_CACHE_ENTRIES:
            oldest = min(_batch_export_cache.keys(), key=lambda k: _batch_export_cache[k][0])
            _batch_export_cache.pop(oldest, None)
    _batch_export_cache[token] = (now, content)


def get_batch_export(token: str) -> Optional[bytes]:
    """Retrieve cached CSV diagnostic report bytes by download token or return None if expired/missing."""
    entry = _batch_export_cache.get(token)
    if entry is None:
        return None
    created_at, content = entry
    if time.time() - created_at > CACHE_TTL_SECONDS:
        _batch_export_cache.pop(token, None)
        return None
    return content


def clear_batch_export_cache() -> None:
    """Clear export cache (useful for test isolation)."""
    _batch_export_cache.clear()


@router.post("/api/predict", response_model=PredictionResponse)
@router.post("/predict", response_model=PredictionResponse)
def predict(
    profile: BehavioralProfileInput,
    settings: Settings = Depends(get_settings),
) -> PredictionResponse:
    """Evaluate single-row behavioral profile with sub-20ms latency SLA.

    Constructs 74-feature vector using cached population priors and returns
    PredictionResponse populated with probability, classification, 3-tier
    clinical spectrum status_label, behavioral ratios, and recommended interventions.
    """
    engine = get_inference_engine(
        use_mock=settings.USE_MOCK_MODEL,
        model_path=settings.MODEL_PATH,
        priors_path=settings.PRIORS_PATH,
    )
    return engine.predict(profile)


@router.post("/api/predict/batch", response_model=BatchPredictionResponse)
@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    file: UploadFile = File(...),
    threshold: Optional[float] = Query(default=None, ge=0.05, le=0.95),
    settings: Settings = Depends(get_settings),
) -> BatchPredictionResponse:
    """Batch scoring endpoint for multi-row CSV participant datasets (AC-3.3).

    Validates mandatory header schemas (AC-3.4), enforces a 10,000 row upper limit (AC-3.5),
    vectorizes tabular inference, and stores an enriched diagnostic report for export (AC-3.6).
    """
    content = await file.read()
    if not content or len(content.strip()) == 0:
        raise HTTPException(status_code=422, detail="Empty CSV file uploaded.")

    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse CSV file: {str(e)}")

    if len(df) == 0:
        raise HTTPException(status_code=422, detail="CSV file contains zero data rows.")

    # 1. Enforce row limit (AC-3.5)
    if len(df) > 10_000:
        raise HTTPException(
            status_code=413,
            detail="Batch upload exceeds maximum limit of 10,000 rows",
        )

    # 2. Mandatory header validation (AC-3.4)
    missing_cols: list[str] = []
    if "age" not in df.columns:
        missing_cols.append("age")
    if "gender" not in df.columns:
        missing_cols.append("gender")
    if "stress_level" not in df.columns:
        missing_cols.append("stress_level")
    if "academic_work_impact" not in df.columns:
        missing_cols.append("academic_work_impact")
    if "daily_screen_time_hours" not in df.columns and "daily_screen_time" not in df.columns:
        missing_cols.append("daily_screen_time_hours")
    if "social_media_hours" not in df.columns:
        missing_cols.append("social_media_hours")
    if "gaming_hours" not in df.columns:
        missing_cols.append("gaming_hours")
    if "work_study_hours" not in df.columns:
        missing_cols.append("work_study_hours")
    if "weekend_screen_time" not in df.columns and "weekend_screen_time_hours" not in df.columns:
        missing_cols.append("weekend_screen_time")
    if "sleep_hours" not in df.columns and "sleep_duration_hours" not in df.columns:
        missing_cols.append("sleep_hours")
    if "notifications_per_day" not in df.columns:
        missing_cols.append("notifications_per_day")
    if "app_opens_per_day" not in df.columns:
        missing_cols.append("app_opens_per_day")

    if missing_cols:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required CSV columns: {missing_cols}",
        )

    eff_threshold = threshold if threshold is not None else 0.50

    engine = get_inference_engine(
        use_mock=settings.USE_MOCK_MODEL,
        model_path=settings.MODEL_PATH,
        priors_path=settings.PRIORS_PATH,
    )
    df_enriched, response_payload = engine.batch_predict(df, default_threshold=eff_threshold)

    # Cache enriched CSV for export (AC-3.6)
    csv_bytes = df_enriched.to_csv(index=False).encode("utf-8")
    store_batch_export(response_payload.download_token, csv_bytes)

    return response_payload


@router.get("/api/predict/batch/export")
@router.get("/predict/batch/export")
def export_batch(token: str = Query(..., description="Batch download token")) -> Response:
    """Stream full enriched diagnostic CSV report matching AC-3.6.

    Returns HTTP 200 with Content-Disposition attachment 'diagnostic_report.csv'
    containing original columns plus predicted_probability, classification,
    screen_to_sleep_ratio, and primary_intervention.
    """
    csv_bytes = get_batch_export(token)
    if csv_bytes is None:
        raise HTTPException(
            status_code=404,
            detail="Export token not found or expired",
        )

    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={
            "Content-Disposition": 'attachment; filename="diagnostic_report.csv"'
        },
    )
