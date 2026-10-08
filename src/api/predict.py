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
MAX_CACHE_BYTES = 50 * 1024 * 1024


def store_batch_export(token: str, content: bytes) -> None:
    """Store generated CSV diagnostic report bytes keyed by download token.

    The cache is bounded by entry count and by total bytes; the oldest entries are evicted first.
    """
    now = time.time()
    if len(_batch_export_cache) >= MAX_CACHE_ENTRIES:
        expired = [k for k, (t, _) in _batch_export_cache.items() if now - t > CACHE_TTL_SECONDS]
        for k in expired:
            _batch_export_cache.pop(k, None)
        if len(_batch_export_cache) >= MAX_CACHE_ENTRIES:
            oldest = min(_batch_export_cache.keys(), key=lambda k: _batch_export_cache[k][0])
            _batch_export_cache.pop(oldest, None)
    _batch_export_cache[token] = (now, content)

    total_bytes = sum(len(data) for _, data in _batch_export_cache.values())
    while total_bytes > MAX_CACHE_BYTES and _batch_export_cache:
        oldest = min(_batch_export_cache.keys(), key=lambda k: _batch_export_cache[k][0])
        total_bytes -= len(_batch_export_cache.pop(oldest)[1])


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


MAX_UPLOAD_BYTES = 5 * 1024 * 1024
BYTE_LIMIT_MESSAGE = "Batch upload exceeds maximum size of 5 MB"
MAX_REPORTED_ROWS = 10

# (accepted column names, min, max, integer-only) mirroring BehavioralProfileInput
_NUMERIC_RULES: list[tuple[tuple[str, ...], float, float, bool]] = [
    (("age",), 18, 35, True),
    (("daily_screen_time_hours", "daily_screen_time"), 0.0, 24.0, False),
    (("social_media_hours",), 0.0, 24.0, False),
    (("gaming_hours",), 0.0, 24.0, False),
    (("work_study_hours",), 0.0, 24.0, False),
    (("weekend_screen_time", "weekend_screen_time_hours"), 0.0, 24.0, False),
    (("sleep_hours", "sleep_duration_hours"), 1.0, 18.0, False),
    (("notifications_per_day",), 0, 500, True),
    (("app_opens_per_day",), 0, 500, True),
]


def _format_rows(mask: pd.Series) -> str:
    rows = [str(i + 1) for i in range(len(mask)) if mask.iloc[i]]
    shown = ", ".join(rows[:MAX_REPORTED_ROWS])
    if len(rows) > MAX_REPORTED_ROWS:
        shown += f" and {len(rows) - MAX_REPORTED_ROWS} more"
    return shown


def _coerce_and_validate_numeric(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Coerce numeric columns and report offending 1-based data rows per column."""
    df = df.copy()
    problems: list[str] = []
    for names, lo, hi, integer in _NUMERIC_RULES:
        col = next((n for n in names if n in df.columns), None)
        if col is None:
            continue
        num = pd.to_numeric(df[col], errors="coerce").replace([float("inf"), float("-inf")], float("nan"))
        df[col] = num
        bad = num.isna()
        if bad.any():
            problems.append(f"column '{col}' has missing or non-numeric values in row(s) {_format_rows(bad)}")
        out = ~bad & ((num < lo) | (num > hi))
        if integer:
            out = out | (~bad & (num != num.round()))
        if out.any():
            kind = "whole number " if integer else ""
            problems.append(
                f"column '{col}' must be a {kind}value between {lo:g} and {hi:g} in row(s) {_format_rows(out)}"
            )
    return df, problems


def _neutralize_formulas(df: pd.DataFrame) -> pd.DataFrame:
    """Prefix string cells starting with = + - @ with a quote to block spreadsheet formula injection."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].map(
                lambda v: "'" + v if isinstance(v, str) and v.startswith(("=", "+", "-", "@")) else v
            )
    return df


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
def predict_batch(
    file: UploadFile = File(...),
    threshold: Optional[float] = Query(default=None, ge=0.05, le=0.95),
    settings: Settings = Depends(get_settings),
) -> BatchPredictionResponse:
    """Batch scoring endpoint for multi-row CSV participant datasets (AC-3.3).

    Validates mandatory header schemas (AC-3.4), enforces a 10,000 row upper limit (AC-3.5),
    vectorizes tabular inference, and stores an enriched diagnostic report for export (AC-3.6).
    Sync on purpose: FastAPI runs it in the threadpool so CPU-bound pandas/LightGBM work
    does not block the event loop.
    """
    # Enforce byte cap before parsing: never read more than MAX_UPLOAD_BYTES + 1 bytes
    if file.size is not None and file.size > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=BYTE_LIMIT_MESSAGE)
    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=BYTE_LIMIT_MESSAGE)
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
            detail=f"Missing required CSV columns: {', '.join(missing_cols)}",
        )

    # 3. Numeric coercion and physiological bounds (mirrors BehavioralProfileInput)
    df, problems = _coerce_and_validate_numeric(df)
    if problems:
        raise HTTPException(
            status_code=422,
            detail="Invalid values in CSV: " + "; ".join(problems),
        )

    eff_threshold = threshold if threshold is not None else 0.50

    engine = get_inference_engine(
        use_mock=settings.USE_MOCK_MODEL,
        model_path=settings.MODEL_PATH,
        priors_path=settings.PRIORS_PATH,
    )
    df_enriched, response_payload = engine.batch_predict(df, default_threshold=eff_threshold)

    # Cache enriched CSV for export (AC-3.6)
    csv_bytes = _neutralize_formulas(df_enriched).to_csv(index=False).encode("utf-8")
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
