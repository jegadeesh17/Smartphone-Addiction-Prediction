"""Cohort Analytics API router and population distribution endpoints.

Exposes demographic cohort slicing, 2D screen-sleep joint density matrix,
and personal quantile benchmark calculation (Milestone 2, Journey 2, AC-2.1 to AC-2.5).
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.cohort_service import ALLOWED_DIMENSIONS, get_cohort_service
from src.config import Settings, get_settings
from src.schemas import (
    BenchmarkOverlayRequest,
    BenchmarkOverlayResponse,
    CohortDistributionResponse,
    CohortFilterParams,
    ScreenSleepMatrixResponse,
)

router = APIRouter(prefix="/api/analytics", tags=["Cohort Analytics"])


@router.get(
    "/cohorts",
    response_model=CohortDistributionResponse,
    summary="Get population cohort distributions and metric breakdowns",
)
def get_cohorts(
    dimension: str = Query(
        default="age_bracket",
        description="Primary demographic segmentation dimension ('age_bracket', 'gender', 'stress_level', 'academic_work_impact')",
    ),
    filter_gender: Optional[str] = Query(
        default=None,
        description="Optional filter conditioning on participant gender ('Female', 'Male', 'Other')",
    ),
    filter_stress: Optional[str] = Query(
        default=None,
        description="Optional filter conditioning on chronic stress level ('Low', 'Medium', 'High')",
    ),
    settings: Settings = Depends(get_settings),
) -> CohortDistributionResponse:
    """Retrieve pre-aggregated demographic cohort breakdown and percentiles (AC-2.1, AC-2.2).

    Validates dimension parameters with explicit HTTP 422 error on invalid inputs (AC-2.5).
    """
    if dimension not in ALLOWED_DIMENSIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid dimension '{dimension}'. Permissible dimensions are: {list(ALLOWED_DIMENSIONS)}",
        )

    try:
        params = CohortFilterParams(
            dimension=dimension,  # type: ignore[arg-type]
            filter_gender=filter_gender,
            filter_stress=filter_stress,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    service = get_cohort_service(cohort_summary_path=settings.COHORT_CACHE_PATH)
    return service.get_cohorts(params)


@router.get(
    "/distributions/screen-sleep-matrix",
    response_model=ScreenSleepMatrixResponse,
    summary="Get 2D screen time vs sleep duration joint density grid",
)
def get_screen_sleep_matrix(
    settings: Settings = Depends(get_settings),
) -> ScreenSleepMatrixResponse:
    """Retrieve 2D joint density and addiction prevalence grid (AC-2.3)."""
    service = get_cohort_service(cohort_summary_path=settings.COHORT_CACHE_PATH)
    return service.get_screen_sleep_matrix()


@router.post(
    "/distributions/benchmark-overlay",
    response_model=BenchmarkOverlayResponse,
    summary="Compute personal quantile benchmark overlay",
)
def compute_benchmark_overlay(
    body: BenchmarkOverlayRequest,
    settings: Settings = Depends(get_settings),
) -> BenchmarkOverlayResponse:
    """Compute quantile percentiles against 691k population curves (AC-2.4)."""
    service = get_cohort_service(cohort_summary_path=settings.COHORT_CACHE_PATH)
    return service.compute_benchmark_overlay(body)
