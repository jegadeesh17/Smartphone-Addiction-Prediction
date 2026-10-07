"""Cohort Analytics Service and Quantile Benchmarking Engine.

Provides high-performance in-memory cohort queries, 2D screen-sleep density
matrices, and personal percentile benchmarking against the 691,369-participant
population cache (ADR-0003, SPEC Journey 2).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Optional
import numpy as np

from src.schemas import (
    BenchmarkOverlayRequest,
    BenchmarkOverlayResponse,
    CohortDistributionResponse,
    CohortFilterParams,
    CohortItem,
    CohortMetricSummary,
    DensityCell,
    ScreenSleepMatrixResponse,
)

ALLOWED_DIMENSIONS = ("age_bracket", "gender", "stress_level", "academic_work_impact")

# 101 population quantiles (0 to 100) from 691,369 records in data/train.csv
APP_OPENS_QUANTILES_100: list[float] = [
    15.0, 16.0, 17.0, 19.0, 21.0, 23.0, 26.0, 28.0, 30.0, 33.0,
    35.0, 36.0, 37.0, 38.0, 41.0, 42.0, 43.0, 47.0, 49.0, 50.0,
    53.0, 54.0, 55.0, 58.0, 61.0, 64.0, 65.0, 66.0, 68.0, 69.0,
    71.0, 73.0, 74.0, 77.0, 78.0, 80.0, 81.0, 84.0, 85.0, 87.0,
    90.0, 91.0, 94.0, 96.0, 96.0, 97.0, 99.0, 100.0, 102.0, 103.0,
    104.0, 106.0, 109.0, 109.0, 110.0, 113.0, 114.0, 116.0, 118.0, 119.0,
    121.0, 122.0, 124.0, 125.0, 127.0, 129.0, 130.0, 131.0, 133.0, 135.0,
    136.0, 138.0, 140.0, 142.0, 144.0, 145.0, 146.0, 148.0, 150.0, 151.0,
    154.0, 155.0, 156.0, 158.0, 159.0, 161.0, 162.0, 163.0, 164.0, 165.0,
    167.0, 168.0, 169.0, 171.0, 173.0, 175.0, 176.0, 177.0, 178.0, 179.0,
    180.0,
]

NOTIFICATIONS_QUANTILES_100: list[float] = [
    20.0, 23.0, 25.0, 29.0, 32.0, 33.0, 37.0, 40.0, 44.0, 47.0,
    49.0, 53.0, 55.0, 58.0, 59.0, 62.0, 65.0, 68.0, 71.0, 76.0,
    78.0, 81.0, 84.0, 88.0, 91.0, 93.0, 96.0, 99.0, 103.0, 105.0,
    108.0, 110.0, 112.0, 113.0, 115.0, 118.0, 119.0, 120.0, 122.0, 124.0,
    127.0, 129.0, 131.0, 133.0, 136.0, 139.0, 142.0, 143.0, 145.0, 149.0,
    150.0, 153.0, 154.0, 155.0, 158.0, 161.0, 163.0, 166.0, 167.0, 170.0,
    173.0, 175.0, 179.0, 182.0, 183.0, 184.0, 185.0, 187.0, 190.0, 193.0,
    194.0, 195.0, 197.0, 198.0, 202.0, 204.0, 206.0, 209.0, 211.0, 213.0,
    213.0, 215.0, 217.0, 220.0, 222.0, 224.0, 225.0, 227.0, 228.0, 230.0,
    232.0, 234.0, 235.0, 237.0, 239.0, 240.0, 242.0, 244.0, 245.0, 248.0,
    250.0,
]


def format_ordinal(n: int) -> str:
    """Format an integer into an English ordinal string (e.g., 1st, 2nd, 3rd, 84th)."""
    n = int(n)
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


class CohortService:
    """Analytical cohort engine querying pre-aggregated population distributions."""

    def __init__(self, cohort_summary_path: str | Path = Path("data/cohort_summary.json")) -> None:
        """Initialize and cache population cohort summary in memory for sub-2ms queries.

        Parameters
        ----------
        cohort_summary_path : str | Path
            Path to serialized data/cohort_summary.json artifact.
        """
        self.cohort_summary_path = Path(cohort_summary_path).resolve()
        if not self.cohort_summary_path.is_file():
            raise FileNotFoundError(
                f"Cohort summary cache not found at {self.cohort_summary_path}. "
                "Ensure scripts/generate_cohort_cache.py has been executed."
            )

        with open(self.cohort_summary_path, "r", encoding="utf-8") as f:
            self._cache: dict[str, Any] = json.load(f)

        self._total_records: int = int(self._cache.get("total_population_records", 691369))
        self._dimensions: dict[str, Any] = self._cache.get("dimensions", {})
        self._pct_points: np.ndarray = np.linspace(0.0, 100.0, 101)

        # Quantile benchmarks for sub-millisecond overlay lookup
        benchmarks = self._cache.get("quantile_benchmarks", {})
        self._screen_mean: float = float(benchmarks.get("population_mean_screen", 7.64))
        self._sleep_mean: float = float(benchmarks.get("population_mean_sleep", 6.80))

        screen_q = benchmarks.get("screen_time", {}).get("quantiles_100")
        if not screen_q:
            screen_q = self._cache.get("population_overall", {}).get("screen_time_quantiles_100", [])
        self._screen_quantiles: np.ndarray = np.array(screen_q, dtype=float)

        sleep_q = benchmarks.get("sleep_hours", {}).get("quantiles_100")
        if not sleep_q:
            sleep_q = self._cache.get("population_overall", {}).get("sleep_hours_quantiles_100", [])
        self._sleep_quantiles: np.ndarray = np.array(sleep_q, dtype=float)

        # App opens and notifications quantiles (empirical 691k population curves)
        opens_q = benchmarks.get("app_opens", {}).get("quantiles_100", APP_OPENS_QUANTILES_100)
        self._app_opens_quantiles: np.ndarray = np.array(opens_q, dtype=float)

        notifs_q = benchmarks.get("notifications", {}).get("quantiles_100", NOTIFICATIONS_QUANTILES_100)
        self._notifications_quantiles: np.ndarray = np.array(notifs_q, dtype=float)

        # Pre-cache 2D joint density matrix response
        self._screen_sleep_matrix: ScreenSleepMatrixResponse = self._build_screen_sleep_matrix()

    def _build_screen_sleep_matrix(self) -> ScreenSleepMatrixResponse:
        """Construct immutable ScreenSleepMatrixResponse from cache payload."""
        matrix_data = self._cache.get("screen_sleep_matrix", {})
        screen_bins = matrix_data.get("screen_bins", ["0-4", "4-6", "6-8", "8-10", "10-12", "12+"])
        sleep_bins = matrix_data.get("sleep_bins", ["<5", "5-6", "6-7", "7-8", "8+"])

        raw_cells = matrix_data.get("grid_cells", [])
        cells = [
            DensityCell(
                screen_bin=str(c.get("screen_bin", "")),
                sleep_bin=str(c.get("sleep_bin", "")),
                count=int(c.get("sample_count", c.get("count", 0))),
                density_pct=float(c.get("cell_density_pct", c.get("density_pct", 0.0))),
                addiction_rate_pct=float(c.get("addiction_rate_pct", 0.0)),
            )
            for c in raw_cells
        ]

        matrix_pct = matrix_data.get("density_matrix_pct", matrix_data.get("density_matrix", []))
        norm_matrix = matrix_data.get("density_matrix", None)
        addiction_matrix = matrix_data.get("addiction_rate_matrix_pct", None)

        return ScreenSleepMatrixResponse(
            screen_bins=screen_bins,
            sleep_bins=sleep_bins,
            matrix=matrix_pct,
            cells=cells,
            density_matrix=norm_matrix,
            addiction_rate_matrix=addiction_matrix,
        )

    def _extract_metric_summary(self, raw_c: dict[str, Any]) -> CohortMetricSummary:
        """Extract and coerce raw cached cohort metrics into typed CohortMetricSummary."""
        count = int(raw_c.get("count", raw_c.get("sample_count", 0)))
        addiction_prevalence = float(raw_c.get("addiction_prevalence", 0.0))

        screen_dict = raw_c.get("screen_time", {})
        screen_time_mean = float(screen_dict.get("mean", raw_c.get("mean_screen_time", 0.0)))
        screen_time_std = float(screen_dict.get("std", 0.0))
        screen_time_p50 = float(screen_dict.get("percentiles", {}).get("p50", 0.0))

        sleep_dict = raw_c.get("sleep_hours", {})
        sleep_hours_mean = float(sleep_dict.get("mean", raw_c.get("mean_sleep_hours", 0.0)))
        sleep_hours_std = float(sleep_dict.get("std", 0.0))
        sleep_hours_p50 = float(sleep_dict.get("percentiles", {}).get("p50", 0.0))

        social_media_mean = float(raw_c.get("social_media", {}).get("mean", 0.0))
        gaming_mean = float(raw_c.get("gaming", {}).get("mean", 0.0))
        work_study_mean = float(raw_c.get("work_study", {}).get("mean", 0.0))

        app_opens_dict = raw_c.get("app_opens", {})
        app_opens_mean = float(app_opens_dict.get("mean", raw_c.get("mean_app_opens", 0.0)))

        notifications_dict = raw_c.get("notifications", {})
        notifications_mean = float(notifications_dict.get("mean", 0.0))

        return CohortMetricSummary(
            count=count,
            addiction_prevalence=addiction_prevalence,
            screen_time_mean=screen_time_mean,
            screen_time_std=screen_time_std,
            sleep_hours_mean=sleep_hours_mean,
            sleep_hours_std=sleep_hours_std,
            social_media_mean=social_media_mean,
            gaming_mean=gaming_mean,
            work_study_mean=work_study_mean,
            app_opens_mean=app_opens_mean,
            notifications_mean=notifications_mean,
            screen_time_p50=screen_time_p50,
            sleep_hours_p50=sleep_hours_p50,
        )

    def get_cohorts(self, params: CohortFilterParams | None = None, **kwargs: Any) -> CohortDistributionResponse:
        """Retrieve aggregated demographic cohort breakdown and percentiles.

        Parameters
        ----------
        params : CohortFilterParams, optional
            Dimension and demographic filter parameters.

        Returns
        -------
        CohortDistributionResponse
            Aggregated cohort distribution conforming to SPEC AC-2.1 and AC-2.2.
        """
        if params is None:
            params = CohortFilterParams(**kwargs)

        # Explicit dimension validation (AC-2.5)
        if params.dimension not in ALLOWED_DIMENSIONS:
            raise ValueError(
                f"Invalid dimension '{params.dimension}'. Permissible dimensions are: {list(ALLOWED_DIMENSIONS)}"
            )

        dim_entry = self._dimensions.get(params.dimension, {})

        # Normalize filter values
        filter_stress = params.filter_stress
        if filter_stress in ("all", "All", ""):
            filter_stress = None
        elif filter_stress is not None:
            filter_stress = filter_stress.capitalize()

        filter_gender = params.filter_gender
        if filter_gender in ("all", "All", ""):
            filter_gender = None
        elif filter_gender is not None:
            filter_gender = filter_gender.capitalize()

        raw_cohorts: list[dict[str, Any]] = []

        # Check conditioned sub-cohort slices (AC-2.2)
        if filter_stress and "by_stress" in dim_entry and filter_stress in dim_entry["by_stress"]:
            raw_cohorts = dim_entry["by_stress"][filter_stress]
        elif filter_gender and "by_gender" in dim_entry and filter_gender in dim_entry["by_gender"]:
            raw_cohorts = dim_entry["by_gender"][filter_gender]
        else:
            raw_cohorts = dim_entry.get("all", [])

        cohort_items: list[CohortItem] = []
        for raw_c in raw_cohorts:
            label = str(raw_c.get("cohort_name", raw_c.get("label", "")))
            cohort_id = f"{params.dimension}_{label.lower().replace(' ', '_').replace('-', '_')}"
            metrics = self._extract_metric_summary(raw_c)
            cohort_items.append(
                CohortItem(
                    cohort_id=cohort_id,
                    label=label,
                    metrics=metrics,
                )
            )

        total_records = sum(item.metrics.count for item in cohort_items)

        return CohortDistributionResponse(
            dimension=params.dimension,
            total_records=total_records,
            cohorts=cohort_items,
        )

    def get_screen_sleep_matrix(self) -> ScreenSleepMatrixResponse:
        """Return 6x5 2D joint density and addiction prevalence grid (AC-2.3)."""
        return self._screen_sleep_matrix

    def compute_benchmark_overlay(self, profile: BenchmarkOverlayRequest) -> BenchmarkOverlayResponse:
        """Compute user quantile percentiles against 691k population curves (AC-2.4).

        Parameters
        ----------
        profile : BenchmarkOverlayRequest
            User digital habit time budget and interaction counts.

        Returns
        -------
        BenchmarkOverlayResponse
            Percentile ranks, clinical narrative labels, and population means.
        """
        # Screen time percentile lookup
        screen_raw = float(np.interp(profile.daily_screen_time_hours, self._screen_quantiles, self._pct_points))
        screen_pct = round(float(np.clip(screen_raw, 0.0, 100.0)), 2)

        # Sleep duration percentile lookup
        sleep_raw = float(np.interp(profile.sleep_hours, self._sleep_quantiles, self._pct_points))
        sleep_pct = round(float(np.clip(sleep_raw, 0.0, 100.0)), 2)

        # Optional app opens percentile lookup
        opens_pct: Optional[float] = None
        if profile.app_opens_per_day is not None:
            opens_raw = float(np.interp(profile.app_opens_per_day, self._app_opens_quantiles, self._pct_points))
            opens_pct = round(float(np.clip(opens_raw, 0.0, 100.0)), 2)

        # Optional notifications percentile lookup
        notifs_pct: Optional[float] = None
        if profile.notifications_per_day is not None:
            notifs_raw = float(np.interp(profile.notifications_per_day, self._notifications_quantiles, self._pct_points))
            notifs_pct = round(float(np.clip(notifs_raw, 0.0, 100.0)), 2)

        # Narrative clinical labels
        screen_label = f"{format_ordinal(round(screen_pct))} percentile in screen time"
        sleep_label = f"{format_ordinal(round(sleep_pct))} percentile in sleep duration"

        return BenchmarkOverlayResponse(
            screen_time_percentile=screen_pct,
            sleep_hours_percentile=sleep_pct,
            screen_time_label=screen_label,
            sleep_hours_label=sleep_label,
            app_opens_percentile=opens_pct,
            notifications_percentile=notifs_pct,
            population_mean_screen=self._screen_mean,
            population_mean_sleep=self._sleep_mean,
        )


_cohort_service_instance: Optional[CohortService] = None


def get_cohort_service(cohort_summary_path: str | Path | None = None) -> CohortService:
    """Provide singleton instance of CohortService initialized with cohort cache."""
    global _cohort_service_instance
    if _cohort_service_instance is None or (
        cohort_summary_path is not None
        and Path(cohort_summary_path).resolve() != _cohort_service_instance.cohort_summary_path
    ):
        path = cohort_summary_path or Path("data/cohort_summary.json")
        _cohort_service_instance = CohortService(cohort_summary_path=path)
    return _cohort_service_instance


def reset_cohort_service() -> None:
    """Reset singleton instance (primarily for testing and fixtures)."""
    global _cohort_service_instance
    _cohort_service_instance = None
