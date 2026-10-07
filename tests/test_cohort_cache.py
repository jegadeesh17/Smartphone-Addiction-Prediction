"""Unit and integration tests for population cohort cache generation and validation.

Verifies SPEC AC-2.1, AC-2.3, REG-3, and ADR-0003 for M2-TASK-01.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pytest

from scripts.generate_cohort_cache import (
    validate_cohort_summary,
    compute_cohort_metrics,
    get_percentiles,
    compute_screen_sleep_grid,
)
import pandas as pd
import numpy as np


@pytest.fixture(scope="module")
def cohort_cache_path(root_dir: Path) -> Path:
    """Return path to data/cohort_summary.json."""
    return root_dir / "data" / "cohort_summary.json"


@pytest.fixture(scope="module")
def cohort_cache_data(cohort_cache_path: Path) -> dict[str, Any]:
    """Load and return parsed cohort summary cache."""
    assert cohort_cache_path.exists(), f"Cohort cache not found at {cohort_cache_path}"
    with open(cohort_cache_path, "r", encoding="utf-8") as f:
        return json.load(f)


class TestCohortCacheIntegrity:
    """Validate schema integrity and contents of data/cohort_summary.json (M2-TASK-01, ADR-0003)."""

    def test_cohort_summary_file_exists_and_validates(self, cohort_cache_path: Path) -> None:
        """Verify data/cohort_summary.json exists and passes self-validation routine."""
        assert cohort_cache_path.is_file()
        assert cohort_cache_path.stat().st_size > 1000  # Non-empty file
        is_valid = validate_cohort_summary(cohort_cache_path)
        assert is_valid is True

    def test_root_structure_and_population_count(self, cohort_cache_data: dict[str, Any]) -> None:
        """Verify root structure keys and total population record count of 691,369."""
        expected_keys = {
            "version",
            "total_population_records",
            "population_overall",
            "dimensions",
            "cohorts_by_dimension",
            "screen_sleep_matrix",
            "quantile_benchmarks",
        }
        assert expected_keys.issubset(set(cohort_cache_data.keys()))
        assert cohort_cache_data["total_population_records"] == 691369
        assert cohort_cache_data["population_overall"]["sample_count"] == 691369

    def test_demographic_dimensions_present_and_non_empty(self, cohort_cache_data: dict[str, Any]) -> None:
        """Verify demographic dimensions (age, gender, stress, academic impact) meet AC-2.1."""
        dimensions = cohort_cache_data["dimensions"]
        expected_dims = ["age_bracket", "gender", "stress_level", "academic_work_impact"]

        for dim in expected_dims:
            assert dim in dimensions, f"Missing dimension: {dim}"
            cohorts = dimensions[dim]["all"]
            assert len(cohorts) > 0, f"Dimension {dim} has empty cohorts"

            for c in cohorts:
                assert "cohort_name" in c and len(c["cohort_name"]) > 0
                assert c["sample_count"] > 0
                assert 0.0 <= c["addiction_prevalence"] <= 1.0
                assert c["mean_screen_time"] > 0.0
                assert c["mean_sleep_hours"] > 0.0
                assert c["mean_app_opens"] > 0.0

    def test_screen_sleep_density_grid_conforms_to_spec(self, cohort_cache_data: dict[str, Any]) -> None:
        """Verify 2D density and prevalence matrix conforms to AC-2.3."""
        grid = cohort_cache_data["screen_sleep_matrix"]
        assert len(grid["screen_bins"]) == 6
        assert len(grid["sleep_bins"]) == 5
        assert len(grid["grid_cells"]) == 30

        assert len(grid["density_matrix"]) == 6
        assert all(len(row) == 5 for row in grid["density_matrix"])
        assert len(grid["addiction_rate_matrix"]) == 6
        assert all(len(row) == 5 for row in grid["addiction_rate_matrix"])

        total_density = sum(cell["cell_density_pct"] for cell in grid["grid_cells"])
        assert 99.5 <= total_density <= 100.5, f"Density does not sum to 100%: {total_density}"

        for cell in grid["grid_cells"]:
            assert cell["sample_count"] > 0
            assert 0.0 <= cell["addiction_rate_pct"] <= 100.0

    def test_quantile_benchmarks_validity(self, cohort_cache_data: dict[str, Any]) -> None:
        """Verify quantile benchmarks for sub-millisecond overlays (AC-2.4)."""
        benchmarks = cohort_cache_data["quantile_benchmarks"]
        assert 7.55 <= benchmarks["population_mean_screen"] <= 7.75
        assert 6.70 <= benchmarks["population_mean_sleep"] <= 6.90

        screen_q = benchmarks["screen_time"]["quantiles_100"]
        sleep_q = benchmarks["sleep_hours"]["quantiles_100"]
        assert len(screen_q) == 101
        assert len(sleep_q) == 101

        # Check non-decreasing order
        for i in range(100):
            assert screen_q[i] <= screen_q[i + 1]
            assert sleep_q[i] <= sleep_q[i + 1]


class TestCohortValidationEdgeCases:
    """Invalid-input tests and defensive failure cases for cohort cache validation."""

    def test_missing_root_key_raises_assertion_error(self, cohort_cache_data: dict[str, Any]) -> None:
        """Verify validate_cohort_summary raises AssertionError if a mandatory key is missing."""
        corrupted = dict(cohort_cache_data)
        del corrupted["screen_sleep_matrix"]
        with pytest.raises(AssertionError) as exc_info:
            validate_cohort_summary(corrupted)
        assert "Missing root key 'screen_sleep_matrix'" in str(exc_info.value)

    def test_wrong_record_count_raises_assertion_error(self, cohort_cache_data: dict[str, Any]) -> None:
        """Verify validate_cohort_summary fails when record count doesn't match 691,369."""
        corrupted = dict(cohort_cache_data)
        corrupted["total_population_records"] = 1000
        with pytest.raises(AssertionError) as exc_info:
            validate_cohort_summary(corrupted)
        assert "Expected 691,369 records" in str(exc_info.value)

    def test_nonexistent_path_raises_file_not_found(self, tmp_path: Path) -> None:
        """Verify validate_cohort_summary raises FileNotFoundError for non-existent path."""
        fake_path = tmp_path / "nonexistent_cohort.json"
        with pytest.raises(FileNotFoundError):
            validate_cohort_summary(fake_path)

    def test_empty_dataframe_metrics_fallback(self) -> None:
        """Verify compute_cohort_metrics handles empty slice safely with 0.0 fallbacks."""
        empty_df = pd.DataFrame(columns=[
            "addicted_label", "daily_screen_time_hours", "sleep_hours",
            "social_media_hours", "gaming_hours", "work_study_hours",
            "app_opens_per_day", "notifications_per_day",
        ])
        metrics = compute_cohort_metrics(empty_df, cohort_name="EmptyCohort")
        assert metrics["cohort_name"] == "EmptyCohort"
        assert metrics["sample_count"] == 0
        assert metrics["addiction_prevalence"] == 0.0
        assert metrics["mean_screen_time"] == 0.0

    def test_get_percentiles_empty_fallback(self) -> None:
        """Verify get_percentiles returns zeros on empty input array."""
        empty_arr = np.array([])
        pcts = get_percentiles(empty_arr, [10, 50, 90])
        assert pcts == {"p10": 0.0, "p50": 0.0, "p90": 0.0}
