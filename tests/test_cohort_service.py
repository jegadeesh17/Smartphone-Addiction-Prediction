"""Unit and latency benchmark tests for CohortService (Milestone 2, Journey 2)."""

import time
from pathlib import Path
import pytest
from pydantic import ValidationError

from src.cohort_service import (
    CohortService,
    format_ordinal,
    get_cohort_service,
    reset_cohort_service,
)
from src.schemas import (
    BenchmarkOverlayRequest,
    BenchmarkOverlayResponse,
    CohortDistributionResponse,
    CohortFilterParams,
    CohortItem,
    DensityCell,
    ScreenSleepMatrixResponse,
)


@pytest.fixture(autouse=True)
def reset_service_fixture() -> None:
    """Ensure clean service singleton state for each test."""
    reset_cohort_service()
    yield
    reset_cohort_service()


@pytest.fixture
def cohort_service() -> CohortService:
    """Fixture returning an initialized CohortService instance."""
    return get_cohort_service()


class TestCohortServiceInitialization:
    """Verification of cache loading and singleton management."""

    def test_singleton_accessor(self, cohort_service: CohortService) -> None:
        """Verify get_cohort_service returns the same initialized instance."""
        service_2 = get_cohort_service()
        assert cohort_service is service_2
        assert cohort_service._total_records == 691369

    def test_missing_cache_file_raises_error(self, tmp_path: Path) -> None:
        """Verify non-existent summary path raises FileNotFoundError."""
        missing = tmp_path / "non_existent_cohorts.json"
        with pytest.raises(FileNotFoundError, match="Cohort summary cache not found"):
            CohortService(cohort_summary_path=missing)

    def test_ordinal_formatting_rules(self) -> None:
        """Verify English ordinal suffix rules (1st, 2nd, 3rd, 11th, 22nd, 84th)."""
        assert format_ordinal(1) == "1st"
        assert format_ordinal(2) == "2nd"
        assert format_ordinal(3) == "3rd"
        assert format_ordinal(4) == "4th"
        assert format_ordinal(11) == "11th"
        assert format_ordinal(12) == "12th"
        assert format_ordinal(13) == "13th"
        assert format_ordinal(21) == "21st"
        assert format_ordinal(22) == "22nd"
        assert format_ordinal(23) == "23rd"
        assert format_ordinal(50) == "50th"
        assert format_ordinal(84) == "84th"
        assert format_ordinal(100) == "100th"


class TestCohortDistributionQueries:
    """Verification of get_cohorts across all demographic dimensions."""

    @pytest.mark.parametrize(
        ("dimension", "expected_labels"),
        [
            ("age_bracket", ["18-21", "22-25", "26-30", "31-35"]),
            ("gender", ["Female", "Male", "Other"]),
            ("stress_level", ["Low", "Medium", "High"]),
            ("academic_work_impact", ["Yes", "No"]),
        ],
    )
    def test_all_four_primary_dimensions(
        self, cohort_service: CohortService, dimension: str, expected_labels: list[str]
    ) -> None:
        """Verify get_cohorts returns non-empty cohorts with valid labels for all 4 dimensions (AC-2.1)."""
        params = CohortFilterParams(dimension=dimension)  # type: ignore[arg-type]
        response = cohort_service.get_cohorts(params)

        assert isinstance(response, CohortDistributionResponse)
        assert response.dimension == dimension
        assert len(response.cohorts) == len(expected_labels)
        assert response.total_records > 0

        cohort_labels = [c.label for c in response.cohorts]
        assert cohort_labels == expected_labels

        for item in response.cohorts:
            assert isinstance(item, CohortItem)
            assert item.cohort_id.startswith(dimension)
            assert item.metrics.count > 0
            assert 0.0 <= item.metrics.addiction_prevalence <= 1.0
            assert item.metrics.screen_time_mean > 0.0
            assert item.metrics.sleep_hours_mean > 0.0
            assert item.metrics.screen_time_p50 > 0.0
            assert item.metrics.sleep_hours_p50 > 0.0

            # SPEC AC-2.1 aliases
            assert item.cohort_name == item.label
            assert item.sample_count == item.metrics.count
            assert item.addiction_prevalence == item.metrics.addiction_prevalence
            assert item.mean_screen_time == item.metrics.screen_time_mean
            assert item.mean_sleep_hours == item.metrics.sleep_hours_mean
            assert item.mean_app_opens == item.metrics.app_opens_mean

    def test_cohort_total_records_matches_sum(self, cohort_service: CohortService) -> None:
        """Verify response total_records equals sum of individual cohort counts."""
        params = CohortFilterParams(dimension="gender")
        response = cohort_service.get_cohorts(params)
        expected_total = sum(c.metrics.count for c in response.cohorts)
        assert response.total_records == expected_total


class TestDemographicSlicing:
    """Verification of conditioned cohort filters (AC-2.2)."""

    def test_slicing_by_gender_and_high_stress(self, cohort_service: CohortService) -> None:
        """Verify dimension=gender conditioned on High stress returns distinct high-stress summaries (AC-2.2)."""
        params = CohortFilterParams(dimension="gender", filter_stress="High")
        response = cohort_service.get_cohorts(params)

        assert len(response.cohorts) == 3
        labels = [c.label for c in response.cohorts]
        assert labels == ["Female", "Male", "Other"]

        # High-stress subpopulation counts
        for c in response.cohorts:
            assert c.metrics.count > 0
            assert 0.0 <= c.metrics.addiction_prevalence <= 1.0
            # Conditioned cohorts have different prevalence than population average
            assert c.metrics.addiction_prevalence > 0.65

        # Compare against unconditioned baseline
        baseline = cohort_service.get_cohorts(CohortFilterParams(dimension="gender"))
        assert response.total_records < baseline.total_records
        assert response.total_records == 211889

    def test_slicing_by_stress_and_female_gender(self, cohort_service: CohortService) -> None:
        """Verify dimension=stress_level conditioned on Female returns distinct Female slices."""
        params = CohortFilterParams(dimension="stress_level", filter_gender="Female")
        response = cohort_service.get_cohorts(params)

        assert len(response.cohorts) == 3
        labels = [c.label for c in response.cohorts]
        assert labels == ["Low", "Medium", "High"]

        total_female_samples = sum(c.metrics.count for c in response.cohorts)
        assert response.total_records == total_female_samples
        assert response.total_records > 0

    def test_case_insensitive_filter_values(self, cohort_service: CohortService) -> None:
        """Verify lowercase or uppercase filter parameters are normalized correctly."""
        p_lower = CohortFilterParams(dimension="gender", filter_stress="high")
        p_upper = CohortFilterParams(dimension="gender", filter_stress="HIGH")
        res_lower = cohort_service.get_cohorts(p_lower)
        res_upper = cohort_service.get_cohorts(p_upper)

        assert res_lower.total_records == res_upper.total_records
        assert [c.metrics.count for c in res_lower.cohorts] == [c.metrics.count for c in res_upper.cohorts]

    def test_filter_all_treated_as_unfiltered(self, cohort_service: CohortService) -> None:
        """Verify filter value 'all' returns unconditioned primary dimension."""
        unfiltered = cohort_service.get_cohorts(CohortFilterParams(dimension="age_bracket"))
        filtered_all = cohort_service.get_cohorts(
            CohortFilterParams(dimension="age_bracket", filter_gender="all", filter_stress="all")
        )
        assert unfiltered.total_records == filtered_all.total_records


class TestInvalidDimensionHandling:
    """Verification of invalid parameter rejection (AC-2.5)."""

    def test_pydantic_validation_error_on_invalid_dimension(self) -> None:
        """Verify invalid dimension parameter raises ValidationError in CohortFilterParams (AC-2.5)."""
        with pytest.raises(ValidationError) as exc_info:
            CohortFilterParams(dimension="invalid_dimension")  # type: ignore[arg-type]

        errors = str(exc_info.value)
        assert "dimension" in errors

    def test_service_value_error_on_bypassed_invalid_dimension(self, cohort_service: CohortService) -> None:
        """Verify CohortService explicitly raises ValueError if invalid dimension is bypassed (AC-2.5)."""
        bypassed_params = CohortFilterParams.model_construct(dimension="nonexistent_dim")
        with pytest.raises(ValueError, match="Permissible dimensions are"):
            cohort_service.get_cohorts(bypassed_params)


class TestScreenSleepMatrix:
    """Verification of 2D screen time vs sleep duration joint density grid (AC-2.3)."""

    def test_matrix_dimensions_and_structure(self, cohort_service: CohortService) -> None:
        """Verify get_screen_sleep_matrix returns 6x5 grid conforming to AC-2.3."""
        matrix_resp = cohort_service.get_screen_sleep_matrix()

        assert isinstance(matrix_resp, ScreenSleepMatrixResponse)
        assert len(matrix_resp.screen_bins) == 6
        assert len(matrix_resp.sleep_bins) == 5

        # 6x5 matrix
        assert len(matrix_resp.matrix) == 6
        assert all(len(row) == 5 for row in matrix_resp.matrix)

        # 30 cells
        assert len(matrix_resp.cells) == 30

    def test_cell_attributes_and_density_sum(self, cohort_service: CohortService) -> None:
        """Verify DensityCell attributes and population percentage sum equals ~100%."""
        matrix_resp = cohort_service.get_screen_sleep_matrix()

        total_density = 0.0
        total_samples = 0

        for cell in matrix_resp.cells:
            assert isinstance(cell, DensityCell)
            assert cell.screen_bin in matrix_resp.screen_bins
            assert cell.sleep_bin in matrix_resp.sleep_bins
            assert cell.count > 0
            assert 0.0 <= cell.density_pct <= 100.0
            assert 0.0 <= cell.addiction_rate_pct <= 100.0

            # Test aliases
            assert cell.sample_count == cell.count
            assert cell.cell_density_pct == cell.density_pct

            total_density += cell.density_pct
            total_samples += cell.count

        # Density sum over all cells must be ~100%
        assert 99.0 <= total_density <= 101.0
        assert total_samples > 0

    def test_matrix_optional_fields_populated(self, cohort_service: CohortService) -> None:
        """Verify density_matrix and addiction_rate_matrix are available for consumers."""
        matrix_resp = cohort_service.get_screen_sleep_matrix()
        assert matrix_resp.density_matrix is not None
        assert len(matrix_resp.density_matrix) == 6
        assert all(len(row) == 5 for row in matrix_resp.density_matrix)

        assert matrix_resp.addiction_rate_matrix is not None
        assert len(matrix_resp.addiction_rate_matrix) == 6


class TestBenchmarkOverlay:
    """Verification of personal quantile overlay calculation (AC-2.4)."""

    def test_median_inputs_yield_approx_50th_percentile(self, cohort_service: CohortService) -> None:
        """Verify population medians (7.77h screen, 6.8h sleep) yield ~50th percentiles."""
        req = BenchmarkOverlayRequest(daily_screen_time_hours=7.77, sleep_hours=6.8)
        resp = cohort_service.compute_benchmark_overlay(req)

        assert isinstance(resp, BenchmarkOverlayResponse)
        assert 49.0 <= resp.screen_time_percentile <= 51.0
        assert 49.0 <= resp.sleep_hours_percentile <= 51.0
        assert "50th percentile" in resp.screen_time_label
        assert "50th percentile" in resp.sleep_hours_label

    def test_extreme_inputs_high_screen_low_sleep(self, cohort_service: CohortService) -> None:
        """Verify 12h screen is >90th percentile and 5h sleep is <20th percentile (AC-2.4)."""
        req = BenchmarkOverlayRequest(daily_screen_time_hours=12.0, sleep_hours=5.0)
        resp = cohort_service.compute_benchmark_overlay(req)

        assert resp.screen_time_percentile > 90.0
        assert resp.sleep_hours_percentile < 20.0
        assert "percentile in screen time" in resp.screen_time_label
        assert "percentile in sleep duration" in resp.sleep_hours_label

        # SPEC AC-2.4 reference constants
        assert 7.55 <= resp.population_mean_screen <= 7.75
        assert 6.70 <= resp.population_mean_sleep <= 6.90
        # SPEC AC-2.4 alias
        assert resp.sleep_duration_percentile == resp.sleep_hours_percentile

    def test_optional_app_opens_and_notifications(self, cohort_service: CohortService) -> None:
        """Verify app opens and notifications percentiles are calculated when provided."""
        # When omitted
        req_omitted = BenchmarkOverlayRequest(daily_screen_time_hours=8.0, sleep_hours=7.0)
        resp_omitted = cohort_service.compute_benchmark_overlay(req_omitted)
        assert resp_omitted.app_opens_percentile is None
        assert resp_omitted.notifications_percentile is None

        # When provided
        req_provided = BenchmarkOverlayRequest(
            daily_screen_time_hours=8.0,
            sleep_hours=7.0,
            app_opens_per_day=104,  # exact median
            notifications_per_day=150,  # exact median
        )
        resp_provided = cohort_service.compute_benchmark_overlay(req_provided)
        assert resp_provided.app_opens_percentile is not None
        assert 48.0 <= resp_provided.app_opens_percentile <= 52.0
        assert resp_provided.notifications_percentile is not None
        assert 48.0 <= resp_provided.notifications_percentile <= 52.0

    def test_boundary_clamping(self, cohort_service: CohortService) -> None:
        """Verify inputs beyond population edges clamp gracefully to [0.0, 100.0]."""
        req_low = BenchmarkOverlayRequest(daily_screen_time_hours=0.0, sleep_hours=18.0)
        resp_low = cohort_service.compute_benchmark_overlay(req_low)
        assert 0.0 <= resp_low.screen_time_percentile <= 5.0
        assert 95.0 <= resp_low.sleep_hours_percentile <= 100.0

        req_high = BenchmarkOverlayRequest(daily_screen_time_hours=24.0, sleep_hours=1.0)
        resp_high = cohort_service.compute_benchmark_overlay(req_high)
        assert 95.0 <= resp_high.screen_time_percentile <= 100.0
        assert 0.0 <= resp_high.sleep_hours_percentile <= 5.0


class TestCohortServicePerformance:
    """Verification of sub-2ms query SLA (ADR-0003)."""

    def test_sub_2ms_query_execution_speed(self, cohort_service: CohortService) -> None:
        """Verify get_cohorts, get_screen_sleep_matrix, and compute_benchmark_overlay execute under 2ms."""
        filter_params = CohortFilterParams(dimension="age_bracket")
        overlay_req = BenchmarkOverlayRequest(
            daily_screen_time_hours=9.5,
            sleep_hours=5.5,
            app_opens_per_day=120,
            notifications_per_day=180,
        )

        # Warm-up pass
        for _ in range(10):
            cohort_service.get_cohorts(filter_params)
            cohort_service.get_screen_sleep_matrix()
            cohort_service.compute_benchmark_overlay(overlay_req)

        num_iterations = 100

        # Benchmark get_cohorts
        start = time.perf_counter()
        for _ in range(num_iterations):
            cohort_service.get_cohorts(filter_params)
        duration_ms = ((time.perf_counter() - start) / num_iterations) * 1000
        assert duration_ms < 2.0, f"get_cohorts exceeded 2ms SLA: {duration_ms:.3f} ms"

        # Benchmark get_screen_sleep_matrix
        start = time.perf_counter()
        for _ in range(num_iterations):
            cohort_service.get_screen_sleep_matrix()
        duration_ms = ((time.perf_counter() - start) / num_iterations) * 1000
        assert duration_ms < 2.0, f"get_screen_sleep_matrix exceeded 2ms SLA: {duration_ms:.3f} ms"

        # Benchmark compute_benchmark_overlay
        start = time.perf_counter()
        for _ in range(num_iterations):
            cohort_service.compute_benchmark_overlay(overlay_req)
        duration_ms = ((time.perf_counter() - start) / num_iterations) * 1000
        assert duration_ms < 2.0, f"compute_benchmark_overlay exceeded 2ms SLA: {duration_ms:.3f} ms"
