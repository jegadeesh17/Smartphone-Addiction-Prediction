"""Integration tests for modular FastAPI application REST endpoints.

Validates:
- GET /health and GET /api/health schema and telemetry (AC-1.8, AC-1.9).
- POST /api/predict schema, classification, ratios, interventions (AC-1.1, AC-1.4, AC-1.5).
- Threshold responsiveness (AC-1.3).
- Physiological bounds validation errors returning HTTP 422 (AC-1.6).
- Service unavailable 503 on missing model artifacts (AC-1.9).
- Mock model fallback toggle via settings override (ADR-0005, ADR-0007).
- Latency SLA <20ms p95 performance benchmark (AC-1.2).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.predict import clear_batch_export_cache
from src.config import Settings, get_settings
from src.inference import reset_inference_engine
from src.main import app


@pytest.fixture
def client() -> TestClient:
    """Create a FastAPI test client instance."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def clean_engine_and_overrides() -> None:
    """Reset singletons and FastAPI dependency overrides after each test."""
    reset_inference_engine()
    clear_batch_export_cache()
    app.dependency_overrides.clear()
    get_settings.cache_clear()
    yield
    reset_inference_engine()
    clear_batch_export_cache()
    app.dependency_overrides.clear()
    get_settings.cache_clear()


# ==============================================================================
# 1. Health & Telemetry Endpoints
# ==============================================================================


class TestHealthEndpoint:
    """Test suite for GET /health and GET /api/health endpoints."""

    def test_health_returns_200_and_valid_schema(self, client: TestClient) -> None:
        """Verify GET /health returns 200 OK and conforms to HealthResponse schema (AC-1.8)."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["version"] == "1.0.0"
        assert data["model_family"] == "LightGBM"
        assert data["fold"] == 1
        assert data["priors_loaded"] is True
        assert data["cohort_priors_cached"] is True
        assert data["mock_mode"] is False
        assert isinstance(data["uptime_seconds"], (int, float))
        assert data["uptime_seconds"] >= 0.0

    def test_api_health_endpoint_alias(self, client: TestClient) -> None:
        """Verify GET /api/health alias returns 200 OK with identical telemetry structure."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["version"] == "1.0.0"

    def test_health_mock_mode_telemetry(self, client: TestClient) -> None:
        """Verify /health reflects mock_mode=True when configured with USE_MOCK_MODEL=True."""
        mock_settings = Settings(USE_MOCK_MODEL=True)
        app.dependency_overrides[get_settings] = lambda: mock_settings

        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["mock_mode"] is True
        assert data["model_family"] == "MockHeuristic"

    def test_health_returns_503_when_model_artifact_missing(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """Verify GET /health returns 503 Service Unavailable if model checkpoint is missing (AC-1.9)."""
        missing_settings = Settings(
            MODEL_PATH=tmp_path / "nonexistent_model.joblib",
            USE_MOCK_MODEL=False,
        )
        app.dependency_overrides[get_settings] = lambda: missing_settings

        response = client.get("/health")
        assert response.status_code == 503
        data = response.json()
        assert "artifact missing" in str(data).lower()


# ==============================================================================
# 2. Prediction Endpoint (POST /api/predict)
# ==============================================================================


class TestPredictEndpoint:
    """Test suite for POST /api/predict live inference endpoint."""

    def test_predict_valid_payload_success(
        self, client: TestClient, sample_valid_profile: dict[str, Any]
    ) -> None:
        """Verify POST /api/predict returns 200 OK and complete PredictionResponse (AC-1.1)."""
        response = client.post("/api/predict", json=sample_valid_profile)
        assert response.status_code == 200
        data = response.json()

        # Contract fields
        assert "probability" in data
        assert 0.0 <= data["probability"] <= 1.0
        assert data["prediction"] == 1
        assert data["classification"] == "ADDICTION DETECTED"
        assert data["severity"] == "Moderate"
        assert data["risk_tier"] == "MODERATE"
        assert data["status_label"].startswith("ADDICTION DETECTED • MODERATE RISK")
        assert data["latency_ms"] > 0.0
        assert data["decision_threshold"] == sample_valid_profile["decision_threshold"]

        # Ratios / metrics fields
        ratios = data["ratios"]
        assert "screen_to_sleep" in ratios or "screen_to_sleep_ratio" in ratios
        assert "recreational_to_screen" in ratios or "recreational_share" in ratios
        assert "avg_unlock_minutes" in ratios
        assert "weekend_diff" in ratios or "weekend_surge_hours" in ratios

        # Interventions list
        assert isinstance(data["interventions"], list)

    def test_predict_alias_endpoint(
        self, client: TestClient, sample_valid_profile: dict[str, Any]
    ) -> None:
        """Verify POST /predict alias endpoint works interchangeably with /api/predict."""
        response = client.post("/predict", json=sample_valid_profile)
        assert response.status_code == 200
        assert "probability" in response.json()

    def test_predict_threshold_responsiveness(
        self, client: TestClient, sample_valid_profile: dict[str, Any]
    ) -> None:
        """Verify classification flips between ADDICTION DETECTED and HEALTHY across thresholds (AC-1.3)."""
        # Low threshold (0.10) guarantees addiction detection
        low_tau_payload = {**sample_valid_profile, "decision_threshold": 0.10}
        res_low = client.post("/api/predict", json=low_tau_payload)
        assert res_low.status_code == 200
        data_low = res_low.json()
        assert data_low["prediction"] == 1
        assert data_low["classification"] == "ADDICTION DETECTED"
        assert data_low["severity"] == "Moderate"
        assert data_low["risk_tier"] == "MODERATE"
        assert data_low["status_label"].startswith("ADDICTION DETECTED • MODERATE RISK")

        # High threshold (0.95) guarantees healthy classification
        high_tau_payload = {**sample_valid_profile, "decision_threshold": 0.95}
        res_high = client.post("/api/predict", json=high_tau_payload)
        assert res_high.status_code == 200
        data_high = res_high.json()
        assert data_high["prediction"] == 0
        assert data_high["classification"] == "HEALTHY"
        assert data_high["severity"] == "Moderate"
        assert data_high["risk_tier"] == "MODERATE"
        assert data_high["status_label"].startswith("HEALTHY PATTERN • MODERATE RISK")

        # Underlying model probabilities should remain identical
        assert data_low["probability"] == pytest.approx(data_high["probability"], abs=1e-4)

    def test_predict_ratio_precision(self, client: TestClient) -> None:
        """Verify exact mathematical computation of behavioral ratios (AC-1.4)."""
        payload = {
            "age": 25,
            "gender": "Male",
            "stress_level": "Medium",
            "academic_work_impact": "Yes",
            "daily_screen_time_hours": 8.0,
            "sleep_hours": 6.0,
            "social_media_hours": 3.0,
            "gaming_hours": 1.0,
            "work_study_hours": 2.0,
            "app_opens_per_day": 80,
            "weekend_screen_time": 11.0,
            "notifications_per_day": 100,
            "decision_threshold": 0.50,
        }
        response = client.post("/api/predict", json=payload)
        assert response.status_code == 200
        ratios = response.json()["ratios"]

        assert ratios["screen_to_sleep_ratio"] == pytest.approx(8.0 / 6.0, abs=0.001)
        assert ratios["recreational_share"] == pytest.approx((3.0 + 1.0) / 8.0, abs=0.001)
        assert ratios["avg_unlock_minutes"] == pytest.approx((8.0 * 60.0) / 80.0, abs=0.001)
        assert ratios["weekend_surge_hours"] == pytest.approx(11.0 - 8.0, abs=0.001)

    def test_predict_rule_based_intervention_triggers(self, client: TestClient) -> None:
        """Verify all 4 clinical digital hygiene recommendations are triggered (AC-1.5)."""
        # Triggers:
        # 1. Screen/Sleep > 1.2 -> screen 8.5 / sleep 5.5 = 1.54
        # 2. Recreational > 0.60 -> (3.5 + 2.5) / 8.5 = 0.706
        # 3. App Opens > 120 -> 135
        # 4. Weekend surge > 2.5h -> 11.5 - 8.5 = 3.0h
        payload = {
            "age": 24,
            "gender": "Female",
            "stress_level": "High",
            "academic_work_impact": "Yes",
            "daily_screen_time_hours": 8.5,
            "sleep_hours": 5.5,
            "social_media_hours": 3.5,
            "gaming_hours": 2.5,
            "work_study_hours": 2.0,
            "weekend_screen_time": 11.5,
            "notifications_per_day": 180,
            "app_opens_per_day": 135,
            "decision_threshold": 0.50,
        }
        response = client.post("/api/predict", json=payload)
        assert response.status_code == 200
        interventions = response.json()["interventions"]

        joined_recs = " ".join(interventions)
        assert "Sleep Protection" in joined_recs
        assert "Recreation Audit" in joined_recs
        assert "Notification Hygiene" in joined_recs
        assert "Weekend Disconnect" in joined_recs

    @pytest.mark.parametrize(
        "invalid_field, invalid_value",
        [
            ("age", 15),  # Min age is 18
            ("age", 45),  # Max age is 35
            ("daily_screen_time_hours", -1.0),  # Negative screen time
            ("daily_screen_time_hours", 25.0),  # Screen time > 24h
            ("sleep_hours", 0.5),  # Min sleep is 1.0h
            ("sleep_hours", 19.0),  # Max sleep is 18.0h
            ("notifications_per_day", -5),  # Negative notification count
            ("app_opens_per_day", -10),  # Negative opens count
            ("gender", "InvalidGender"),  # Categorical enum mismatch
            ("stress_level", "Extreme"),  # Categorical enum mismatch
            ("academic_work_impact", "Maybe"),  # Categorical enum mismatch
        ],
    )
    def test_predict_physiological_boundary_validation_rejection(
        self,
        client: TestClient,
        sample_valid_profile: dict[str, Any],
        invalid_field: str,
        invalid_value: Any,
    ) -> None:
        """Verify physiological boundary breaches return HTTP 422 Unprocessable Entity (AC-1.6)."""
        payload = {**sample_valid_profile, invalid_field: invalid_value}
        response = client.post("/api/predict", json=payload)
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        assert "error" in data or "detail" in data

    def test_predict_service_unavailable_on_missing_model_file(
        self, client: TestClient, sample_valid_profile: dict[str, Any], tmp_path: Path
    ) -> None:
        """Verify POST /api/predict returns 503 Service Unavailable when model file is missing (AC-1.9)."""
        missing_settings = Settings(
            MODEL_PATH=tmp_path / "missing_model.joblib",
            USE_MOCK_MODEL=False,
        )
        app.dependency_overrides[get_settings] = lambda: missing_settings

        response = client.post("/api/predict", json=sample_valid_profile)
        assert response.status_code == 503
        data = response.json()
        assert "error" in data
        assert "Service Unavailable" in data["error"]


# ==============================================================================
# 3. Mock Model Fallback Toggle via Settings Override
# ==============================================================================


class TestMockModelFallbackToggle:
    """Test suite for mock inference adapter execution via settings toggle (ADR-0005, ADR-0007)."""

    def test_mock_model_evaluates_with_calibrated_probability(
        self, client: TestClient, sample_valid_profile: dict[str, Any]
    ) -> None:
        """Verify USE_MOCK_MODEL=True executes calibrated logit heuristic without LightGBM."""
        mock_settings = Settings(USE_MOCK_MODEL=True)
        app.dependency_overrides[get_settings] = lambda: mock_settings

        response = client.post("/api/predict", json=sample_valid_profile)
        assert response.status_code == 200
        data = response.json()

        # Calibrated baseline centered heuristic yields ~50.5% (48-52%) on standard population medians (ADR-0007)
        assert 0.48 <= data["probability"] <= 0.53
        assert data["latency_ms"] < 10.0
        assert data["severity"] == "Moderate"
        assert data["risk_tier"] == "MODERATE"
        assert data["status_label"].startswith("ADDICTION DETECTED • MODERATE RISK")


# ==============================================================================
# 4. Latency Benchmark Performance (<20ms SLA)
# ==============================================================================


class TestApiLatencyPerformance:
    """Latency benchmark suite validating AC-1.2 (<20ms p95 single-row inference SLA)."""

    def test_predict_latency_p95_under_20ms(
        self, client: TestClient, sample_valid_profile: dict[str, Any]
    ) -> None:
        """Verify 50 sequential requests to /api/predict achieve p95 latency < 20ms (AC-1.2)."""
        # Warmup calls to initialize runtime caches and JIT paths
        for _ in range(5):
            client.post("/api/predict", json=sample_valid_profile)

        server_latencies_ms: list[float] = []
        client_latencies_ms: list[float] = []
        for _ in range(50):
            t0 = time.perf_counter()
            response = client.post("/api/predict", json=sample_valid_profile)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            assert response.status_code == 200
            data = response.json()
            client_latencies_ms.append(elapsed_ms)
            server_latencies_ms.append(float(data["latency_ms"]))

        server_p95 = float(np.percentile(server_latencies_ms, 95))
        server_mean = float(np.mean(server_latencies_ms))
        assert (
            server_p95 < 20.0
        ), f"Server p95 latency {server_p95:.2f}ms exceeded 20ms SLA (mean: {server_mean:.2f}ms)"
        assert (
            server_mean < 15.0
        ), f"Server mean latency {server_mean:.2f}ms exceeded 15ms target"


# ==============================================================================
# 5. CORS Middleware & Static Index Serving
# ==============================================================================


class TestCorsAndRootRoutes:
    """Test suite for CORS headers and static asset delivery."""

    def test_cors_headers_present(self, client: TestClient) -> None:
        """Verify CORS headers are returned for cross-origin requests."""
        response = client.options(
            "/api/predict",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers

    def test_root_and_app_endpoints(self, client: TestClient) -> None:
        """Verify GET / and GET /app return 200 OK with text/html content."""
        res_root = client.get("/")
        assert res_root.status_code == 200
        assert "text/html" in res_root.headers.get("content-type", "")

        res_app = client.get("/app")
        assert res_app.status_code == 200
        assert "text/html" in res_app.headers.get("content-type", "")

    def test_root_serves_app_shell_and_telemetry(self, client: TestClient) -> None:
        """Verify GET / delivers app shell header, brand title, and telemetry strip."""
        response = client.get("/")
        assert response.status_code == 200
        html = response.text

        # Brand mark and title
        assert "Smartphone Addiction Analytical Platform" in html
        assert "LightGBM Tabular Risk Assessment" in html

        # Status indicator and telemetry strip
        assert "telemetry-strip" in html
        assert "status-indicator-static" in html
        assert "MODEL: LIGHTGBM FOLD 1" in html
        assert "691,369" in html

        # 4-tab navigation structure
        assert "Individual Diagnostic" in html
        assert "Population Cohort Analytics" in html
        assert "What-If Simulation" in html
        assert "Batch Diagnostics" in html

    def test_root_serves_multi_model_benchmark_table_par7(self, client: TestClient) -> None:
        """Verify GET / renders the 5-fold CV benchmark performance table conforming to PAR-7."""
        response = client.get("/")
        assert response.status_code == 200
        html = response.text

        # Table presence
        assert "benchmark-table-wrapper" in html
        assert "Model Benchmark Performance" in html

        # PAR-7 Architectures and Metrics
        assert "LightGBM Fold 1 (Active)" in html
        assert "0.96394" in html
        assert "90.26%" in html
        assert "Hist XGBoost" in html
        assert "0.96342" in html
        assert "Symmetric CatBoost" in html
        assert "0.96000" in html
        assert "PyTorch Tabular ResNet" in html
        assert "0.95750" in html
        assert "Optimized Logit Ensemble" in html
        assert "0.96410+" in html

    def test_static_css_assets_served(self, client: TestClient) -> None:
        """Verify static CSS design tokens and style assets are served under /static."""
        for path in ["/static/css/app.css", "/static/css/tokens.css", "/static/css/style.css"]:
            response = client.get(path)
            assert response.status_code == 200
            assert "text/css" in response.headers.get("content-type", "")
            assert len(response.text) > 0

    def test_nonexistent_static_asset_returns_404(self, client: TestClient) -> None:
        """Verify requesting nonexistent static asset returns HTTP 404 Not Found."""
        response = client.get("/static/css/nonexistent_token_file.css")
        assert response.status_code == 404


# ==============================================================================
# 6. Individual Diagnostic View & UI Components (M1-TASK-07)
# ==============================================================================


class TestIndividualDiagnosticView:
    """Test suite validating UI template delivery and components for M1-TASK-07."""

    def test_static_js_assets_served(self, client: TestClient) -> None:
        """Verify static JavaScript visualization and client application files are served."""
        for path in ["/static/js/charts.js", "/static/js/app.js"]:
            response = client.get(path)
            assert response.status_code == 200
            assert any(
                js_type in response.headers.get("content-type", "")
                for js_type in ("javascript", "application/x-javascript", "text/plain")
            )
            assert len(response.text) > 0

        # Check charts.js exports and geometry
        charts_js = client.get("/static/js/charts.js").text
        assert "renderRiskGauge" in charts_js
        assert "getClinicalTier" in charts_js
        assert "GAUGE_ARC_LENGTH = 326.4" in charts_js
        assert "3-tier" in charts_js or "Elevated Risk Tier" in charts_js

        # Check app.js debouncing and API integration
        app_js = client.get("/static/js/app.js").text
        assert "/api/predict" in app_js
        assert "triggerDiagnosticPrediction" in app_js
        assert "schedulePrediction" in app_js

    def test_individual_diagnostic_input_controls_par1(self, client: TestClient) -> None:
        """Verify index.html delivers all 12 behavioral input controls + threshold slider (PAR-1)."""
        response = client.get("/")
        assert response.status_code == 200
        html = response.text

        # 12 features controls
        assert 'id="input-age"' in html
        assert 'data-group="gender"' in html
        assert 'data-group="stress"' in html
        assert 'data-group="impact"' in html
        assert 'id="input-daily-screen"' in html
        assert 'id="input-social-media"' in html
        assert 'id="input-gaming"' in html
        assert 'id="input-work-study"' in html
        assert 'id="input-weekend-screen"' in html
        assert 'id="input-sleep"' in html
        assert 'id="input-notifications"' in html
        assert 'id="input-app-opens"' in html

        # Threshold slider (tau)
        assert 'id="input-threshold"' in html
        assert 'id="val-threshold"' in html

    def test_individual_diagnostic_svg_gauge_adr0008(self, client: TestClient) -> None:
        """Verify SVG circular arc risk gauge adheres to geometry and clearance contract (ADR-0008)."""
        response = client.get("/")
        assert response.status_code == 200
        html = response.text

        # 220-degree sweep arc in viewBox 0 0 240 170
        assert 'viewBox="0 0 240 170"' in html
        assert 'class="gauge-bg"' in html
        assert 'id="gauge-arc"' in html
        assert 'id="gauge-probability-num"' in html

        # Dedicated external caption container with positive margin clearance
        assert 'class="gauge-caption"' in html
        assert "Addiction Risk Probability" in html

    def test_individual_diagnostic_status_pill_adr0006(self, client: TestClient) -> None:
        """Verify consolidated authoritative status pill element is present (ADR-0006)."""
        response = client.get("/")
        assert response.status_code == 200
        html = response.text

        assert 'id="status-pill"' in html
        assert 'id="badge-status"' in html
        assert 'id="classification-summary"' in html

    def test_individual_diagnostic_ratio_metric_cards_ac14(self, client: TestClient) -> None:
        """Verify 4 granular ratio diagnostic cards with target indicators (AC-1.4, PAR-2)."""
        response = client.get("/")
        assert response.status_code == 200
        html = response.text

        assert 'id="metric-screen-sleep"' in html
        assert 'id="target-screen-sleep"' in html
        assert 'id="metric-rec-share"' in html
        assert 'id="target-rec-share"' in html
        assert 'id="metric-unlock-mins"' in html
        assert 'id="target-unlock-mins"' in html
        assert 'id="metric-weekend-surge"' in html
        assert 'id="target-weekend-surge"' in html

    def test_individual_diagnostic_boundary_warning_ac17(self, client: TestClient) -> None:
        """Verify physiological boundary warning indicator markup is present (AC-1.7)."""
        response = client.get("/")
        assert response.status_code == 200
        html = response.text

        assert 'id="boundary-warning"' in html
        assert 'id="excess-hours-msg"' in html

    def test_individual_diagnostic_interventions_container_ac15(self, client: TestClient) -> None:
        """Verify clinical interventions list container is present (AC-1.5, PAR-5)."""
        response = client.get("/")
        assert response.status_code == 200
        html = response.text

        assert 'id="intervention-list"' in html
        assert "Recommended Clinical Interventions" in html

    def test_no_misspelled_addiciton_par6(self, client: TestClient) -> None:
        """Verify absence of legacy typo 'ADDICITON' across UI templates, scripts, and API responses (PAR-6)."""
        # Template markup check
        html = client.get("/").text
        assert "ADDICITON" not in html
        assert "addiciton" not in html

        # JS scripts check
        app_js = client.get("/static/js/app.js").text
        assert "ADDICITON" not in app_js
        assert "addiciton" not in app_js

        charts_js = client.get("/static/js/charts.js").text
        assert "ADDICITON" not in charts_js
        assert "addiciton" not in charts_js

        # API payload check
        resp = client.post(
            "/api/predict",
            json={
                "age": 25,
                "gender": "Male",
                "stress_level": "High",
                "academic_work_impact": "Yes",
                "daily_screen_time_hours": 10.0,
                "sleep_hours": 5.0,
                "social_media_hours": 4.0,
                "gaming_hours": 2.0,
                "work_study_hours": 2.0,
                "app_opens_per_day": 120,
                "weekend_screen_time": 12.0,
                "notifications_per_day": 180,
                "decision_threshold": 0.50,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "ADDICITON" not in str(data)
        assert data["classification"] == "ADDICTION DETECTED"


# ==============================================================================
# 9. Cohort Analytics Endpoints (Milestone 2, Journey 2)
# ==============================================================================


class TestCohortAnalyticsApi:
    """Integration test suite for Cohort Analytics endpoints (AC-2.1 to AC-2.5)."""

    def test_get_cohorts_default_dimension_returns_200(self, client: TestClient) -> None:
        """Verify GET /api/analytics/cohorts returns 200 OK and default age_bracket cohort summaries (AC-2.1)."""
        response = client.get("/api/analytics/cohorts")
        assert response.status_code == 200
        data = response.json()

        assert data["dimension"] == "age_bracket"
        assert data["total_records"] > 0
        assert isinstance(data["cohorts"], list)
        assert len(data["cohorts"]) == 4

        for cohort in data["cohorts"]:
            # Standard & AC-2.1 aliases
            assert "cohort_id" in cohort
            assert cohort["cohort_id"].startswith("age_bracket")
            assert "label" in cohort
            assert "cohort_name" in cohort
            assert cohort["cohort_name"] in ["18-21", "22-25", "26-30", "31-35"]
            assert cohort["sample_count"] > 0
            assert 0.0 <= cohort["addiction_prevalence"] <= 1.0
            assert cohort["mean_screen_time"] > 0.0
            assert cohort["mean_sleep_hours"] > 0.0
            assert cohort["mean_app_opens"] > 0.0

            # Metrics sub-object
            metrics = cohort["metrics"]
            assert metrics["count"] == cohort["sample_count"]
            assert metrics["addiction_prevalence"] == cohort["addiction_prevalence"]

    @pytest.mark.parametrize(
        "dimension, expected_labels",
        [
            ("age_bracket", ["18-21", "22-25", "26-30", "31-35"]),
            ("gender", ["Female", "Male", "Other"]),
            ("stress_level", ["Low", "Medium", "High"]),
            ("academic_work_impact", ["Yes", "No"]),
        ],
    )
    def test_get_cohorts_all_dimensions_return_200(
        self, client: TestClient, dimension: str, expected_labels: list[str]
    ) -> None:
        """Verify GET /api/analytics/cohorts returns 200 for all 4 primary demographic dimensions (AC-2.1)."""
        response = client.get(f"/api/analytics/cohorts?dimension={dimension}")
        assert response.status_code == 200
        data = response.json()

        assert data["dimension"] == dimension
        assert len(data["cohorts"]) == len(expected_labels)
        labels = [c["cohort_name"] for c in data["cohorts"]]
        assert labels == expected_labels
        assert data["total_records"] == sum(c["sample_count"] for c in data["cohorts"])

    def test_get_cohorts_filtering_by_gender_and_stress(self, client: TestClient) -> None:
        """Verify conditioned demographic slicing by gender and chronic stress level (AC-2.2)."""
        # Slicing gender by High stress
        response = client.get("/api/analytics/cohorts?dimension=gender&filter_stress=High")
        assert response.status_code == 200
        data = response.json()

        assert data["dimension"] == "gender"
        assert len(data["cohorts"]) == 3
        labels = [c["cohort_name"] for c in data["cohorts"]]
        assert labels == ["Female", "Male", "Other"]
        # High stress subpopulation count must be strictly less than unconditioned population
        assert data["total_records"] < 691369
        for c in data["cohorts"]:
            assert c["sample_count"] > 0
            assert c["addiction_prevalence"] > 0.65

        # Slicing stress_level by Female gender
        res_gender_filter = client.get("/api/analytics/cohorts?dimension=stress_level&filter_gender=Female")
        assert res_gender_filter.status_code == 200
        data_gender = res_gender_filter.json()
        assert data_gender["dimension"] == "stress_level"
        assert len(data_gender["cohorts"]) == 3
        assert [c["cohort_name"] for c in data_gender["cohorts"]] == ["Low", "Medium", "High"]

    @pytest.mark.parametrize(
        "invalid_dimension",
        [
            "invalid_dimension",
            "income_level",
            "occupation",
            "device_brand",
            "unknown",
        ],
    )
    def test_get_cohorts_invalid_dimension_returns_422(
        self, client: TestClient, invalid_dimension: str
    ) -> None:
        """Verify invalid dimension query parameter returns HTTP 422 with permissible dimensions (AC-2.5)."""
        response = client.get(f"/api/analytics/cohorts?dimension={invalid_dimension}")
        assert response.status_code == 422
        data = response.json()
        # Must contain explicit list of permissible dimensions per AC-2.5
        err_msg = str(data)
        assert "Invalid dimension" in err_msg or "Permissible dimensions" in err_msg
        assert "age_bracket" in err_msg
        assert "gender" in err_msg
        assert "stress_level" in err_msg
        assert "academic_work_impact" in err_msg

    def test_get_screen_sleep_matrix_returns_200(self, client: TestClient) -> None:
        """Verify GET /api/analytics/distributions/screen-sleep-matrix returns 200 OK and 6x5 grid (AC-2.3)."""
        response = client.get("/api/analytics/distributions/screen-sleep-matrix")
        assert response.status_code == 200
        data = response.json()

        assert len(data["screen_bins"]) == 6
        assert len(data["sleep_bins"]) == 5
        assert len(data["matrix"]) == 6
        assert all(len(row) == 5 for row in data["matrix"])
        assert len(data["cells"]) == 30

        total_density = sum(c["density_pct"] for c in data["cells"])
        assert 99.0 <= total_density <= 101.0

        for cell in data["cells"]:
            assert cell["screen_bin"] in data["screen_bins"]
            assert cell["sleep_bin"] in data["sleep_bins"]
            assert cell["count"] > 0
            assert cell["sample_count"] == cell["count"]
            assert 0.0 <= cell["density_pct"] <= 100.0
            assert 0.0 <= cell["addiction_rate_pct"] <= 100.0

    def test_post_benchmark_overlay_returns_200(self, client: TestClient) -> None:
        """Verify POST /api/analytics/distributions/benchmark-overlay returns 200 OK with percentiles (AC-2.4)."""
        payload = {
            "daily_screen_time_hours": 9.5,
            "sleep_hours": 5.0,
            "app_opens_per_day": 120,
            "notifications_per_day": 180,
        }
        response = client.post("/api/analytics/distributions/benchmark-overlay", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert 0.0 <= data["screen_time_percentile"] <= 100.0
        assert 0.0 <= data["sleep_hours_percentile"] <= 100.0
        # 9.5h screen is ~70.33 percentile (between median 7.77h and 75th pct 9.84h), 5.0h sleep is low (<20%)
        assert 68.0 <= data["screen_time_percentile"] <= 72.0
        assert data["sleep_hours_percentile"] < 20.0
        assert data["sleep_duration_percentile"] == data["sleep_hours_percentile"]

        # High screen time (12h) is >80th percentile
        high_screen_res = client.post(
            "/api/analytics/distributions/benchmark-overlay",
            json={"daily_screen_time_hours": 12.0, "sleep_hours": 5.0},
        )
        assert high_screen_res.status_code == 200
        assert high_screen_res.json()["screen_time_percentile"] > 80.0

        # SPEC AC-2.4 population mean reference values (7.64 +- 0.05, 6.80 +- 0.05)
        assert 7.55 <= data["population_mean_screen"] <= 7.75
        assert 6.70 <= data["population_mean_sleep"] <= 6.90

        # Percentile narrative labels
        assert "percentile in screen time" in data["screen_time_label"]
        assert "percentile in sleep duration" in data["sleep_hours_label"]

        # Optional interaction percentiles
        assert data["app_opens_percentile"] is not None
        assert data["notifications_percentile"] is not None
        assert 0.0 <= data["app_opens_percentile"] <= 100.0
        assert 0.0 <= data["notifications_percentile"] <= 100.0

    @pytest.mark.parametrize(
        "invalid_payload",
        [
            {"daily_screen_time_hours": -1.0, "sleep_hours": 7.0},  # screen < 0.0
            {"daily_screen_time_hours": 25.0, "sleep_hours": 7.0},  # screen > 24.0
            {"daily_screen_time_hours": 8.0, "sleep_hours": 0.5},   # sleep < 1.0
            {"daily_screen_time_hours": 8.0, "sleep_hours": 19.0},  # sleep > 18.0
            {"daily_screen_time_hours": 8.0, "sleep_hours": 7.0, "app_opens_per_day": -5},
            {"daily_screen_time_hours": 8.0, "sleep_hours": 7.0, "notifications_per_day": 600},
            {"daily_screen_time_hours": 8.0},                       # missing sleep_hours
            {"sleep_hours": 7.0},                                   # missing daily_screen_time_hours
            {},                                                     # empty payload
        ],
    )
    def test_post_benchmark_overlay_invalid_payload_returns_422(
        self, client: TestClient, invalid_payload: dict[str, Any]
    ) -> None:
        """Verify invalid or out-of-bounds benchmark overlay payloads return HTTP 422 Unprocessable Entity."""
        response = client.post("/api/analytics/distributions/benchmark-overlay", json=invalid_payload)
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data or "error" in data


# ==============================================================================
# 10. Population Cohort Analytics Explorer UI View (M2-TASK-04, Journey 2)
# ==============================================================================


class TestCohortAnalyticsView:
    """Test suite validating UI template delivery, controls, and script exports for M2-TASK-04."""

    def test_cohort_tab_and_segmented_controls(self, client: TestClient) -> None:
        """Verify index.html delivers #tab-cohorts with 4 primary dimension and stress filter buttons (AC-2.1, AC-2.2)."""
        response = client.get("/")
        assert response.status_code == 200
        html = response.text

        # Tab container
        assert 'id="tab-cohorts"' in html
        assert 'role="tabpanel"' in html

        # Primary dimension segmented control buttons
        assert 'data-cohort-dim="age_bracket"' in html
        assert 'data-cohort-dim="gender"' in html
        assert 'data-cohort-dim="stress_level"' in html
        assert 'data-cohort-dim="academic_work_impact"' in html

        # Stress condition filter buttons
        assert 'data-stress-filter="All"' in html
        assert 'data-stress-filter="Low"' in html
        assert 'data-stress-filter="Medium"' in html
        assert 'data-stress-filter="High"' in html

    def test_cohort_summary_metrics_markup(self, client: TestClient) -> None:
        """Verify population summary metrics cards are delivered in #tab-cohorts."""
        response = client.get("/")
        assert response.status_code == 200
        html = response.text

        assert 'id="cohort-total-records"' in html
        assert 'id="cohort-overall-prevalence"' in html
        assert 'id="cohort-mean-screen"' in html
        assert 'id="cohort-mean-sleep"' in html

    def test_cohort_cards_container_markup(self, client: TestClient) -> None:
        """Verify dynamic demographic sub-cohort card container markup is delivered."""
        response = client.get("/")
        assert response.status_code == 200
        html = response.text

        assert 'id="cohort-cards-container"' in html
        assert 'class="cohort-card-grid"' in html

    def test_heatmap_container_and_grid_markup(self, client: TestClient) -> None:
        """Verify 2D Screen vs Sleep Joint Density Heatmap markup is delivered (AC-2.3)."""
        response = client.get("/")
        assert response.status_code == 200
        html = response.text

        assert 'class="heatmap-container"' in html
        assert 'class="heatmap-wrapper"' in html
        assert 'id="heatmap-grid"' in html
        assert "2D Joint Density &amp; Addiction Heatmap" in html or "2D Joint Density" in html

    def test_quantile_overlay_panel_markup(self, client: TestClient) -> None:
        """Verify personal quantile benchmark overlay panel and ranking tracks are delivered (AC-2.4)."""
        response = client.get("/")
        assert response.status_code == 200
        html = response.text

        # Quantile metric value and progress bar IDs
        assert 'id="val-quantile-screen"' in html
        assert 'id="bar-quantile-screen"' in html
        assert 'id="val-user-screen-echo"' in html

        assert 'id="val-quantile-sleep"' in html
        assert 'id="bar-quantile-sleep"' in html
        assert 'id="val-user-sleep-echo"' in html

        assert 'id="val-quantile-opens"' in html
        assert 'id="bar-quantile-opens"' in html
        assert 'id="val-user-opens-echo"' in html

        assert 'id="val-quantile-notifs"' in html
        assert 'id="bar-quantile-notifs"' in html
        assert 'id="val-user-notifs-echo"' in html

    def test_cohort_charts_and_app_js_exports(self, client: TestClient) -> None:
        """Verify charts.js and app.js implement cohort rendering and endpoint integration (M2-TASK-04)."""
        # charts.js exports
        charts_js = client.get("/static/js/charts.js").text
        assert "renderCohortCards" in charts_js
        assert "renderScreenSleepHeatmap" in charts_js
        assert "renderQuantileOverlays" in charts_js

        # app.js cohort logic and analytics routes
        app_js = client.get("/static/js/app.js").text
        assert "loadCohortAnalytics" in app_js
        assert "loadHeatmapMatrix" in app_js
        assert "updateBenchmarkOverlay" in app_js
        assert "/api/analytics/cohorts" in app_js
        assert "/api/analytics/distributions/screen-sleep-matrix" in app_js
        assert "/api/analytics/distributions/benchmark-overlay" in app_js


# ==============================================================================
# 11. Counterfactual What-If Simulation and Habit Optimizer (M3-TASK-01, AC-3.1, AC-3.2)
# ==============================================================================


class TestWhatIfAndOptimizerIntegration:
    """Integration test suite for POST /api/analytics/what-if and /what-if/optimize endpoints."""

    def test_whatif_habit_improvement_scenario(self, client: TestClient) -> None:
        """Verify counterfactual What-If scenario evaluation returns risk delta < 0 (AC-3.1)."""
        payload = {
            "baseline_profile": {
                "age": 25,
                "gender": "Male",
                "stress_level": "Medium",
                "academic_work_impact": "Yes",
                "daily_screen_time_hours": 8.0,
                "social_media_hours": 3.5,
                "gaming_hours": 1.5,
                "work_study_hours": 2.5,
                "weekend_screen_time": 9.5,
                "sleep_hours": 6.5,
                "notifications_per_day": 140,
                "app_opens_per_day": 100,
                "decision_threshold": 0.50,
            },
            "delta_social_media_hours": -2.0,
            "delta_sleep_hours": 1.5,
            "delta_app_opens_per_day": -30,
        }
        response = client.post("/api/analytics/what-if", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert 0.76 <= data["baseline_probability"] <= 0.80
        assert data["simulated_probability"] < data["baseline_probability"]
        assert data["risk_delta"] < 0.0
        assert data["baseline_classification"] == "ADDICTION DETECTED"
        assert data["simulated_classification"] in ("HEALTHY", "ADDICTION DETECTED")
        assert "screen_to_sleep_ratio" in data["simulated_ratios"]
        assert "recreational_share" in data["simulated_ratios"]
        assert data["simulated_ratios"]["screen_to_sleep_ratio"] < data["baseline_ratios"]["screen_to_sleep_ratio"]

    def test_whatif_worsening_habits_scenario(self, client: TestClient) -> None:
        """Verify worsening habits increases simulated probability and risk delta > 0."""
        payload = {
            "baseline_profile": {
                "daily_screen_time_hours": 5.0,
                "sleep_hours": 7.5,
                "app_opens_per_day": 50,
                "notifications_per_day": 60,
            },
            "delta_daily_screen_time_hours": 3.5,
            "delta_sleep_hours": -2.0,
            "delta_app_opens_per_day": 40,
            "delta_notifications_per_day": 50,
        }
        response = client.post("/api/analytics/what-if", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["simulated_probability"] > data["baseline_probability"]
        assert data["risk_delta"] > 0.0

    def test_whatif_boundary_clamping(self, client: TestClient) -> None:
        """Verify boundary clamping enforces valid physiological ranges in simulated profile."""
        payload = {
            "baseline_profile": {
                "daily_screen_time_hours": 6.0,
                "sleep_hours": 7.0,
            },
            "delta_sleep_hours": -20.0,
            "delta_daily_screen_time_hours": 30.0,
            "delta_app_opens_per_day": -500,
        }
        response = client.post("/api/analytics/what-if", json=payload)
        assert response.status_code == 200
        sim = response.json()["simulated_profile"]
        assert sim["sleep_hours"] == 1.0
        assert sim["daily_screen_time_hours"] == 24.0
        assert sim["app_opens_per_day"] == 0

    def test_optimize_at_risk_profile(self, client: TestClient) -> None:
        """Verify habit optimizer generates achievable pathway with projected_prob < tau (AC-3.2)."""
        payload = {
            "baseline_profile": {
                "age": 25,
                "gender": "Male",
                "stress_level": "Medium",
                "academic_work_impact": "Yes",
                "daily_screen_time_hours": 8.0,
                "social_media_hours": 3.5,
                "gaming_hours": 1.5,
                "work_study_hours": 2.5,
                "weekend_screen_time": 9.5,
                "sleep_hours": 6.5,
                "notifications_per_day": 140,
                "app_opens_per_day": 100,
                "decision_threshold": 0.50,
            },
            "target_threshold": 0.50,
        }
        response = client.post("/api/analytics/what-if/optimize", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["baseline_probability"] >= 0.50
        assert data["target_threshold"] == 0.50
        assert data["target_screen_time_reduction_hours"] > 0.0
        assert data["target_sleep_increase_hours"] > 0.0
        assert data["projected_probability"] < 0.50
        assert data["projected_classification"] == "HEALTHY"
        assert data["achievable"] is True
        assert len(data["recommended_pathway"]) >= 3

    def test_optimize_already_healthy_profile(self, client: TestClient) -> None:
        """Verify habit optimizer returns 0 reductions for already healthy profile."""
        payload = {
            "baseline_profile": {
                "daily_screen_time_hours": 3.0,
                "social_media_hours": 0.5,
                "gaming_hours": 0.5,
                "sleep_hours": 8.0,
                "notifications_per_day": 30,
                "app_opens_per_day": 25,
                "decision_threshold": 0.50,
            },
            "target_threshold": 0.50,
        }
        response = client.post("/api/analytics/what-if/optimize", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["target_screen_time_reduction_hours"] == 0.0
        assert data["target_sleep_increase_hours"] == 0.0
        assert data["target_app_opens_reduction"] == 0
        assert data["target_notifications_reduction"] == 0
        assert data["projected_probability"] == data["baseline_probability"]
        assert data["achievable"] is True

    @pytest.mark.parametrize(
        "invalid_body",
        [
            {"baseline_profile": {"age": 12}},  # age < 18
            {"baseline_profile": {"daily_screen_time_hours": 26.0}},  # screen > 24
            {"delta_app_opens_per_day": "bad_type"},
        ],
    )
    def test_whatif_validation_errors_return_422(
        self, client: TestClient, invalid_body: dict[str, Any]
    ) -> None:
        """Verify invalid what-if payload returns HTTP 422 Unprocessable Entity."""
        response = client.post("/api/analytics/what-if", json=invalid_body)
        assert response.status_code == 422

    def test_optimize_validation_errors_return_422(self, client: TestClient) -> None:
        """Verify invalid target threshold returns HTTP 422 Unprocessable Entity."""
        response = client.post(
            "/api/analytics/what-if/optimize",
            json={"target_threshold": 2.0},
        )
        assert response.status_code == 422


# ==============================================================================
# 5. Batch CSV Prediction & Export Endpoints (AC-3.3, AC-3.4, AC-3.5, AC-3.6)
# ==============================================================================


class TestBatchEndpoints:
    """Integration tests for batch CSV upload and export endpoints."""

    @staticmethod
    def _create_sample_csv(n_rows: int = 50) -> bytes:
        """Generate valid synthetic CSV bytes for batch testing."""
        rows = [
            "participant_id,age,gender,stress_level,academic_work_impact,"
            "daily_screen_time_hours,social_media_hours,gaming_hours,"
            "work_study_hours,weekend_screen_time,sleep_hours,"
            "notifications_per_day,app_opens_per_day"
        ]
        for i in range(n_rows):
            is_high = i % 2 == 0
            rows.append(
                f"P-{1000 + i},{20 + (i % 12)},{'Female' if i % 2 == 0 else 'Male'},"
                f"{'High' if is_high else 'Low'},{'Yes' if is_high else 'No'},"
                f"{9.0 if is_high else 4.0},{3.5 if is_high else 1.5},{1.5 if is_high else 0.5},"
                f"{2.5},{11.0 if is_high else 5.5},{5.5 if is_high else 7.5},"
                f"{190 if is_high else 70},{130 if is_high else 50}"
            )
        return "\n".join(rows).encode("utf-8")

    def test_batch_lifecycle_upload_and_export(self, client: TestClient) -> None:
        """Verify full lifecycle: upload 50-row CSV, inspect response, download export (AC-3.3, AC-3.6)."""
        csv_bytes = self._create_sample_csv(50)

        # 1. Upload
        upload_resp = client.post(
            "/api/predict/batch",
            files={"file": ("cohort_test.csv", csv_bytes, "text/csv")},
        )
        assert upload_resp.status_code == 200
        data = upload_resp.json()

        assert data["total_records"] == 50
        assert data["processed_records"] == 50
        assert 0 <= data["addiction_count"] <= 50
        assert 0.0 <= data["addiction_prevalence_pct"] <= 100.0
        assert bool(data["download_token"])
        assert len(data["sample_records"]) == 20
        assert len(data["preview_rows"]) == 20

        # 2. Export
        token = data["download_token"]
        export_resp = client.get(f"/api/predict/batch/export?token={token}")
        assert export_resp.status_code == 200
        assert export_resp.headers["content-type"].startswith("text/csv")
        assert 'attachment; filename="diagnostic_report.csv"' in export_resp.headers["content-disposition"]

        # 3. Validate export contents
        csv_text = export_resp.content.decode("utf-8")
        assert "participant_id" in csv_text
        assert "predicted_probability" in csv_text
        assert "classification" in csv_text
        assert "screen_to_sleep_ratio" in csv_text
        assert "primary_intervention" in csv_text

    def test_batch_missing_required_column_returns_422(self, client: TestClient) -> None:
        """Verify upload missing daily_screen_time_hours returns HTTP 422 with header details (AC-3.4)."""
        header = "age,gender,stress_level,academic_work_impact,social_media_hours,gaming_hours,work_study_hours,weekend_screen_time,sleep_hours,notifications_per_day,app_opens_per_day\n"
        row = "25,Male,Medium,Yes,3.5,1.2,2.5,9.5,6.8,140,100\n"
        content = (header + row).encode("utf-8")

        response = client.post(
            "/api/predict/batch",
            files={"file": ("missing_screen.csv", content, "text/csv")},
        )
        assert response.status_code == 422
        assert "daily_screen_time_hours" in str(response.json()["detail"])

    def test_batch_exceeds_10000_rows_returns_413(self, client: TestClient) -> None:
        """Verify upload exceeding 10,000 rows returns HTTP 413 Payload Too Large (AC-3.5)."""
        header = "age,gender,stress_level,academic_work_impact,daily_screen_time_hours,social_media_hours,gaming_hours,work_study_hours,weekend_screen_time,sleep_hours,notifications_per_day,app_opens_per_day\n"
        row = "25,Male,Medium,Yes,7.5,3.5,1.2,2.5,9.5,6.8,140,100\n"
        content = (header + row * 10_001).encode("utf-8")

        response = client.post(
            "/api/predict/batch",
            files={"file": ("too_large.csv", content, "text/csv")},
        )
        assert response.status_code == 413
        assert response.json()["detail"] == "Batch upload exceeds maximum limit of 10,000 rows"

    def test_batch_export_invalid_token_returns_404(self, client: TestClient) -> None:
        """Verify invalid token on export returns HTTP 404."""
        response = client.get("/api/predict/batch/export?token=invalid-export-token")
        assert response.status_code == 404
        assert response.json()["detail"] == "Export token not found or expired"

    def test_batch_upload_mock_mode(self, client: TestClient) -> None:
        """Verify batch endpoint functions correctly when USE_MOCK_MODEL is active."""
        mock_settings = Settings(USE_MOCK_MODEL=True)
        app.dependency_overrides[get_settings] = lambda: mock_settings

        csv_bytes = self._create_sample_csv(20)
        response = client.post(
            "/api/predict/batch",
            files={"file": ("mock_batch.csv", csv_bytes, "text/csv")},
        )
        assert response.status_code == 200
        assert response.json()["total_records"] == 20


# ==============================================================================
# 13. What-If Scenario Simulator & Batch Diagnostics UI Views (M3-TASK-03)
# ==============================================================================


class TestWhatIfAndBatchViews:
    """Test suite validating UI template delivery, controls, and script wiring for M3-TASK-03."""

    def test_whatif_tab_and_counterfactual_levers_markup(self, client: TestClient) -> None:
        """Verify index.html delivers #tab-whatif with 3 interactive counterfactual sliders and buttons."""
        response = client.get("/")
        assert response.status_code == 200
        html = response.text

        # Tab container and panel
        assert 'id="tab-btn-whatif"' in html
        assert 'id="tab-whatif"' in html
        assert 'role="tabpanel"' in html

        # Counterfactual levers
        assert 'id="lever-recreation-reduce"' in html
        assert 'id="val-lever-recreation"' in html
        assert 'id="lever-sleep-extend"' in html
        assert 'id="val-lever-sleep"' in html
        assert 'id="lever-opens-batch"' in html
        assert 'id="val-lever-opens"' in html

        # Optimal target generator controls
        assert 'id="btn-generate-optimal"' in html
        assert 'id="optimal-target-card"' in html
        assert 'id="optimal-target-text"' in html
        assert 'id="btn-apply-optimal"' in html

    def test_whatif_comparison_cards_and_delta_badge_markup(self, client: TestClient) -> None:
        """Verify What-If comparison cards, reactive risk delta badge, and metrics table markup."""
        response = client.get("/")
        assert response.status_code == 200
        html = response.text

        # Comparison cards
        assert 'class="comparison-box"' in html
        assert 'id="sim-baseline-prob"' in html
        assert 'id="sim-baseline-badge"' in html
        assert 'id="sim-counterfactual-prob"' in html
        assert 'id="sim-counterfactual-badge"' in html

        # Reactive risk delta badge
        assert 'id="sim-delta-badge"' in html

        # Behavioral metric comparison readouts
        assert 'id="sim-base-ss"' in html
        assert 'id="sim-new-ss"' in html
        assert 'id="sim-base-rec"' in html
        assert 'id="sim-new-rec"' in html
        assert 'id="sim-base-screen"' in html
        assert 'id="sim-new-screen"' in html
        assert 'id="sim-base-sleep"' in html
        assert 'id="sim-new-sleep"' in html

        # Clinical status container
        assert 'id="sim-interventions-list"' in html

    def test_batch_tab_and_dropzone_markup(self, client: TestClient) -> None:
        """Verify index.html delivers #tab-batch with CSV dropzone, demo loader, and state containers."""
        response = client.get("/")
        assert response.status_code == 200
        html = response.text

        # Tab button and panel
        assert 'id="tab-btn-batch"' in html
        assert 'id="tab-batch"' in html

        # Dropzone and file input
        assert 'id="batch-dropzone"' in html
        assert 'id="batch-file-input"' in html
        assert 'id="btn-load-demo-batch"' in html

        # State containers
        assert 'id="batch-loading"' in html
        assert 'id="batch-error"' in html
        assert 'id="batch-error-msg"' in html

    def test_batch_summary_strip_and_preview_table_markup(self, client: TestClient) -> None:
        """Verify batch cohort summary cards, CSV export trigger, and preview table markup."""
        response = client.get("/")
        assert response.status_code == 200
        html = response.text

        # Summary strip
        assert 'class="batch-summary-strip"' in html
        assert 'id="batch-total-records"' in html
        assert 'id="batch-addiction-rate"' in html
        assert 'id="batch-high-risk-rate"' in html
        assert 'id="batch-mean-ss"' in html

        # Export CSV trigger button
        assert 'id="btn-export-csv"' in html

        # Diagnostic preview table
        assert 'id="batch-table"' in html
        assert 'id="batch-table-body"' in html

    def test_whatif_and_batch_client_script_wiring(self, client: TestClient) -> None:
        """Verify app.js binds What-If simulation and batch CSV workflow interactions."""
        app_js = client.get("/static/js/app.js").text

        # What-If interactions and API calls
        assert "runWhatIfSimulation" in app_js
        assert "generateOptimalTarget" in app_js
        assert "applyOptimalTargetToSliders" in app_js
        assert "/api/analytics/what-if" in app_js
        assert "/api/analytics/what-if/optimize" in app_js

        # Batch upload interactions and API calls
        assert "bindBatchControls" in app_js
        assert "handleBatchUpload" in app_js
        assert "renderBatchResults" in app_js
        assert "generateAndSubmitDemoBatch" in app_js
        assert "/api/predict/batch" in app_js
        assert "/api/predict/batch/export" in app_js

    def test_whatif_and_batch_css_styling_delivered(self, client: TestClient) -> None:
        """Verify style.css delivers styles for What-If and Batch views."""
        style_css = client.get("/static/css/style.css").text

        assert ".comparison-box" in style_css
        assert ".delta-badge" in style_css
        assert ".delta-reduction" in style_css
        assert ".delta-increase" in style_css
        assert ".optimal-target-card" in style_css
        assert ".dropzone-container" in style_css
        assert ".batch-summary-strip" in style_css
        assert ".batch-error-alert" in style_css


# ==============================================================================
# 14. Frontend Craft Polish Pass, Responsive Hardening, and A11y Compliance (M3-TASK-04)
# ==============================================================================


class TestFrontendCraftAndAccessibility:
    """Test suite validating frontend craft polish pass, responsive hardening, and WCAG AA accessibility compliance."""

    def test_editorial_design_tokens_and_color_contrast(self, client: TestClient) -> None:
        """Verify tokens.css delivers editorial warm parchment palette, font stacks, and status tokens with AA contrast."""
        tokens_css = client.get("/static/css/tokens.css").text

        # Canvas and surfaces
        assert "--bg: #F3F5F9" in tokens_css
        assert "--panel: #FFFFFF" in tokens_css
        assert "--panel-subtle: #F7F8FB" in tokens_css

        # Brand accent token (indigo) with high contrast (>= 4.5:1 on light panels)
        assert "--gold: #2B4A86" in tokens_css

        # Text ink tokens
        assert "--text-1: #16202F" in tokens_css
        assert "--text-2: #55617A" in tokens_css
        assert "--text-3: #525E73" in tokens_css

        # Status & severity tokens
        assert "--low-risk: #1E7E34" in tokens_css
        assert "--amber: #8A4C0E" in tokens_css
        assert "--danger: #9C3F2C" in tokens_css

        # Typography stacks
        assert "--font-serif: 'Source Serif 4'" in tokens_css
        assert "--font-sans: 'IBM Plex Sans'" in tokens_css

    def test_accessibility_focus_visible_and_reduced_motion(self, client: TestClient) -> None:
        """Verify style.css delivers :focus-visible outlines and respects prefers-reduced-motion."""
        style_css = client.get("/static/css/style.css").text

        # Keyboard accessibility focus rings
        assert ":focus-visible" in style_css
        assert "outline: 2px solid var(--gold)" in style_css

        # Reduced motion media query (WCAG 2.2 AA)
        assert "@media (prefers-reduced-motion: reduce)" in style_css
        assert "animation-duration: 0.01ms" in style_css
        assert "transition-duration: 0.01ms" in style_css

    def test_semantic_aria_roles_and_accessibility_attributes(self, client: TestClient) -> None:
        """Verify index.html delivers semantic ARIA roles across tabs, radiogroups, sliders, alerts, and live regions."""
        html = client.get("/").text

        # Navigation tablist & tabs
        assert 'role="tablist"' in html
        assert 'role="tab"' in html
        assert 'aria-selected="true"' in html
        assert 'aria-controls="tab-diagnostic"' in html
        assert 'aria-controls="tab-cohorts"' in html
        assert 'aria-controls="tab-whatif"' in html
        assert 'aria-controls="tab-batch"' in html

        # Segmented control radiogroups & radios
        assert 'role="radiogroup"' in html
        assert 'role="radio"' in html
        assert 'aria-checked="true"' in html
        assert 'aria-checked="false"' in html

        # Range slider ARIA attributes
        assert 'aria-valuemin="18"' in html
        assert 'aria-valuemax="35"' in html
        assert 'aria-valuenow="25"' in html
        assert 'aria-label="Age in years"' in html
        assert 'aria-label="Daily screen time in hours"' in html
        assert 'aria-label="Classification decision threshold tau"' in html

        # Physiological boundary warning alert
        assert 'id="boundary-warning"' in html
        assert 'role="alert"' in html

        # Notification region for system and offline alerts
        assert 'id="toast-container"' in html
        assert 'role="region"' in html
        assert 'aria-label="Notifications"' in html
        assert 'aria-live="polite"' in html

    def test_responsive_mobile_hardening_and_touch_targets(self, client: TestClient) -> None:
        """Verify style.css enforces fluid 1-column mobile stacks and accessible >= 44px touch targets."""
        style_css = client.get("/static/css/style.css").text

        # Mobile media query breakpoint
        assert "@media (max-width: 640px)" in style_css

        # Accessible touch target heights (>= 44px)
        assert "min-height: 44px;" in style_css
        assert "height: 44px; /* 44px hit area" in style_css  # 44px tap zone for range sliders

        # Horizontal scrolling wrappers to prevent layout blowout on narrow viewports
        assert ".heatmap-wrapper" in style_css
        assert ".benchmark-table-wrapper" in style_css
        assert ".batch-table-wrapper" in style_css

    def test_subgauge_clearance_and_authoritative_status_pill(self, client: TestClient) -> None:
        """Verify ADR-0008 sub-gauge clearance and ADR-0006 authoritative status pill."""
        style_css = client.get("/static/css/style.css").text
        html = client.get("/").text

        # ADR-0008: positive top margin clearance for gauge caption (no negative margin)
        assert ".gauge-caption" in style_css
        assert "margin-top: 1.25rem" in style_css or "margin-top:" in style_css

        # ADR-0006: single authoritative status pill
        assert 'id="status-pill"' in html
        assert 'id="badge-status"' in html

        # Typo correction safeguard (PAR-6)
        assert "ADDICITON" not in html
        app_js = client.get("/static/js/app.js").text
        assert "ADDICITON" not in app_js
        assert "ADDICTION DETECTED" in app_js
        assert "Addiction" in html

    def test_tabular_numbers_typography_applied(self, client: TestClient) -> None:
        """Verify tabular numeric typography is enforced for figures and metrics."""
        style_css = client.get("/static/css/style.css").text
        html = client.get("/").text

        assert "tabular-nums" in style_css
        assert "font-variant-numeric: tabular-nums;" in style_css or "tabular-nums" in html

    def test_toast_notification_manager_and_offline_handling(self, client: TestClient) -> None:
        """Verify app.js implements toast notifications and graceful offline fallback handling."""
        app_js = client.get("/static/js/app.js").text

        assert "toastManager" in app_js
        assert "showToast" in app_js
        assert "window.addEventListener('offline'" in app_js
        assert "window.addEventListener('online'" in app_js

    def test_invalid_input_and_error_handling(self, client: TestClient) -> None:
        """Verify invalid requests return expected HTTP error status codes."""
        # Non-existent static resource returns 404
        missing_asset_resp = client.get("/static/css/nonexistent_stylesheet.css")
        assert missing_asset_resp.status_code == 404

        # Malformed predict request (out of bounds age) returns 422
        invalid_predict_resp = client.post(
            "/api/predict",
            json={"age": 50, "gender": "Male", "stress_level": "Medium", "daily_screen_time_hours": 8.0},
        )
        assert invalid_predict_resp.status_code == 422

        # Invalid cohort dimension returns 422
        invalid_cohort_resp = client.get("/api/analytics/cohorts?dimension=unsupported_dimension")
        assert invalid_cohort_resp.status_code in (400, 422)









