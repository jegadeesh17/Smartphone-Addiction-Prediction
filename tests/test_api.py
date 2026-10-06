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
    app.dependency_overrides.clear()
    get_settings.cache_clear()
    yield
    reset_inference_engine()
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



