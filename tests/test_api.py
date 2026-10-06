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
        assert data["prediction"] in (0, 1)
        assert data["classification"] in ("ADDICTION DETECTED", "HEALTHY")
        assert data["severity"] in ("HIGH", "MODERATE", "LOW", "High", "Moderate", "Healthy")
        assert data["status_label"] in (
            "Elevated Risk Tier",
            "Compensatory Usage Pattern",
            "Balanced Habit Profile",
        )
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

        # High threshold (0.95) guarantees healthy classification
        high_tau_payload = {**sample_valid_profile, "decision_threshold": 0.95}
        res_high = client.post("/api/predict", json=high_tau_payload)
        assert res_high.status_code == 200
        data_high = res_high.json()
        assert data_high["prediction"] == 0
        assert data_high["classification"] == "HEALTHY"

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
        assert data["status_label"] in (
            "Elevated Risk Tier",
            "Compensatory Usage Pattern",
            "Balanced Habit Profile",
        )


# ==============================================================================
# 4. Latency Benchmark Performance (<20ms SLA)
# ==============================================================================


class TestApiLatencyPerformance:
    """Latency benchmark suite validating AC-1.2 (<20ms p95 single-row inference SLA)."""

    def test_predict_latency_p95_under_20ms(
        self, client: TestClient, sample_valid_profile: dict[str, Any]
    ) -> None:
        """Verify 50 sequential requests to /api/predict achieve p95 latency < 20ms (AC-1.2)."""
        # Warmup call
        client.post("/api/predict", json=sample_valid_profile)

        latencies_ms: list[float] = []
        for _ in range(50):
            t0 = time.perf_counter()
            response = client.post("/api/predict", json=sample_valid_profile)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            assert response.status_code == 200
            latencies_ms.append(elapsed_ms)

        p95_latency = float(np.percentile(latencies_ms, 95))
        mean_latency = float(np.mean(latencies_ms))
        assert (
            p95_latency < 20.0
        ), f"p95 latency {p95_latency:.2f}ms exceeded 20ms SLA (mean: {mean_latency:.2f}ms)"


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
        """Verify GET / and GET /app return 200 OK."""
        res_root = client.get("/")
        assert res_root.status_code == 200

        res_app = client.get("/app")
        assert res_app.status_code == 200
