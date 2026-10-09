"""Unit and integration tests for counterfactual What-If simulation and habit optimizer.

Validates:
- Counterfactual delta evaluation returns simulated probability < baseline when habits improve (AC-3.1).
- Worsening habits increases risk delta > 0.
- Behavioral ratio updates and physiological boundary clamping.
- Automated optimal habit target recommendation returns achievable pathway with projected_prob < tau (AC-3.2).
- Already-healthy baseline profile returns 0 habit reductions.
- Target threshold customization and risk tier constraints.
- Mock inference adapter compatibility (ADR-0005, ADR-0007).
- Request validation boundaries returning HTTP 422 Unprocessable Entity.
"""

from __future__ import annotations

from typing import Any
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
# 1. Counterfactual What-If Simulation Tests (POST /api/analytics/what-if, AC-3.1)
# ==============================================================================


class TestWhatIfEndpoint:
    """Test suite for POST /api/analytics/what-if counterfactual scenario evaluation."""

    def test_whatif_habit_improvement_reduces_risk(self, client: TestClient) -> None:
        """Verify habit improvements reduce addiction probability below baseline (AC-3.1).

        Given a baseline profile with addiction probability P_base ~ 0.78:
        delta_social_media_hours: -2.0
        delta_sleep_hours: +1.5
        delta_app_opens_per_day: -30
        """
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

        # Contract fields
        assert "baseline_probability" in data
        assert "simulated_probability" in data
        assert "risk_delta" in data
        assert "baseline_classification" in data
        assert "simulated_classification" in data
        assert "baseline_status_label" in data
        assert "simulated_status_label" in data
        assert "baseline_ratios" in data
        assert "simulated_ratios" in data
        assert "simulated_profile" in data
        assert "interventions" in data

        # Numerical expectations per SPEC AC-3.1
        # Baseline probability around 0.78 +- 0.02
        assert 0.76 <= data["baseline_probability"] <= 0.80
        # Simulated probability strictly lower than baseline
        assert data["simulated_probability"] < data["baseline_probability"]
        # Risk delta is negative (simulated - baseline)
        assert data["risk_delta"] < 0.0
        assert data["risk_delta"] == pytest.approx(
            data["simulated_probability"] - data["baseline_probability"], abs=1e-4
        )

        # Classification verification
        assert data["baseline_classification"] == "ADDICTION DETECTED"
        assert data["simulated_classification"] in ("HEALTHY", "ADDICTION DETECTED")

        # Ratio verification
        base_ratios = data["baseline_ratios"]
        sim_ratios = data["simulated_ratios"]
        assert sim_ratios["screen_to_sleep_ratio"] < base_ratios["screen_to_sleep_ratio"]
        assert sim_ratios["recreational_share"] < base_ratios["recreational_share"]

        # Simulated profile updates
        sim_prof = data["simulated_profile"]
        assert sim_prof["social_media_hours"] == pytest.approx(1.5, abs=0.01)
        assert sim_prof["sleep_hours"] == pytest.approx(8.0, abs=0.01)
        assert sim_prof["app_opens_per_day"] == 70

    def test_whatif_worsening_habits_increases_risk(self, client: TestClient) -> None:
        """Verify deteriorating digital habits increase risk delta (risk_delta > 0)."""
        payload = {
            "baseline_profile": {
                "age": 22,
                "gender": "Female",
                "stress_level": "Low",
                "academic_work_impact": "No",
                "daily_screen_time_hours": 4.5,
                "social_media_hours": 1.5,
                "gaming_hours": 0.5,
                "work_study_hours": 2.5,
                "weekend_screen_time": 5.0,
                "sleep_hours": 8.0,
                "notifications_per_day": 50,
                "app_opens_per_day": 40,
                "decision_threshold": 0.50,
            },
            "delta_daily_screen_time_hours": 4.0,
            "delta_social_media_hours": 2.5,
            "delta_sleep_hours": -2.5,
            "delta_notifications_per_day": 80,
            "delta_app_opens_per_day": 50,
        }

        response = client.post("/api/analytics/what-if", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["simulated_probability"] > data["baseline_probability"]
        assert data["risk_delta"] > 0.0
        assert data["risk_delta"] == pytest.approx(
            data["simulated_probability"] - data["baseline_probability"], abs=1e-4
        )

        sim_prof = data["simulated_profile"]
        assert sim_prof["daily_screen_time_hours"] == pytest.approx(8.5, abs=0.01)
        assert sim_prof["sleep_hours"] == pytest.approx(5.5, abs=0.01)
        assert sim_prof["app_opens_per_day"] == 90
        assert sim_prof["notifications_per_day"] == 130

    def test_whatif_boundary_clamping(self, client: TestClient) -> None:
        """Verify extreme lifestyle adjustments are clamped to physiological limits."""
        payload = {
            "baseline_profile": {
                "daily_screen_time_hours": 6.0,
                "sleep_hours": 7.0,
                "app_opens_per_day": 80,
                "notifications_per_day": 100,
            },
            # Push sleep far below minimum (1.0h)
            "delta_sleep_hours": -15.0,
            # Push screen time above 24h limit
            "delta_daily_screen_time_hours": 30.0,
            # Push interactions below zero
            "delta_app_opens_per_day": -200,
            "delta_notifications_per_day": -300,
        }

        response = client.post("/api/analytics/what-if", json=payload)
        assert response.status_code == 200
        data = response.json()
        sim_prof = data["simulated_profile"]

        # Bound checks
        assert sim_prof["sleep_hours"] == 1.0  # Min sleep clamped to 1.0h
        assert sim_prof["daily_screen_time_hours"] == 24.0  # Max screen clamped to 24.0h
        assert sim_prof["app_opens_per_day"] == 0  # Min opens >= 0
        assert sim_prof["notifications_per_day"] == 0  # Min notifs >= 0

    def test_whatif_recreational_delta_screen_time_sync(self, client: TestClient) -> None:
        """Verify changing recreational delta automatically adjusts daily screen time when delta_daily is 0."""
        payload = {
            "baseline_profile": {
                "daily_screen_time_hours": 7.0,
                "social_media_hours": 3.0,
                "gaming_hours": 1.0,
                "sleep_hours": 7.0,
            },
            "delta_social_media_hours": -1.5,
            "delta_gaming_hours": -0.5,
            "delta_daily_screen_time_hours": 0.0,  # Unspecified delta
        }

        response = client.post("/api/analytics/what-if", json=payload)
        assert response.status_code == 200
        sim_prof = response.json()["simulated_profile"]

        # Net recreation reduction is -2.0h -> daily screen time reduces to 5.0h
        assert sim_prof["social_media_hours"] == pytest.approx(1.5, abs=0.01)
        assert sim_prof["gaming_hours"] == pytest.approx(0.5, abs=0.01)
        assert sim_prof["daily_screen_time_hours"] == pytest.approx(5.0, abs=0.01)

    def test_whatif_ratio_aliases(self, client: TestClient) -> None:
        """Verify baseline and simulated ratios contain both canonical and SPEC alias keys."""
        payload = {
            "baseline_profile": {
                "daily_screen_time_hours": 8.0,
                "sleep_hours": 6.0,
            },
            "delta_sleep_hours": 2.0,
        }

        response = client.post("/api/analytics/what-if", json=payload)
        assert response.status_code == 200
        data = response.json()

        for ratio_dict in (data["baseline_ratios"], data["simulated_ratios"]):
            assert "screen_to_sleep_ratio" in ratio_dict
            assert "screen_to_sleep" in ratio_dict
            assert "recreational_share" in ratio_dict
            assert "recreational_to_screen" in ratio_dict
            assert "avg_unlock_minutes" in ratio_dict
            assert "weekend_surge_hours" in ratio_dict
            assert "weekend_diff" in ratio_dict
            assert "total_accounted_hours" in ratio_dict

    def test_whatif_mock_adapter_support(self, client: TestClient) -> None:
        """Verify /api/analytics/what-if operates correctly under MockInferenceEngine."""
        mock_settings = Settings(USE_MOCK_MODEL=True)
        app.dependency_overrides[get_settings] = lambda: mock_settings

        payload = {
            "baseline_profile": {
                "daily_screen_time_hours": 8.5,
                "sleep_hours": 6.0,
            },
            "delta_daily_screen_time_hours": -2.0,
            "delta_sleep_hours": 1.5,
        }

        response = client.post("/api/analytics/what-if", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["simulated_probability"] < data["baseline_probability"]
        assert data["risk_delta"] < 0.0


_REAL_MODEL_SETTINGS = Settings(USE_MOCK_MODEL=False)


@pytest.mark.skipif(
    not _REAL_MODEL_SETTINGS.MODEL_PATH.exists(),
    reason="trained LightGBM checkpoint not available",
)
class TestWhatIfRealModel:
    """What-If direction checks against the trained LightGBM model (no mock)."""

    # Accounted hours (social + gaming + work/study) = 6.0h, consistent with training data
    BASELINE = {
        "age": 21,
        "gender": "Male",
        "stress_level": "Medium",
        "academic_work_impact": "No",
        "daily_screen_time_hours": 9.0,
        "social_media_hours": 2.0,
        "gaming_hours": 1.0,
        "work_study_hours": 3.0,
        "weekend_screen_time": 10.0,
        "sleep_hours": 7.0,
        "notifications_per_day": 100,
        "app_opens_per_day": 80,
    }

    @pytest.mark.parametrize("delta", [-0.5, -1.0, -2.0, -3.0, -5.0])
    def test_reducing_screen_time_never_raises_risk(self, client: TestClient, delta: float) -> None:
        """Cutting screen time must not increase risk, even past the accounted-hours floor."""
        app.dependency_overrides[get_settings] = lambda: _REAL_MODEL_SETTINGS

        response = client.post(
            "/api/analytics/what-if",
            json={"baseline_profile": self.BASELINE, "delta_daily_screen_time_hours": delta},
        )
        assert response.status_code == 200
        data = response.json()

        assert data["risk_delta"] <= 0.0
        floor = self.BASELINE["social_media_hours"] + self.BASELINE["gaming_hours"] + self.BASELINE["work_study_hours"]
        assert data["simulated_profile"]["daily_screen_time_hours"] >= floor


# ==============================================================================
# 2. Habit Optimizer Tests (POST /api/analytics/what-if/optimize, AC-3.2)
# ==============================================================================


class TestHabitOptimizerEndpoint:
    """Test suite for POST /api/analytics/what-if/optimize automated habit target pathway."""

    def test_optimize_at_risk_profile_returns_achievable_pathway(
        self, client: TestClient
    ) -> None:
        """Verify automated optimal habit target returns achievable pathway with projected_prob < tau (AC-3.2).

        Given an at-risk profile where P_base >= 0.68 and tau = 0.50:
        Returns recommended pathway detailing:
        - Target daily screen time reduction (in hours)
        - Target sleep duration increase (in hours)
        - Projected new probability < 0.50
        """
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

        # Contract fields
        assert "baseline_probability" in data
        assert "target_threshold" in data
        assert "target_screen_time_reduction_hours" in data
        assert "target_sleep_increase_hours" in data
        assert "target_app_opens_reduction" in data
        assert "target_notifications_reduction" in data
        assert "projected_probability" in data
        assert "projected_classification" in data
        assert "projected_status_label" in data
        assert "achievable" in data
        assert "recommended_pathway" in data
        assert "optimized_profile" in data

        # AC-3.2 requirements
        assert data["baseline_probability"] >= 0.50
        assert data["target_threshold"] == 0.50
        assert data["target_screen_time_reduction_hours"] > 0.0
        assert data["target_sleep_increase_hours"] > 0.0
        assert data["projected_probability"] < 0.50
        assert data["projected_classification"] == "HEALTHY"
        assert data["achievable"] is True

        # Pathway narrative verification
        pathway = data["recommended_pathway"]
        assert len(pathway) >= 3
        joined_pathway = " ".join(pathway)
        assert "Daily Screen Time" in joined_pathway or "screen time" in joined_pathway.lower()
        assert "Sleep Restoration" in joined_pathway or "sleep" in joined_pathway.lower()
        assert "Projected Outcome" in joined_pathway

        # Optimized profile properties
        opt_prof = data["optimized_profile"]
        assert opt_prof["daily_screen_time_hours"] < payload["baseline_profile"]["daily_screen_time_hours"]
        assert opt_prof["sleep_hours"] > payload["baseline_profile"]["sleep_hours"]

    def test_optimize_already_healthy_profile_returns_zero_reductions(
        self, client: TestClient
    ) -> None:
        """Verify an already healthy profile returns zero habit reductions."""
        payload = {
            "baseline_profile": {
                "age": 24,
                "gender": "Female",
                "stress_level": "Low",
                "academic_work_impact": "No",
                "daily_screen_time_hours": 3.0,
                "social_media_hours": 0.5,
                "gaming_hours": 0.5,
                "work_study_hours": 2.0,
                "weekend_screen_time": 3.5,
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

        assert data["baseline_probability"] < 0.50
        assert data["target_screen_time_reduction_hours"] == 0.0
        assert data["target_sleep_increase_hours"] == 0.0
        assert data["target_app_opens_reduction"] == 0
        assert data["target_notifications_reduction"] == 0
        assert data["projected_probability"] == data["baseline_probability"]
        assert data["projected_classification"] == "HEALTHY"
        assert data["achievable"] is True
        assert any("already within healthy bounds" in step.lower() for step in data["recommended_pathway"])

    def test_optimize_custom_target_threshold(self, client: TestClient) -> None:
        """Verify optimizer respects custom stricter target threshold tau."""
        payload = {
            "baseline_profile": {
                "daily_screen_time_hours": 7.5,
                "sleep_hours": 6.8,
            },
            "target_threshold": 0.35,  # Stricter than default 0.50
        }

        response = client.post("/api/analytics/what-if/optimize", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["target_threshold"] == 0.35
        assert data["projected_probability"] < 0.35
        assert data["achievable"] is True

    def test_optimize_target_risk_tier_low(self, client: TestClient) -> None:
        """Verify target_risk_tier='LOW' steers projected probability to LOW tier (<0.40)."""
        payload = {
            "baseline_profile": {
                "daily_screen_time_hours": 8.0,
                "sleep_hours": 6.5,
                "decision_threshold": 0.50,
            },
            "target_risk_tier": "LOW",
        }

        response = client.post("/api/analytics/what-if/optimize", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["projected_probability"] < 0.40
        assert data["achievable"] is True

    def test_optimize_mock_adapter_support(self, client: TestClient) -> None:
        """Verify POST /api/analytics/what-if/optimize operates correctly under MockInferenceEngine."""
        mock_settings = Settings(USE_MOCK_MODEL=True)
        app.dependency_overrides[get_settings] = lambda: mock_settings

        payload = {
            "baseline_profile": {
                "daily_screen_time_hours": 9.2,
                "sleep_hours": 6.5,
                "decision_threshold": 0.50,
            },
            "target_threshold": 0.50,
        }

        response = client.post("/api/analytics/what-if/optimize", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert 0.65 <= data["baseline_probability"] <= 0.70
        assert data["projected_probability"] < 0.50
        assert data["target_screen_time_reduction_hours"] > 0.0
        assert data["target_sleep_increase_hours"] > 0.0
        assert data["achievable"] is True


# ==============================================================================
# 3. Payload Validation & Boundary Errors (HTTP 422)
# ==============================================================================


class TestValidationErrors:
    """Test suite for boundary violations and malformed payload error handling."""

    @pytest.mark.parametrize(
        "invalid_field, invalid_value",
        [
            ("age", 15),  # Age < 18
            ("age", 45),  # Age > 35
            ("daily_screen_time_hours", -2.0),  # Screen < 0
            ("daily_screen_time_hours", 26.0),  # Screen > 24
            ("sleep_hours", 0.5),  # Sleep < 1.0
            ("sleep_hours", 20.0),  # Sleep > 18.0
            ("app_opens_per_day", -10),  # Opens < 0
            ("notifications_per_day", -5),  # Notifs < 0
        ],
    )
    def test_whatif_invalid_profile_bounds_returns_422(
        self, client: TestClient, invalid_field: str, invalid_value: Any
    ) -> None:
        """Verify bounds violations in baseline_profile return HTTP 422 Unprocessable Entity."""
        payload = {
            "baseline_profile": {
                invalid_field: invalid_value,
            },
            "delta_sleep_hours": 1.0,
        }
        response = client.post("/api/analytics/what-if", json=payload)
        assert response.status_code == 422

    def test_whatif_malformed_delta_type_returns_422(self, client: TestClient) -> None:
        """Verify non-numeric delta parameter types return HTTP 422."""
        payload = {
            "delta_daily_screen_time_hours": "non_numeric_string",
        }
        response = client.post("/api/analytics/what-if", json=payload)
        assert response.status_code == 422

    @pytest.mark.parametrize(
        "invalid_threshold",
        [-0.1, 0.01, 0.99, 1.5],
    )
    def test_optimize_invalid_target_threshold_returns_422(
        self, client: TestClient, invalid_threshold: float
    ) -> None:
        """Verify target_threshold out of [0.05, 0.95] returns HTTP 422."""
        payload = {
            "target_threshold": invalid_threshold,
        }
        response = client.post("/api/analytics/what-if/optimize", json=payload)
        assert response.status_code == 422
