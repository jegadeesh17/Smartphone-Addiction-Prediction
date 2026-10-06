"""Unit tests for Pydantic v2 domain schemas in src/schemas.py."""

import pytest
from pydantic import ValidationError

from src.schemas import (
    BehavioralMetrics,
    BehavioralProfileInput,
    HealthResponse,
    PredictionResponse,
)


class TestBehavioralProfileInput:
    """Validation test suite for BehavioralProfileInput schema."""

    def test_default_instantiation(self) -> None:
        """Verify BehavioralProfileInput defaults populate with population medians."""
        profile = BehavioralProfileInput()
        assert profile.age == 25
        assert profile.gender == "Male"
        assert profile.stress_level == "Medium"
        assert profile.academic_work_impact == "Yes"
        assert profile.daily_screen_time_hours == 7.5
        assert profile.social_media_hours == 3.5
        assert profile.gaming_hours == 1.2
        assert profile.work_study_hours == 2.5
        assert profile.weekend_screen_time == 9.5
        assert profile.sleep_hours == 6.8
        assert profile.notifications_per_day == 140
        assert profile.app_opens_per_day == 100
        assert profile.threshold == 0.50
        assert profile.decision_threshold == 0.50
        assert profile.exceeds_daily_budget is False

    def test_explicit_valid_profile(self) -> None:
        """Verify valid custom attributes within bounds pass validation."""
        profile = BehavioralProfileInput(
            age=22,
            gender="Female",
            stress_level="High",
            academic_work_impact="Yes",
            daily_screen_time_hours=8.0,
            social_media_hours=4.0,
            gaming_hours=1.0,
            work_study_hours=3.0,
            weekend_screen_time=10.0,
            sleep_hours=7.0,
            notifications_per_day=150,
            app_opens_per_day=120,
            threshold=0.60,
        )
        assert profile.age == 22
        assert profile.gender == "Female"
        assert profile.threshold == 0.60
        assert profile.decision_threshold == 0.60
        assert profile.total_accounted_hours == 18.0
        assert profile.exceeds_daily_budget is False

    def test_alias_decision_threshold_input(self) -> None:
        """Verify decision_threshold alias is accepted in constructor and dict validation."""
        p1 = BehavioralProfileInput(decision_threshold=0.65)
        assert p1.threshold == 0.65
        assert p1.decision_threshold == 0.65

        p2 = BehavioralProfileInput.model_validate({"decision_threshold": 0.45})
        assert p2.threshold == 0.45

    def test_alias_time_inputs(self) -> None:
        """Verify alias names for screen time and sleep duration."""
        p = BehavioralProfileInput.model_validate({
            "daily_screen_time": 6.5,
            "weekend_screen_time_hours": 8.5,
            "sleep_duration_hours": 7.5,
        })
        assert p.daily_screen_time_hours == 6.5
        assert p.weekend_screen_time == 8.5
        assert p.sleep_hours == 7.5

    @pytest.mark.parametrize("invalid_age", [15, 17, 36, 50, -1])
    def test_age_out_of_bounds(self, invalid_age: int) -> None:
        """Verify age outside [18, 35] raises ValidationError (SPEC AC-1.6)."""
        with pytest.raises(ValidationError) as exc_info:
            BehavioralProfileInput(age=invalid_age)
        assert "age" in str(exc_info.value)

    @pytest.mark.parametrize("invalid_gender", ["Unknown", "OtherGender", ""])
    def test_gender_invalid_literal(self, invalid_gender: str) -> None:
        """Verify unsupported gender literal raises ValidationError."""
        with pytest.raises(ValidationError):
            BehavioralProfileInput(gender=invalid_gender)  # type: ignore[arg-type]

    @pytest.mark.parametrize("invalid_stress", ["Severe", "None", "123"])
    def test_stress_level_invalid(self, invalid_stress: str) -> None:
        """Verify unsupported stress level literal raises ValidationError."""
        with pytest.raises(ValidationError):
            BehavioralProfileInput(stress_level=invalid_stress)  # type: ignore[arg-type]

    @pytest.mark.parametrize("invalid_impact", ["Maybe", "true", "False"])
    def test_academic_impact_invalid(self, invalid_impact: str) -> None:
        """Verify unsupported academic work impact literal raises ValidationError."""
        with pytest.raises(ValidationError):
            BehavioralProfileInput(academic_work_impact=invalid_impact)  # type: ignore[arg-type]

    @pytest.mark.parametrize("invalid_sleep", [0.5, 0.0, -1.0, 18.5, 24.0])
    def test_sleep_hours_out_of_bounds(self, invalid_sleep: float) -> None:
        """Verify sleep duration outside [1.0, 18.0] raises ValidationError (SPEC AC-1.6)."""
        with pytest.raises(ValidationError) as exc_info:
            BehavioralProfileInput(sleep_hours=invalid_sleep)
        assert "sleep_hours" in str(exc_info.value)

    @pytest.mark.parametrize("invalid_screen", [-1.0, -0.1, 24.5, 30.0])
    def test_daily_screen_time_out_of_bounds(self, invalid_screen: float) -> None:
        """Verify daily screen time outside [0.0, 24.0] raises ValidationError (SPEC AC-1.6)."""
        with pytest.raises(ValidationError) as exc_info:
            BehavioralProfileInput(daily_screen_time_hours=invalid_screen)
        assert "daily_screen_time_hours" in str(exc_info.value)

    @pytest.mark.parametrize("invalid_notifs", [-5, -1, 501, 1000])
    def test_notifications_out_of_bounds(self, invalid_notifs: int) -> None:
        """Verify notifications per day outside [0, 500] raises ValidationError (SPEC AC-1.6)."""
        with pytest.raises(ValidationError) as exc_info:
            BehavioralProfileInput(notifications_per_day=invalid_notifs)
        assert "notifications_per_day" in str(exc_info.value)

    @pytest.mark.parametrize("invalid_opens", [-1, 501, 600])
    def test_app_opens_out_of_bounds(self, invalid_opens: int) -> None:
        """Verify app opens per day outside [0, 500] raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            BehavioralProfileInput(app_opens_per_day=invalid_opens)
        assert "app_opens_per_day" in str(exc_info.value)

    @pytest.mark.parametrize("invalid_threshold", [0.01, 0.04, 0.96, 1.0, -0.1])
    def test_threshold_out_of_bounds(self, invalid_threshold: float) -> None:
        """Verify decision threshold outside [0.05, 0.95] raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            BehavioralProfileInput(threshold=invalid_threshold)
        assert "threshold" in str(exc_info.value)

    def test_physiological_budget_validator_exceeded(self) -> None:
        """Verify root validator sets exceeds_daily_budget flag when total hours > 24h (SPEC AC-1.7)."""
        profile = BehavioralProfileInput(
            daily_screen_time_hours=12.0,
            work_study_hours=8.0,
            sleep_hours=6.0,
        )
        assert profile.total_accounted_hours == 26.0
        assert profile.exceeds_daily_budget is True

    def test_physiological_budget_validator_within_limit(self) -> None:
        """Verify exceeds_daily_budget remains False when accounted hours <= 24h."""
        profile = BehavioralProfileInput(
            daily_screen_time_hours=6.0,
            work_study_hours=6.0,
            sleep_hours=8.0,
        )
        assert profile.total_accounted_hours == 20.0
        assert profile.exceeds_daily_budget is False


class TestBehavioralMetrics:
    """Validation test suite for BehavioralMetrics schema and aliases."""

    def test_instantiation_canonical(self) -> None:
        """Verify BehavioralMetrics instantiates with canonical field names."""
        metrics = BehavioralMetrics(
            screen_to_sleep_ratio=1.3333,
            recreational_share=0.5000,
            avg_unlock_minutes=6.0000,
            weekend_surge_hours=3.0000,
            total_accounted_hours=16.5,
            exceeds_daily_budget=False,
        )
        assert metrics.screen_to_sleep_ratio == 1.3333
        assert metrics.screen_to_sleep == 1.3333
        assert metrics.recreational_share == 0.5000
        assert metrics.recreational_to_screen == 0.5000
        assert metrics.avg_unlock_minutes == 6.0000
        assert metrics.weekend_surge_hours == 3.0000
        assert metrics.weekend_diff == 3.0000
        assert metrics.total_accounted_hours == 16.5
        assert metrics.exceeds_daily_budget is False

    def test_instantiation_with_legacy_aliases(self) -> None:
        """Verify BehavioralMetrics accepts legacy / SPEC AC-1.4 aliases."""
        metrics = BehavioralMetrics.model_validate({
            "screen_to_sleep": 1.45,
            "recreational_to_screen": 0.65,
            "avg_unlock_minutes": 5.5,
            "weekend_diff": 2.8,
            "total_accounted_hours": 18.0,
            "exceeds_daily_budget": False,
        })
        assert metrics.screen_to_sleep_ratio == 1.45
        assert metrics.recreational_share == 0.65
        assert metrics.weekend_surge_hours == 2.8

    def test_serialization_contains_computed_aliases(self) -> None:
        """Verify model_dump includes both canonical fields and computed alias fields."""
        metrics = BehavioralMetrics(
            screen_to_sleep_ratio=1.2,
            recreational_share=0.4,
            avg_unlock_minutes=5.0,
            weekend_surge_hours=2.0,
            total_accounted_hours=17.0,
            exceeds_daily_budget=False,
        )
        data = metrics.model_dump()
        assert "screen_to_sleep_ratio" in data
        assert "screen_to_sleep" in data
        assert "recreational_share" in data
        assert "recreational_to_screen" in data
        assert "weekend_surge_hours" in data
        assert "weekend_diff" in data


class TestPredictionResponse:
    """Validation test suite for PredictionResponse schema."""

    @pytest.fixture
    def sample_metrics(self) -> BehavioralMetrics:
        return BehavioralMetrics(
            screen_to_sleep_ratio=1.3333,
            recreational_share=0.5000,
            avg_unlock_minutes=6.0000,
            weekend_surge_hours=3.0000,
            total_accounted_hours=16.8,
            exceeds_daily_budget=False,
        )

    def test_instantiation_spec_format(self, sample_metrics: BehavioralMetrics) -> None:
        """Verify instantiation using SPEC AC-1.1 attribute names."""
        res = PredictionResponse(
            probability=0.78,
            prediction=1,
            classification="ADDICTION DETECTED",
            severity="High",
            status_label="ADDICTION DETECTED • HIGH RISK",
            ratios=sample_metrics,
            interventions=["Sleep Protection: Enforce digital curfew"],
            latency_ms=1.24,
            decision_threshold=0.50,
        )
        assert res.probability == 0.78
        assert res.addiction_probability == 0.78
        assert res.prediction == 1
        assert res.is_addicted is True
        assert res.classification == "ADDICTION DETECTED"
        assert res.severity == "High"
        assert res.risk_tier == "HIGH"
        assert res.status_label == "ADDICTION DETECTED • HIGH RISK"
        assert res.ratios.screen_to_sleep_ratio == 1.3333
        assert res.metrics.screen_to_sleep == 1.3333
        assert len(res.interventions) == 1
        assert len(res.recommendations) == 1
        assert res.latency_ms == 1.24

    def test_instantiation_architecture_format(self, sample_metrics: BehavioralMetrics) -> None:
        """Verify instantiation using ARCHITECTURE.md / mock adapter attribute names."""
        res = PredictionResponse(
            addiction_probability=0.35,
            is_addicted=False,
            risk_tier="LOW",
            status_label="HEALTHY PATTERN • LOW RISK",
            decision_threshold=0.50,
            metrics=sample_metrics,
            recommendations=[],
            latency_ms=0.45,
        )
        assert res.probability == 0.35
        assert res.addiction_probability == 0.35
        assert res.prediction == 0
        assert res.is_addicted is False
        assert res.classification == "HEALTHY"
        assert res.severity == "LOW"
        assert res.risk_tier == "LOW"
        assert res.interventions == []
        assert res.recommendations == []

    def test_coercion_from_probability_and_threshold(self, sample_metrics: BehavioralMetrics) -> None:
        """Verify prediction and classification auto-derive when omitted."""
        res_addicted = PredictionResponse.model_validate({
            "probability": 0.65,
            "decision_threshold": 0.50,
            "severity": "Moderate",
            "status_label": "Compensatory Usage Pattern",
            "ratios": sample_metrics,
            "latency_ms": 2.0,
        })
        assert res_addicted.prediction == 1
        assert res_addicted.classification == "ADDICTION DETECTED"
        assert res_addicted.is_addicted is True

        res_healthy = PredictionResponse.model_validate({
            "probability": 0.40,
            "decision_threshold": 0.50,
            "severity": "Healthy",
            "status_label": "Balanced Habit Profile",
            "ratios": sample_metrics,
            "latency_ms": 2.0,
        })
        assert res_healthy.prediction == 0
        assert res_healthy.classification == "HEALTHY"
        assert res_healthy.is_addicted is False

    @pytest.mark.parametrize("invalid_prob", [-0.1, 1.05, 2.0])
    def test_probability_out_of_bounds(self, sample_metrics: BehavioralMetrics, invalid_prob: float) -> None:
        """Verify probability outside [0.0, 1.0] raises ValidationError."""
        with pytest.raises(ValidationError):
            PredictionResponse(
                probability=invalid_prob,
                prediction=1,
                classification="ADDICTION DETECTED",
                severity="High",
                status_label="ADDICTION DETECTED • HIGH RISK",
                ratios=sample_metrics,
                latency_ms=1.0,
            )

    def test_negative_latency_raises(self, sample_metrics: BehavioralMetrics) -> None:
        """Verify negative latency raises ValidationError."""
        with pytest.raises(ValidationError):
            PredictionResponse(
                probability=0.5,
                prediction=1,
                classification="ADDICTION DETECTED",
                severity="High",
                status_label="ADDICTION DETECTED • HIGH RISK",
                ratios=sample_metrics,
                latency_ms=-0.5,
            )


class TestHealthResponse:
    """Validation test suite for HealthResponse schema."""

    def test_health_response_defaults(self) -> None:
        """Verify default HealthResponse attributes match SPEC AC-1.8 and ARCHITECTURE."""
        health = HealthResponse()
        assert health.status == "healthy"
        assert health.model_loaded is True
        assert health.version == "1.0.0"
        assert health.model_family == "LightGBM"
        assert health.fold == 1
        assert health.priors_loaded is True
        assert health.cohort_priors_cached is True
        assert health.cohort_cache_loaded is True
        assert health.mock_mode is False
        assert health.uptime_seconds == 0.0

    def test_health_response_degraded(self) -> None:
        """Verify custom attributes for degraded service state."""
        health = HealthResponse(
            status="degraded",
            model_loaded=False,
            version="1.0.0",
            mock_mode=True,
            uptime_seconds=120.5,
        )
        assert health.status == "degraded"
        assert health.model_loaded is False
        assert health.mock_mode is True
        assert health.uptime_seconds == 120.5
