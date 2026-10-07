"""Unit and latency benchmark tests for LightGBM inference engine and calibrated mock adapter.

Validates SPEC AC-1.1 (response payload schema), AC-1.2 (single-row inference p95 latency < 20ms),
AC-1.3 (threshold responsiveness), AC-1.9 (fail-fast on missing weights), REG-2 (deserialization),
ADR-0005 (mock adapter), ADR-0006 (status pill clinical spectrum), and ADR-0007 (baseline calibration).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from src.inference import (
    DEFAULT_MODEL_PATH,
    InferenceEngine,
    get_inference_engine,
    reset_inference_engine,
)
from src.mock_adapter import MockInferenceEngine
from src.priors import DEFAULT_PRIORS_PATH
from src.schemas import BehavioralMetrics, BehavioralProfileInput, PredictionResponse


# ==============================================================================
# 1. InferenceEngine Unit Tests
# ==============================================================================


class TestInferenceEngine:
    """Test suite for production LightGBM inference engine."""

    @pytest.fixture(autouse=True)
    def clean_singletons(self) -> None:
        """Ensure clean singleton state before and after each test."""
        reset_inference_engine()
        yield
        reset_inference_engine()

    def test_model_deserialization_and_priors_loading(self) -> None:
        """Verify model weights and population priors load successfully without error (REG-2)."""
        engine = InferenceEngine()
        assert engine.model is not None
        assert hasattr(engine.model, "predict_proba")
        assert engine.priors is not None
        assert "feature_names" in engine.priors
        assert len(engine.priors["feature_names"]) == 74

    def test_predict_schema_compliance(self) -> None:
        """Verify inference output matches PredictionResponse domain contract (AC-1.1)."""
        engine = InferenceEngine()
        profile = BehavioralProfileInput()
        response = engine.predict(profile)

        assert isinstance(response, PredictionResponse)
        assert isinstance(response.probability, float)
        assert 0.0 <= response.probability <= 1.0
        assert response.prediction in (0, 1)
        assert response.classification in ("ADDICTION DETECTED", "HEALTHY")
        assert response.severity in ("High", "Moderate", "Healthy")
        assert response.risk_tier in ("HIGH", "MODERATE", "LOW")
        assert response.status_label in (
            "ADDICTION DETECTED • HIGH RISK",
            "ADDICTION DETECTED • MODERATE RISK",
            "ADDICTION DETECTED • LOW RISK",
            "HEALTHY PATTERN • HIGH RISK",
            "HEALTHY PATTERN • MODERATE RISK",
            "HEALTHY PATTERN • LOW RISK",
        )
        assert isinstance(response.ratios, BehavioralMetrics)
        assert isinstance(response.interventions, list)
        assert response.latency_ms > 0.0
        assert response.decision_threshold == profile.decision_threshold

        # Verify computed aliases
        assert response.addiction_probability == response.probability
        assert response.is_addicted == (response.prediction == 1)
        assert response.metrics == response.ratios
        assert response.recommendations == response.interventions

    def test_probability_within_unit_interval(self) -> None:
        """Verify evaluated probability is strictly within [0.0, 1.0]."""
        engine = InferenceEngine()
        profile = BehavioralProfileInput()
        response = engine.predict(profile)
        assert 0.0 <= response.probability <= 1.0

    def test_threshold_responsiveness(self) -> None:
        """Verify threshold alterations re-evaluate classification while preserving probability and severity (AC-1.3, PAR-3, PAR-4)."""
        engine = InferenceEngine()

        # At default profile, LightGBM Fold 1 produces probability ~0.5282
        profile_lower = BehavioralProfileInput(threshold=0.50)
        res_lower = engine.predict(profile_lower)

        profile_higher = BehavioralProfileInput(threshold=0.60)
        res_higher = engine.predict(profile_higher)

        # Probabilities should be virtually identical
        assert abs(res_lower.probability - res_higher.probability) < 1e-4

        # With tau=0.50 (0.5282 >= 0.50): ADDICTION DETECTED, risk tier MODERATE / Moderate
        assert res_lower.prediction == 1
        assert res_lower.classification == "ADDICTION DETECTED"
        assert res_lower.status_label == "ADDICTION DETECTED • MODERATE RISK"
        assert res_lower.risk_tier == "MODERATE"
        assert res_lower.severity == "Moderate"

        # With tau=0.60 (0.5282 < 0.60): HEALTHY, while intrinsic risk tier remains MODERATE / Moderate
        assert res_higher.prediction == 0
        assert res_higher.classification == "HEALTHY"
        assert res_higher.status_label == "HEALTHY PATTERN • MODERATE RISK"
        assert res_higher.risk_tier == "MODERATE"
        assert res_higher.severity == "Moderate"

    def test_three_tier_clinical_spectrum_mapping(self) -> None:
        """Verify 3-tier clinical spectrum status_label and risk_tier per AC-1.1, PAR-4, and ADR-0006."""
        engine = InferenceEngine()

        # 1. High Severity / Elevated Risk Tier (P >= 0.70)
        p_high = BehavioralProfileInput(
            daily_screen_time_hours=14.0,
            social_media_hours=7.0,
            gaming_hours=3.0,
            weekend_screen_time=15.0,
            sleep_hours=4.5,
            notifications_per_day=300,
            app_opens_per_day=200,
            stress_level="High",
            academic_work_impact="Yes",
            threshold=0.50,
        )
        res_high = engine.predict(p_high)
        assert res_high.probability >= 0.70
        assert res_high.prediction == 1
        assert res_high.classification == "ADDICTION DETECTED"
        assert res_high.severity == "High"
        assert res_high.risk_tier == "HIGH"
        assert res_high.status_label == "ADDICTION DETECTED • HIGH RISK"

        # 2. Moderate Severity / Moderate Risk Tier (0.40 <= P < 0.70)
        # Default profile produces prob ~ 0.5282
        p_mod = BehavioralProfileInput(threshold=0.50)
        res_mod = engine.predict(p_mod)
        assert 0.40 <= res_mod.probability < 0.70
        assert res_mod.prediction == 1
        assert res_mod.classification == "ADDICTION DETECTED"
        assert res_mod.severity == "Moderate"
        assert res_mod.risk_tier == "MODERATE"
        assert res_mod.status_label == "ADDICTION DETECTED • MODERATE RISK"

        # 3. Healthy Usage Pattern / Low Risk Tier (P < 0.40)
        p_low = BehavioralProfileInput(
            daily_screen_time_hours=1.5,
            social_media_hours=0.5,
            gaming_hours=0.0,
            work_study_hours=1.0,
            weekend_screen_time=2.0,
            sleep_hours=9.0,
            notifications_per_day=20,
            app_opens_per_day=15,
            stress_level="Low",
            academic_work_impact="No",
            threshold=0.50,
        )
        res_low = engine.predict(p_low)
        assert res_low.probability < 0.40
        assert res_low.prediction == 0
        assert res_low.classification == "HEALTHY"
        assert res_low.severity == "Healthy"
        assert res_low.risk_tier == "LOW"
        assert res_low.status_label == "HEALTHY PATTERN • LOW RISK"

    def test_latency_benchmark_under_20ms(self) -> None:
        """Verify repeated inferences achieve p95 latency < 20ms (AC-1.2)."""
        engine = InferenceEngine()
        profile = BehavioralProfileInput()

        # Warm-up runs to initialize thread pools and CPU caches
        for _ in range(5):
            _ = engine.predict(profile)

        latencies_ms: list[float] = []
        response_latencies: list[float] = []
        for _ in range(20):
            t0 = time.perf_counter()
            response = engine.predict(profile)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(elapsed_ms)
            response_latencies.append(response.latency_ms)

        p95_latency = float(np.percentile(latencies_ms, 95))
        p95_response = float(np.percentile(response_latencies, 95))
        mean_latency = float(np.mean(latencies_ms))
        assert p95_latency < 20.0, f"p95 latency {p95_latency:.2f}ms exceeded 20ms SLA"
        assert p95_response < 20.0, f"p95 telemetry latency {p95_response:.2f}ms exceeded 20ms SLA"
        assert mean_latency < 15.0, f"mean latency {mean_latency:.2f}ms exceeded 15ms target"

    def test_missing_model_file_raises_file_not_found(self, tmp_path: Path) -> None:
        """Verify missing model file raises FileNotFoundError on initialization (AC-1.9, REG-2)."""
        non_existent_model = tmp_path / "non_existent_model.joblib"
        with pytest.raises(FileNotFoundError) as exc_info:
            InferenceEngine(model_path=non_existent_model)
        assert "Model checkpoint not found" in str(exc_info.value)

    def test_missing_priors_file_raises_file_not_found(self, tmp_path: Path) -> None:
        """Verify missing population priors file raises FileNotFoundError (AC-1.9)."""
        non_existent_priors = tmp_path / "non_existent_priors.json"
        with pytest.raises(FileNotFoundError) as exc_info:
            InferenceEngine(priors_path=non_existent_priors)
        assert "Population priors file not found" in str(exc_info.value)

    def test_predict_accepts_dict_input(self) -> None:
        """Verify predict() accepts raw valid profile dictionary."""
        engine = InferenceEngine()
        raw_dict = {
            "age": 24,
            "gender": "Female",
            "stress_level": "Medium",
            "academic_work_impact": "Yes",
            "daily_screen_time_hours": 7.0,
            "social_media_hours": 3.0,
            "gaming_hours": 1.0,
            "work_study_hours": 2.0,
            "weekend_screen_time": 8.5,
            "sleep_hours": 7.0,
            "notifications_per_day": 120,
            "app_opens_per_day": 90,
            "decision_threshold": 0.50,
        }
        res = engine.predict(raw_dict)
        assert isinstance(res, PredictionResponse)
        assert res.prediction in (0, 1)

    def test_predict_invalid_input_type_raises_type_error(self) -> None:
        """Verify passing incompatible types to predict() raises TypeError."""
        engine = InferenceEngine()
        with pytest.raises(TypeError):
            engine.predict(["invalid", "list"])  # type: ignore[arg-type]

    def test_predict_invalid_dict_raises_validation_error(self) -> None:
        """Verify passing invalid dict values violating bounds to predict() raises ValidationError."""
        engine = InferenceEngine()
        with pytest.raises(ValidationError):
            engine.predict({"age": 15})


# ==============================================================================
# 2. MockInferenceEngine Unit Tests
# ==============================================================================


class TestMockInferenceEngine:
    """Test suite for calibrated MockInferenceEngine adapter."""

    def test_mock_predict_schema_compliance(self) -> None:
        """Verify mock adapter produces fully compliant PredictionResponse payload."""
        mock = MockInferenceEngine()
        profile = BehavioralProfileInput()
        res = mock.predict(profile)

        assert isinstance(res, PredictionResponse)
        assert 0.0 <= res.probability <= 1.0
        assert res.prediction in (0, 1)
        assert res.classification in ("ADDICTION DETECTED", "HEALTHY")
        assert res.severity in ("High", "Moderate", "Healthy")
        assert res.risk_tier in ("HIGH", "MODERATE", "LOW")
        assert res.status_label in (
            "ADDICTION DETECTED • HIGH RISK",
            "ADDICTION DETECTED • MODERATE RISK",
            "ADDICTION DETECTED • LOW RISK",
            "HEALTHY PATTERN • HIGH RISK",
            "HEALTHY PATTERN • MODERATE RISK",
            "HEALTHY PATTERN • LOW RISK",
        )
        assert isinstance(res.ratios, BehavioralMetrics)
        assert isinstance(res.interventions, list)
        assert res.latency_ms > 0.0

    def test_mock_calibrated_baseline_at_median_inputs(self) -> None:
        """Verify baseline probability is calibrated to ~50.5% at median population inputs (ADR-0005, ADR-0007)."""
        mock = MockInferenceEngine()
        profile = BehavioralProfileInput()
        res = mock.predict(profile)

        # Baseline logit = 0.02 -> 1 / (1 + exp(-0.02)) = ~0.5050
        assert 0.48 <= res.probability <= 0.52
        assert abs(res.probability - 0.5050) < 0.01

    def test_mock_probability_bounds(self) -> None:
        """Verify probability is clipped safely between [0.02, 0.98]."""
        mock = MockInferenceEngine()

        # Extreme high usage
        p_extreme_high = BehavioralProfileInput(
            daily_screen_time_hours=24.0,
            social_media_hours=12.0,
            gaming_hours=10.0,
            weekend_screen_time=24.0,
            sleep_hours=1.0,
            notifications_per_day=500,
            app_opens_per_day=500,
            stress_level="High",
            academic_work_impact="Yes",
        )
        res_high = mock.predict(p_extreme_high)
        assert res_high.probability <= 0.98

        # Extreme low usage
        p_extreme_low = BehavioralProfileInput(
            daily_screen_time_hours=0.5,
            social_media_hours=0.0,
            gaming_hours=0.0,
            work_study_hours=0.0,
            weekend_screen_time=0.5,
            sleep_hours=12.0,
            notifications_per_day=10,
            app_opens_per_day=10,
            stress_level="Low",
            academic_work_impact="No",
        )
        res_low = mock.predict(p_extreme_low)
        assert res_low.probability >= 0.02

    def test_mock_sensitivity_to_screen_time(self) -> None:
        """Verify increasing daily screen time strictly increases predicted risk probability."""
        mock = MockInferenceEngine()
        p_low = BehavioralProfileInput(daily_screen_time_hours=4.0)
        p_mid = BehavioralProfileInput(daily_screen_time_hours=7.5)
        p_high = BehavioralProfileInput(daily_screen_time_hours=12.0)

        res_low = mock.predict(p_low)
        res_mid = mock.predict(p_mid)
        res_high = mock.predict(p_high)

        assert res_low.probability < res_mid.probability < res_high.probability

    def test_mock_sensitivity_to_sleep_hours(self) -> None:
        """Verify increasing sleep duration strictly decreases predicted risk probability."""
        mock = MockInferenceEngine()
        p_short_sleep = BehavioralProfileInput(sleep_hours=4.5)
        p_normal_sleep = BehavioralProfileInput(sleep_hours=6.8)
        p_long_sleep = BehavioralProfileInput(sleep_hours=9.0)

        res_short = mock.predict(p_short_sleep)
        res_normal = mock.predict(p_normal_sleep)
        res_long = mock.predict(p_long_sleep)

        assert res_short.probability > res_normal.probability > res_long.probability

    def test_mock_three_tier_clinical_spectrum(self) -> None:
        """Verify 3-tier clinical spectrum status_label mapping in mock engine."""
        mock = MockInferenceEngine()

        # 1. High Risk (P >= 0.70)
        p_high = BehavioralProfileInput(
            daily_screen_time_hours=14.0,
            social_media_hours=6.0,
            gaming_hours=3.0,
            sleep_hours=4.0,
            threshold=0.50,
        )
        res_high = mock.predict(p_high)
        assert res_high.probability >= 0.70
        assert res_high.prediction == 1
        assert res_high.severity == "High"
        assert res_high.risk_tier == "HIGH"
        assert res_high.status_label == "ADDICTION DETECTED • HIGH RISK"

        # 2. Moderate Risk (0.40 <= P < 0.70)
        p_mod = BehavioralProfileInput(threshold=0.50)  # default prob is ~0.5050
        res_mod = mock.predict(p_mod)
        assert 0.40 <= res_mod.probability < 0.70
        assert res_mod.prediction == 1
        assert res_mod.severity == "Moderate"
        assert res_mod.risk_tier == "MODERATE"
        assert res_mod.status_label == "ADDICTION DETECTED • MODERATE RISK"

        # 3. Healthy / Low Risk (P < 0.40)
        p_low = BehavioralProfileInput(
            daily_screen_time_hours=2.0,
            social_media_hours=0.5,
            gaming_hours=0.0,
            sleep_hours=9.5,
            notifications_per_day=30,
            app_opens_per_day=20,
            stress_level="Low",
            academic_work_impact="No",
            threshold=0.50,
        )
        res_low = mock.predict(p_low)
        assert res_low.probability < 0.40
        assert res_low.prediction == 0
        assert res_low.severity == "Healthy"
        assert res_low.risk_tier == "LOW"
        assert res_low.status_label == "HEALTHY PATTERN • LOW RISK"

    def test_mock_predict_accepts_dict(self) -> None:
        """Verify mock adapter accepts dictionary profile input."""
        mock = MockInferenceEngine()
        res = mock.predict({"daily_screen_time_hours": 6.0, "sleep_hours": 7.0})
        assert isinstance(res, PredictionResponse)

    def test_mock_predict_invalid_input_type_raises_type_error(self) -> None:
        """Verify passing incompatible types to mock predict() raises TypeError."""
        mock = MockInferenceEngine()
        with pytest.raises(TypeError):
            mock.predict(["invalid", "list"])  # type: ignore[arg-type]

    def test_mock_predict_invalid_dict_raises_validation_error(self) -> None:
        """Verify passing invalid dict values violating bounds to mock predict() raises ValidationError."""
        mock = MockInferenceEngine()
        with pytest.raises(ValidationError):
            mock.predict({"age": 15})


# ==============================================================================
# 3. Singleton Factory Tests
# ==============================================================================


class TestInferenceEngineSingleton:
    """Test suite for get_inference_engine singleton provider and mock overrides."""

    @pytest.fixture(autouse=True)
    def clean_singletons(self) -> None:
        """Ensure clean singleton state before and after each test."""
        reset_inference_engine()
        yield
        reset_inference_engine()

    def test_get_inference_engine_returns_real_by_default(self) -> None:
        """Verify get_inference_engine returns InferenceEngine by default."""
        engine = get_inference_engine()
        assert isinstance(engine, InferenceEngine)

    def test_get_inference_engine_returns_cached_instance(self) -> None:
        """Verify consecutive calls return the exact same instance."""
        e1 = get_inference_engine()
        e2 = get_inference_engine()
        assert e1 is e2

    def test_get_inference_engine_mock_override(self) -> None:
        """Verify use_mock=True returns MockInferenceEngine."""
        mock = get_inference_engine(use_mock=True)
        assert isinstance(mock, MockInferenceEngine)

    def test_get_inference_engine_env_var_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify USE_MOCK_MODEL=true environment variable triggers mock engine."""
        monkeypatch.setenv("USE_MOCK_MODEL", "true")
        engine = get_inference_engine()
        assert isinstance(engine, MockInferenceEngine)

    def test_reset_inference_engine(self) -> None:
        """Verify reset_inference_engine clears singleton references."""
        e1 = get_inference_engine()
        reset_inference_engine()
        e2 = get_inference_engine()
        assert e1 is not e2
