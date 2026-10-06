"""Calibrated mock inference adapter for rapid frontend prototyping and isolated testing.

Implements the identical prediction signature as InferenceEngine without requiring
large model artifact loading into memory (ADR-0005). Uses population-centered logit
heuristics calibrated to ~50.5% addiction risk probability at median population inputs
(ADR-0007).
"""

from __future__ import annotations

import time
from typing import Any, Union

import numpy as np

from src.recommendations import compute_metrics, generate_clinical_interventions
from src.schemas import BehavioralProfileInput, PredictionResponse


class MockInferenceEngine:
    """Mock inference adapter implementing identical prediction signature with calibrated baseline."""

    def __init__(self) -> None:
        """Initialize mock inference adapter."""
        pass

    def predict(
        self, profile: Union[BehavioralProfileInput, dict[str, Any]]
    ) -> PredictionResponse:
        """Evaluate behavioral profile using calibrated centered logit heuristics.

        Args:
            profile: BehavioralProfileInput instance or raw dictionary of behavioral inputs.

        Returns:
            PredictionResponse populated with calibrated probability, classification,
            metrics, interventions, and authoritative status label.
        """
        t0 = time.perf_counter()

        if isinstance(profile, dict):
            profile = BehavioralProfileInput.model_validate(profile)
        elif not isinstance(profile, BehavioralProfileInput):
            raise TypeError(
                f"Expected BehavioralProfileInput or dict, got {type(profile).__name__}"
            )

        stress_map = {"Low": 0, "Medium": 1, "High": 2}
        impact_map = {"No": 0, "Yes": 1}

        # Centered logit model: yields ~50.5% (48-52%) on standard population baseline (ADR-0007)
        logit = (
            0.02
            + 0.35 * (profile.daily_screen_time_hours - 7.5)
            + 0.40 * (profile.social_media_hours - 3.5)
            + 0.25 * (profile.gaming_hours - 1.2)
            + 0.18 * (profile.weekend_screen_time - 9.5)
            - 0.35 * (profile.sleep_hours - 6.8)
            + 0.008 * (profile.notifications_per_day - 140)
            + 0.010 * (profile.app_opens_per_day - 100)
            + 0.22 * (stress_map.get(profile.stress_level, 1) - 1)
            + 0.30 * (impact_map.get(profile.academic_work_impact, 1) - 1)
        )
        prob = float(np.clip(1.0 / (1.0 + np.exp(-logit)), 0.02, 0.98))
        prob_rounded = round(prob, 4)

        metrics = compute_metrics(profile)
        interventions = generate_clinical_interventions(profile, metrics)

        threshold = profile.decision_threshold
        if prob_rounded >= threshold:
            prediction = 1
            classification = "ADDICTION DETECTED"
            status_label = "Elevated Risk Tier"
            risk_tier = "HIGH"
        elif prob_rounded >= 0.35:
            prediction = 0
            classification = "HEALTHY"
            status_label = "Compensatory Usage Pattern"
            risk_tier = "MODERATE"
        else:
            prediction = 0
            classification = "HEALTHY"
            status_label = "Balanced Habit Profile"
            risk_tier = "LOW"

        latency_ms = max(0.01, round((time.perf_counter() - t0) * 1000.0, 3))

        return PredictionResponse(
            probability=prob_rounded,
            prediction=prediction,
            classification=classification,
            severity=risk_tier,
            status_label=status_label,
            ratios=metrics,
            interventions=interventions,
            latency_ms=latency_ms,
            decision_threshold=threshold,
        )
