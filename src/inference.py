"""LightGBM tabular inference engine and execution provider.

Loads champion LightGBM Fold 1 model weights and cached population priors to evaluate
single-profile behavioral inputs with sub-20ms latency SLA, constructing validated
PredictionResponse payloads with authoritative clinical spectrum mapping (ADR-0002, ADR-0006).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Optional, Union

import joblib

from src.mock_adapter import MockInferenceEngine
from src.priors import DEFAULT_PRIORS_PATH, build_single_row_features, load_population_priors
from src.recommendations import (
    classify_clinical_spectrum,
    classify_risk_tier,
    classify_severity,
    compute_metrics,
    generate_clinical_interventions,
    get_authoritative_status_label,
)
from src.schemas import BehavioralProfileInput, PredictionResponse

# Default path to champion LightGBM Fold 1 checkpoint
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "lgb_fold_1.joblib"


class InferenceEngine:
    """Production inference engine backed by LightGBM Fold 1 and cached population priors."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        priors_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize the inference engine with serialized model and population priors.

        Args:
            model_path: Path to serialized LightGBM checkpoint. Defaults to MODEL_PATH env var
                        or models/lgb_fold_1.joblib relative to project root.
            priors_path: Path to population priors JSON. Defaults to PRIORS_PATH env var
                         or data/priors.json relative to project root.

        Raises:
            FileNotFoundError: If the model file or priors file cannot be found (AC-1.9, REG-2).
        """
        if model_path is None:
            env_model = os.getenv("MODEL_PATH")
            target_model = Path(env_model) if env_model else DEFAULT_MODEL_PATH
        else:
            target_model = Path(model_path)

        if priors_path is None:
            env_priors = os.getenv("PRIORS_PATH")
            target_priors = Path(env_priors) if env_priors else DEFAULT_PRIORS_PATH
        else:
            target_priors = Path(priors_path)

        target_model = target_model.resolve()
        target_priors = target_priors.resolve()

        if not target_model.is_file():
            raise FileNotFoundError(
                f"Model checkpoint not found at '{target_model}'. "
                "Ensure models/lgb_fold_1.joblib exists (AC-1.9, REG-2)."
            )

        if not target_priors.is_file():
            raise FileNotFoundError(
                f"Population priors file not found at '{target_priors}'. "
                "Ensure data/priors.json exists or generate it via scripts/generate_priors.py (AC-1.9)."
            )

        self.model_path = target_model
        self.priors_path = target_priors
        self.model = joblib.load(target_model)
        self.priors = load_population_priors(target_priors)

    def predict(
        self, profile: Union[BehavioralProfileInput, dict[str, Any]]
    ) -> PredictionResponse:
        """Transform behavioral profile into 74 features and compute risk probability.

        Args:
            profile: BehavioralProfileInput instance or raw dictionary of behavioral inputs.

        Returns:
            PredictionResponse populated with evaluated probability, classification,
            metrics, interventions, latency measurement, and authoritative status label.
        """
        t0 = time.perf_counter()

        if isinstance(profile, dict):
            profile = BehavioralProfileInput.model_validate(profile)
        elif not isinstance(profile, BehavioralProfileInput):
            raise TypeError(
                f"Expected BehavioralProfileInput or dict, got {type(profile).__name__}"
            )

        # 1. Feature transformation using cached population priors
        features_df = build_single_row_features(profile, self.priors)

        # 2. LightGBM inference (class 1 probability)
        probs = self.model.predict_proba(features_df)
        prob = float(probs[0, 1])
        prob_rounded = round(prob, 4)

        # 3. Behavioral ratios and clinical recommendations
        metrics = compute_metrics(profile)
        interventions = generate_clinical_interventions(profile, metrics)

        # 4. Threshold classification and probability-based risk/severity per SPEC AC-1.1, PAR-3, PAR-4 & ADR-0006
        threshold = profile.decision_threshold
        is_addicted = prob_rounded >= threshold
        prediction = 1 if is_addicted else 0
        classification = "ADDICTION DETECTED" if is_addicted else "HEALTHY"
        risk_tier = classify_risk_tier(prob_rounded)
        severity = classify_severity(prob_rounded)
        status_label = get_authoritative_status_label(is_addicted, prob_rounded, threshold)

        latency_ms = max(0.01, round((time.perf_counter() - t0) * 1000.0, 3))

        return PredictionResponse(
            probability=prob_rounded,
            prediction=prediction,
            classification=classification,
            severity=severity,
            status_label=status_label,
            ratios=metrics,
            interventions=interventions,
            latency_ms=latency_ms,
            decision_threshold=threshold,
        )


# Singleton instances
_engine_instance: Optional[InferenceEngine] = None
_mock_engine_instance: Optional[MockInferenceEngine] = None


def get_inference_engine(
    use_mock: Optional[bool] = None,
    model_path: Optional[Union[str, Path]] = None,
    priors_path: Optional[Union[str, Path]] = None,
    force_new: bool = False,
) -> Union[InferenceEngine, MockInferenceEngine]:
    """Retrieve singleton inference engine instance with optional mock override.

    Args:
        use_mock: If True, returns MockInferenceEngine. If False, returns real InferenceEngine.
                  If None, checks USE_MOCK_MODEL environment variable (default False).
        model_path: Optional custom path to model checkpoint.
        priors_path: Optional custom path to priors JSON.
        force_new: If True, re-instantiates the engine even if a singleton already exists.

    Returns:
        Configured InferenceEngine or MockInferenceEngine instance.
    """
    global _engine_instance, _mock_engine_instance

    if use_mock is None:
        use_mock = os.getenv("USE_MOCK_MODEL", "false").lower() in ("true", "1", "yes")

    if use_mock:
        if _mock_engine_instance is None or force_new:
            _mock_engine_instance = MockInferenceEngine()
        return _mock_engine_instance

    target_model = Path(model_path).resolve() if model_path is not None else None
    target_priors = Path(priors_path).resolve() if priors_path is not None else None

    needs_new = (
        _engine_instance is None
        or force_new
        or (target_model is not None and _engine_instance.model_path != target_model)
        or (target_priors is not None and _engine_instance.priors_path != target_priors)
    )
    if needs_new:
        _engine_instance = InferenceEngine(model_path=model_path, priors_path=priors_path)
    return _engine_instance


def reset_inference_engine() -> None:
    """Reset cached singleton instances (useful for test isolation)."""
    global _engine_instance, _mock_engine_instance
    _engine_instance = None
    _mock_engine_instance = None
