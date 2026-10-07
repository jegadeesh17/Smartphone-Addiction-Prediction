"""Calibrated mock inference adapter for rapid frontend prototyping and isolated testing.

Implements the identical prediction signature as InferenceEngine without requiring
large model artifact loading into memory (ADR-0005). Uses population-centered logit
heuristics calibrated to ~50.5% addiction risk probability at median population inputs
(ADR-0007).
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Union

import numpy as np
import pandas as pd

from src.recommendations import (
    RECOMMENDATION_NOTIFICATION_HYGIENE,
    RECOMMENDATION_RECREATION_AUDIT,
    RECOMMENDATION_SLEEP_PROTECTION,
    RECOMMENDATION_WEEKEND_DISCONNECT,
    classify_risk_tier,
    classify_severity,
    compute_metrics,
    generate_clinical_interventions,
    get_authoritative_status_label,
)
from src.schemas import (
    BatchPredictionResponse,
    BatchScoringRecord,
    BehavioralProfileInput,
    PredictionResponse,
)


def _get_primary_intervention(
    screen_to_sleep: float,
    recreational_share: float,
    app_opens: float,
    notifications: float,
    weekend_surge: float,
) -> str:
    """Determine top-priority clinical intervention recommendation based on ratio triggers."""
    if screen_to_sleep > 1.2:
        return RECOMMENDATION_SLEEP_PROTECTION
    if recreational_share > 0.60:
        return RECOMMENDATION_RECREATION_AUDIT
    if app_opens > 120 or notifications > 180:
        return RECOMMENDATION_NOTIFICATION_HYGIENE
    if weekend_surge > 2.5:
        return RECOMMENDATION_WEEKEND_DISCONNECT
    return "Balanced Routine"


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

    def batch_predict(
        self, df: pd.DataFrame, default_threshold: float = 0.50
    ) -> tuple[pd.DataFrame, BatchPredictionResponse]:
        """Evaluate multi-row DataFrame using calibrated centered logit heuristics.

        Args:
            df: Raw participant records DataFrame (up to 10,000 rows).
            default_threshold: Classification decision threshold tau (default 0.50).

        Returns:
            Tuple of (enriched_df, BatchPredictionResponse).
        """
        t0 = time.perf_counter()
        n_rows = len(df)
        if n_rows == 0:
            token = str(uuid.uuid4())
            empty_resp = BatchPredictionResponse(
                total_records=0,
                processed_records=0,
                addiction_count=0,
                addiction_prevalence_pct=0.0,
                download_token=token,
                sample_records=[],
                latency_ms=0.01,
                high_risk_pct=0.0,
                mean_screen_to_sleep=0.0,
                preview_rows=[],
            )
            return df.copy(), empty_resp

        # Standardize column aliases
        col_screen = "daily_screen_time_hours" if "daily_screen_time_hours" in df.columns else "daily_screen_time"
        col_weekend = "weekend_screen_time" if "weekend_screen_time" in df.columns else "weekend_screen_time_hours"
        col_sleep = "sleep_hours" if "sleep_hours" in df.columns else "sleep_duration_hours"

        screen_s = df[col_screen] if col_screen in df.columns else pd.Series(7.5, index=df.index)
        soc_s = df["social_media_hours"] if "social_media_hours" in df.columns else pd.Series(3.5, index=df.index)
        gaming_s = df["gaming_hours"] if "gaming_hours" in df.columns else pd.Series(1.2, index=df.index)
        work_s = df["work_study_hours"] if "work_study_hours" in df.columns else pd.Series(2.5, index=df.index)
        weekend_s = df[col_weekend] if col_weekend in df.columns else pd.Series(9.5, index=df.index)
        sleep_s = df[col_sleep] if col_sleep in df.columns else pd.Series(6.8, index=df.index)
        notifs_s = df["notifications_per_day"] if "notifications_per_day" in df.columns else pd.Series(140.0, index=df.index)
        opens_s = df["app_opens_per_day"] if "app_opens_per_day" in df.columns else pd.Series(100.0, index=df.index)
        gender_s = df["gender"] if "gender" in df.columns else pd.Series("Male", index=df.index)
        stress_s = df["stress_level"] if "stress_level" in df.columns else pd.Series("Medium", index=df.index)
        impact_s = df["academic_work_impact"] if "academic_work_impact" in df.columns else pd.Series("Yes", index=df.index)

        screen_f = screen_s.fillna(7.5).astype(float).to_numpy()
        soc_f = soc_s.fillna(3.5).astype(float).to_numpy()
        gaming_f = gaming_s.fillna(1.2).astype(float).to_numpy()
        weekend_f = weekend_s.fillna(9.5).astype(float).to_numpy()
        sleep_f = sleep_s.fillna(6.8).astype(float).to_numpy()
        notifs_f = notifs_s.fillna(140.0).astype(float).to_numpy()
        opens_f = opens_s.fillna(100.0).astype(float).to_numpy()

        stress_map = {"Low": 0, "Medium": 1, "High": 2}
        impact_map = {"No": 0, "Yes": 1}
        stress_num = stress_s.map(stress_map).fillna(1).astype(float).to_numpy()
        impact_num = impact_s.map(impact_map).fillna(1).astype(float).to_numpy()

        logit = (
            0.02
            + 0.35 * (screen_f - 7.5)
            + 0.40 * (soc_f - 3.5)
            + 0.25 * (gaming_f - 1.2)
            + 0.18 * (weekend_f - 9.5)
            - 0.35 * (sleep_f - 6.8)
            + 0.008 * (notifs_f - 140.0)
            + 0.010 * (opens_f - 100.0)
            + 0.22 * (stress_num - 1.0)
            + 0.30 * (impact_num - 1.0)
        )
        probs = np.round(np.clip(1.0 / (1.0 + np.exp(-logit)), 0.02, 0.98), 4)

        ss_ratios = np.round(screen_f / (sleep_f + 1e-5), 4)
        rec_shares = np.round((soc_f + gaming_f) / (screen_f + 1e-5), 4)
        unlock_mins = np.where(screen_f == 0.0, 0.0, np.round((screen_f * 60.0) / (opens_f + 1e-5), 4))
        weekend_surges = np.round(weekend_f - screen_f, 4)

        is_addicted = probs >= default_threshold
        predictions = is_addicted.astype(int)
        classifications = np.where(is_addicted, "ADDICTION DETECTED", "HEALTHY")
        risk_tiers = np.where(probs >= 0.70, "HIGH", np.where(probs >= 0.40, "MODERATE", "LOW"))
        status_labels = [
            get_authoritative_status_label(bool(pred), float(p), default_threshold)
            for pred, p in zip(predictions, probs)
        ]

        primary_interventions = [
            _get_primary_intervention(ss, rec, op, notf, w_surge)
            for ss, rec, op, notf, w_surge in zip(ss_ratios, rec_shares, opens_f, notifs_f, weekend_surges)
        ]

        df_enriched = df.copy()
        df_enriched["predicted_probability"] = probs
        df_enriched["prediction"] = predictions
        df_enriched["classification"] = classifications
        df_enriched["risk_tier"] = risk_tiers
        df_enriched["status_label"] = status_labels
        df_enriched["screen_to_sleep_ratio"] = ss_ratios
        df_enriched["primary_intervention"] = primary_interventions

        token = str(uuid.uuid4())
        addiction_count = int(predictions.sum())
        addiction_prevalence_pct = round(float(addiction_count / n_rows * 100.0), 2)
        high_risk_count = int((risk_tiers == "HIGH").sum())
        high_risk_pct = round(float(high_risk_count / n_rows * 100.0), 2)
        mean_ss = round(float(np.mean(ss_ratios)), 2)

        preview_limit = min(20, n_rows)
        sample_records: list[BatchScoringRecord] = []
        for i in range(preview_limit):
            row_dict = {
                k: (None if pd.isna(v) else v)
                for k, v in df.iloc[i].to_dict().items()
            }
            rec = BatchScoringRecord(
                row_index=i,
                predicted_probability=float(probs[i]),
                prediction=int(predictions[i]),
                classification=str(classifications[i]),
                status_label=str(status_labels[i]),
                risk_tier=str(risk_tiers[i]),
                primary_intervention=str(primary_interventions[i]),
                screen_to_sleep_ratio=float(ss_ratios[i]),
                ratios={
                    "screen_to_sleep_ratio": float(ss_ratios[i]),
                    "recreational_share": float(rec_shares[i]),
                    "avg_unlock_minutes": float(unlock_mins[i]),
                    "weekend_surge_hours": float(weekend_surges[i]),
                },
                **row_dict,
            )
            sample_records.append(rec)

        latency_ms = max(0.01, round((time.perf_counter() - t0) * 1000.0, 3))

        response = BatchPredictionResponse(
            total_records=n_rows,
            processed_records=n_rows,
            addiction_count=addiction_count,
            addiction_prevalence_pct=addiction_prevalence_pct,
            download_token=token,
            sample_records=sample_records,
            latency_ms=latency_ms,
            high_risk_pct=high_risk_pct,
            mean_screen_to_sleep=mean_ss,
            preview_rows=sample_records,
        )

        return df_enriched, response
