"""LightGBM tabular inference engine and execution provider.

Loads champion LightGBM Fold 1 model weights and cached population priors to evaluate
single-profile behavioral inputs with sub-20ms latency SLA, constructing validated
PredictionResponse payloads with authoritative clinical spectrum mapping (ADR-0002, ADR-0006).
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any, Optional, Union

import joblib
import numpy as np
import pandas as pd

from src.features import (
    GENDER_MAP,
    GENDER_STRESS_MAP,
    IMPACT_MAP,
    STRESS_IMPACT_MAP,
    STRESS_MAP,
)
from src.mock_adapter import MockInferenceEngine
from src.priors import DEFAULT_PRIORS_PATH, build_single_row_features, load_population_priors
from src.recommendations import (
    RECOMMENDATION_NOTIFICATION_HYGIENE,
    RECOMMENDATION_RECREATION_AUDIT,
    RECOMMENDATION_SLEEP_PROTECTION,
    RECOMMENDATION_WEEKEND_DISCONNECT,
    classify_clinical_spectrum,
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

# Default path to champion LightGBM Fold 1 checkpoint
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "lgb_fold_1.joblib"


def _map_batch_freq(series: pd.Series, freq_map: dict[str, Any]) -> pd.Series:
    """Safely map frequency values across a pandas Series with string and numeric fallbacks."""
    if not freq_map:
        return pd.Series(0.0, index=series.index, dtype=np.float64)

    s_str = series.astype(str)
    res = s_str.map(freq_map)
    if res.isna().any():
        try:
            int_str = series.dropna().astype(int).astype(str)
            res = res.fillna(int_str.map(freq_map))
        except (ValueError, TypeError, OverflowError):
            pass
    if res.isna().any():
        try:
            round_str = series.dropna().round(2).astype(str)
            res = res.fillna(round_str.map(freq_map))
        except (ValueError, TypeError, OverflowError):
            pass
    return res.fillna(0.0).astype(np.float64)


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


def _build_batch_features_df(
    priors: dict[str, Any],
    age_s: pd.Series,
    screen_s: pd.Series,
    soc_s: pd.Series,
    gaming_s: pd.Series,
    work_s: pd.Series,
    weekend_s: pd.Series,
    sleep_s: pd.Series,
    notifs_s: pd.Series,
    opens_s: pd.Series,
    gender_s: pd.Series,
    stress_s: pd.Series,
    impact_s: pd.Series,
) -> pd.DataFrame:
    """Vectorized transformation of multi-row DataFrame into exact 74 features matching LightGBM Fold 1."""
    medians = priors.get("numerical_stats", {}).get("medians", {})
    fallbacks = priors.get("population_fallbacks", {})
    freqs = priors.get("frequencies", {})

    # 1. Missingness flags & count
    age_isna = age_s.isna().astype(np.int8)
    screen_isna = screen_s.isna().astype(np.int8)
    soc_isna = soc_s.isna().astype(np.int8)
    gaming_isna = gaming_s.isna().astype(np.int8)
    work_isna = work_s.isna().astype(np.int8)
    sleep_isna = sleep_s.isna().astype(np.int8)
    notifs_isna = notifs_s.isna().astype(np.int8)
    opens_isna = opens_s.isna().astype(np.int8)
    weekend_isna = weekend_s.isna().astype(np.int8)
    gender_isna = gender_s.isna().astype(np.int8)
    stress_isna = stress_s.isna().astype(np.int8)
    impact_isna = impact_s.isna().astype(np.int8)

    num_missing = (
        age_isna + screen_isna + soc_isna + gaming_isna + work_isna
        + sleep_isna + notifs_isna + opens_isna + weekend_isna
        + gender_isna + stress_isna + impact_isna
    ).astype(np.int64)

    # 2. Impute with medians/fallbacks for arithmetic
    age_f = age_s.fillna(medians.get("age", 27.0)).astype(np.float64)
    screen_f = screen_s.fillna(medians.get("daily_screen_time_hours", 7.77)).astype(np.float64)
    soc_f = soc_s.fillna(0.0).astype(np.float64)
    gaming_f = gaming_s.fillna(0.0).astype(np.float64)
    work_f = work_s.fillna(medians.get("work_study_hours", 2.20)).astype(np.float64)
    weekend_f = weekend_s.fillna(medians.get("weekend_screen_time", 9.58)).astype(np.float64)
    sleep_f = sleep_s.fillna(medians.get("sleep_hours", 6.80)).astype(np.float64)
    notifs_f = notifs_s.fillna(medians.get("notifications_per_day", 150.0)).astype(np.float64)
    opens_f = opens_s.fillna(medians.get("app_opens_per_day", 104.0)).astype(np.float64)

    gender_str = gender_s.fillna("Male").astype(str)
    stress_str = stress_s.fillna("Medium").astype(str)
    impact_str = impact_s.fillna("Yes").astype(str)

    gender_stress_str = gender_str + "_" + stress_str
    stress_impact_str = stress_str + "_" + impact_str

    # 3. Categorical encodings
    gender_code = gender_str.map(GENDER_MAP).fillna(-1).astype(np.int8)
    stress_code = stress_str.map(STRESS_MAP).fillna(-1).astype(np.int64)
    impact_code = impact_str.map(IMPACT_MAP).fillna(-1).astype(np.int64)
    gender_stress_code = gender_stress_str.map(GENDER_STRESS_MAP).fillna(-1).astype(np.int8)
    stress_impact_code = stress_impact_str.map(STRESS_IMPACT_MAP).fillna(-1).astype(np.int8)

    # 4. Frequency encodings
    age_freq = _map_batch_freq(age_f, freqs.get("age", {}))
    gender_freq = _map_batch_freq(gender_str, freqs.get("gender", {}))
    stress_level_freq = _map_batch_freq(stress_str, freqs.get("stress_level", {}))
    academic_work_impact_freq = _map_batch_freq(impact_str, freqs.get("academic_work_impact", {}))
    sleep_hours_freq = _map_batch_freq(sleep_f, freqs.get("sleep_hours", {}))
    daily_screen_time_hours_freq = _map_batch_freq(screen_f, freqs.get("daily_screen_time_hours", {}))
    gender_stress_freq = _map_batch_freq(gender_stress_str, freqs.get("gender_stress", {}))
    stress_impact_freq = _map_batch_freq(stress_impact_str, freqs.get("stress_impact", {}))

    # 5. Domain time-budget and fragmentation features
    recreational_hours = soc_f + gaming_f
    recreational_to_screen = recreational_hours / (screen_f + 1e-5)
    non_recreational_screen = np.maximum(0.0, screen_f - recreational_hours)
    total_accounted_hours = screen_f + work_f + sleep_f
    unaccounted_hours = 24.0 - total_accounted_hours
    waking_hours = 24.0 - sleep_f
    screen_fraction_of_waking = screen_f / np.maximum(1.0, waking_hours)
    screen_fraction_of_day = screen_f / 24.0

    weekend_vs_weekday_diff = weekend_f - screen_f
    weekend_to_weekday_ratio = weekend_f / (screen_f + 1e-5)
    weighted_weekly_screen = (5.0 * screen_f + 2.0 * weekend_f) / 7.0

    app_opens_per_screen_hour = opens_f / (screen_f + 1e-5)
    avg_unlock_minutes = (screen_f * 60.0) / (opens_f + 1e-5)
    notifications_per_app_open = notifs_f / (opens_f + 1e-5)
    notifications_per_screen_hour = notifs_f / (screen_f + 1e-5)
    interaction_density = (opens_f * notifs_f) / 1000.0

    screen_to_sleep_ratio = screen_f / (sleep_f + 1e-5)
    work_to_sleep_ratio = work_f / (sleep_f + 1e-5)
    work_to_screen_ratio = work_f / (screen_f + 1e-5)
    social_to_gaming_ratio = (soc_f + 1e-5) / (gaming_f + 1e-5)
    social_share_of_screen = soc_f / (screen_f + 1e-5)
    gaming_share_of_screen = gaming_f / (screen_f + 1e-5)

    screen_x_social = screen_f * soc_f
    screen_x_weekend = screen_f * weekend_f
    screen_x_opens = screen_f * (opens_f / 100.0)

    # 6. Group aggregations
    cohort_ag = priors.get("cohort_ag", {})
    ag_keys = age_f.astype(int).astype(str) + "_" + gender_str
    ag_means = {k: v.get("screen_by_ag_mean", fallbacks.get("screen_by_ag_mean", 7.64)) for k, v in cohort_ag.items()}
    ag_stds = {k: v.get("screen_by_ag_std", fallbacks.get("screen_by_ag_std", 2.72)) for k, v in cohort_ag.items()}
    screen_by_ag_mean = ag_keys.map(ag_means).fillna(fallbacks.get("screen_by_ag_mean", 7.64)).astype(np.float64)
    screen_by_ag_std = ag_keys.map(ag_stds).fillna(fallbacks.get("screen_by_ag_std", 2.72)).astype(np.float64)
    screen_diff_ag_mean = screen_f - screen_by_ag_mean
    screen_zscore_ag = screen_diff_ag_mean / (screen_by_ag_std + 1e-5)

    cohort_stress = priors.get("cohort_stress", {})
    stress_means = {k: v.get("sleep_by_stress_mean", fallbacks.get("sleep_by_stress_mean", 6.80)) for k, v in cohort_stress.items()}
    stress_stds = {k: v.get("sleep_by_stress_std", fallbacks.get("sleep_by_stress_std", 1.23)) for k, v in cohort_stress.items()}
    sleep_by_stress_mean = stress_impact_str.map(stress_means).fillna(fallbacks.get("sleep_by_stress_mean", 6.80)).astype(np.float64)
    sleep_by_stress_std = stress_impact_str.map(stress_stds).fillna(fallbacks.get("sleep_by_stress_std", 1.23)).astype(np.float64)
    sleep_diff_stress_mean = sleep_f - sleep_by_stress_mean
    sleep_zscore_stress = sleep_diff_stress_mean / (sleep_by_stress_std + 1e-5)

    cohort_gs = priors.get("cohort_gs", {})
    gs_opens = {k: v.get("opens_by_gs_mean", fallbacks.get("opens_by_gs_mean", 102.64)) for k, v in cohort_gs.items()}
    gs_notifs = {k: v.get("notifs_by_gs_mean", fallbacks.get("notifs_by_gs_mean", 145.89)) for k, v in cohort_gs.items()}
    gs_screen = {k: v.get("screen_by_gs_mean", fallbacks.get("screen_by_gs_mean", 7.64)) for k, v in cohort_gs.items()}
    opens_by_gs_mean = gender_stress_str.map(gs_opens).fillna(fallbacks.get("opens_by_gs_mean", 102.64)).astype(np.float64)
    notifs_by_gs_mean = gender_stress_str.map(gs_notifs).fillna(fallbacks.get("notifs_by_gs_mean", 145.89)).astype(np.float64)
    screen_by_gs_mean = gender_stress_str.map(gs_screen).fillna(fallbacks.get("screen_by_gs_mean", 7.64)).astype(np.float64)
    opens_diff_gs_mean = opens_f - opens_by_gs_mean
    notifs_diff_gs_mean = notifs_f - notifs_by_gs_mean
    screen_diff_gs_mean = screen_f - screen_by_gs_mean

    # 7. Assemble 74 features
    feature_dict = {
        "age": age_f,
        "daily_screen_time_hours": screen_f,
        "social_media_hours": soc_f,
        "gaming_hours": gaming_f,
        "work_study_hours": work_f,
        "sleep_hours": sleep_f,
        "notifications_per_day": notifs_f,
        "app_opens_per_day": opens_f,
        "weekend_screen_time": weekend_f,
        "age_isna": age_isna,
        "daily_screen_time_hours_isna": screen_isna,
        "social_media_hours_isna": soc_isna,
        "gaming_hours_isna": gaming_isna,
        "work_study_hours_isna": work_isna,
        "sleep_hours_isna": sleep_isna,
        "notifications_per_day_isna": notifs_isna,
        "app_opens_per_day_isna": opens_isna,
        "weekend_screen_time_isna": weekend_isna,
        "gender_isna": gender_isna,
        "stress_level_isna": stress_isna,
        "academic_work_impact_isna": impact_isna,
        "num_missing": num_missing,
        "gender_code": gender_code,
        "stress_code": stress_code,
        "impact_code": impact_code,
        "gender_stress_code": gender_stress_code,
        "stress_impact_code": stress_impact_code,
        "age_freq": age_freq,
        "gender_freq": gender_freq,
        "stress_level_freq": stress_level_freq,
        "academic_work_impact_freq": academic_work_impact_freq,
        "sleep_hours_freq": sleep_hours_freq,
        "daily_screen_time_hours_freq": daily_screen_time_hours_freq,
        "gender_stress_freq": gender_stress_freq,
        "stress_impact_freq": stress_impact_freq,
        "recreational_hours": recreational_hours,
        "recreational_to_screen": recreational_to_screen,
        "non_recreational_screen": non_recreational_screen,
        "total_accounted_hours": total_accounted_hours,
        "unaccounted_hours": unaccounted_hours,
        "waking_hours": waking_hours,
        "screen_fraction_of_waking": screen_fraction_of_waking,
        "screen_fraction_of_day": screen_fraction_of_day,
        "weekend_vs_weekday_diff": weekend_vs_weekday_diff,
        "weekend_to_weekday_ratio": weekend_to_weekday_ratio,
        "weighted_weekly_screen": weighted_weekly_screen,
        "app_opens_per_screen_hour": app_opens_per_screen_hour,
        "avg_unlock_minutes": avg_unlock_minutes,
        "notifications_per_app_open": notifications_per_app_open,
        "notifications_per_screen_hour": notifications_per_screen_hour,
        "interaction_density": interaction_density,
        "screen_to_sleep_ratio": screen_to_sleep_ratio,
        "work_to_sleep_ratio": work_to_sleep_ratio,
        "work_to_screen_ratio": work_to_screen_ratio,
        "social_to_gaming_ratio": social_to_gaming_ratio,
        "social_share_of_screen": social_share_of_screen,
        "gaming_share_of_screen": gaming_share_of_screen,
        "screen_x_social": screen_x_social,
        "screen_x_weekend": screen_x_weekend,
        "screen_x_opens": screen_x_opens,
        "screen_by_ag_mean": screen_by_ag_mean,
        "screen_by_ag_std": screen_by_ag_std,
        "screen_diff_ag_mean": screen_diff_ag_mean,
        "screen_zscore_ag": screen_zscore_ag,
        "sleep_by_stress_mean": sleep_by_stress_mean,
        "sleep_by_stress_std": sleep_by_stress_std,
        "sleep_diff_stress_mean": sleep_diff_stress_mean,
        "sleep_zscore_stress": sleep_zscore_stress,
        "opens_by_gs_mean": opens_by_gs_mean,
        "notifs_by_gs_mean": notifs_by_gs_mean,
        "screen_by_gs_mean": screen_by_gs_mean,
        "opens_diff_gs_mean": opens_diff_gs_mean,
        "notifs_diff_gs_mean": notifs_diff_gs_mean,
        "screen_diff_gs_mean": screen_diff_gs_mean,
    }

    feat_cols = priors.get("feature_names", list(feature_dict.keys()))
    return pd.DataFrame(feature_dict)[feat_cols].fillna(0.0)


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

        # 2. LightGBM inference (class 1 probability, single thread for low latency)
        probs = self.model.predict_proba(features_df, num_threads=1)
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

    def batch_predict(
        self, df: pd.DataFrame, default_threshold: float = 0.50
    ) -> tuple[pd.DataFrame, BatchPredictionResponse]:
        """Perform vectorized batch scoring and diagnostics on a multi-row DataFrame.

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

        # 1. Standardize columns
        col_screen = "daily_screen_time_hours" if "daily_screen_time_hours" in df.columns else "daily_screen_time"
        col_weekend = "weekend_screen_time" if "weekend_screen_time" in df.columns else "weekend_screen_time_hours"
        col_sleep = "sleep_hours" if "sleep_hours" in df.columns else "sleep_duration_hours"

        medians = self.priors.get("numerical_stats", {}).get("medians", {})

        age_s = df["age"] if "age" in df.columns else pd.Series(medians.get("age", 25.0), index=df.index)
        screen_s = df[col_screen] if col_screen in df.columns else pd.Series(medians.get("daily_screen_time_hours", 7.5), index=df.index)
        soc_s = df["social_media_hours"] if "social_media_hours" in df.columns else pd.Series(medians.get("social_media_hours", 3.5), index=df.index)
        gaming_s = df["gaming_hours"] if "gaming_hours" in df.columns else pd.Series(medians.get("gaming_hours", 1.2), index=df.index)
        work_s = df["work_study_hours"] if "work_study_hours" in df.columns else pd.Series(medians.get("work_study_hours", 2.5), index=df.index)
        weekend_s = df[col_weekend] if col_weekend in df.columns else pd.Series(medians.get("weekend_screen_time", 9.5), index=df.index)
        sleep_s = df[col_sleep] if col_sleep in df.columns else pd.Series(medians.get("sleep_hours", 6.8), index=df.index)
        notifs_s = df["notifications_per_day"] if "notifications_per_day" in df.columns else pd.Series(medians.get("notifications_per_day", 140.0), index=df.index)
        opens_s = df["app_opens_per_day"] if "app_opens_per_day" in df.columns else pd.Series(medians.get("app_opens_per_day", 100.0), index=df.index)
        gender_s = df["gender"] if "gender" in df.columns else pd.Series("Male", index=df.index)
        stress_s = df["stress_level"] if "stress_level" in df.columns else pd.Series("Medium", index=df.index)
        impact_s = df["academic_work_impact"] if "academic_work_impact" in df.columns else pd.Series("Yes", index=df.index)

        # 2. Vectorized Feature Extraction
        features_df = _build_batch_features_df(
            priors=self.priors,
            age_s=age_s,
            screen_s=screen_s,
            soc_s=soc_s,
            gaming_s=gaming_s,
            work_s=work_s,
            weekend_s=weekend_s,
            sleep_s=sleep_s,
            notifs_s=notifs_s,
            opens_s=opens_s,
            gender_s=gender_s,
            stress_s=stress_s,
            impact_s=impact_s,
        )

        # 3. Model batch inference
        probs = self.model.predict_proba(features_df)[:, 1]
        probs = np.round(np.clip(probs, 0.0, 1.0), 4)

        # 4. Behavioral metrics & classification vectors
        screen_f = screen_s.fillna(medians.get("daily_screen_time_hours", 7.77)).astype(float).to_numpy()
        sleep_f = sleep_s.fillna(medians.get("sleep_hours", 6.80)).astype(float).to_numpy()
        soc_f = soc_s.fillna(0.0).astype(float).to_numpy()
        gaming_f = gaming_s.fillna(0.0).astype(float).to_numpy()
        weekend_f = weekend_s.fillna(medians.get("weekend_screen_time", 9.58)).astype(float).to_numpy()
        opens_f = opens_s.fillna(medians.get("app_opens_per_day", 104.0)).astype(float).to_numpy()
        notifs_f = notifs_s.fillna(medians.get("notifications_per_day", 150.0)).astype(float).to_numpy()

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

        # Clinical primary interventions
        primary_interventions = [
            _get_primary_intervention(ss, rec, op, notf, w_surge)
            for ss, rec, op, notf, w_surge in zip(ss_ratios, rec_shares, opens_f, notifs_f, weekend_surges)
        ]

        # 5. Enrich DataFrame (preserving all original input columns)
        df_enriched = df.copy()
        df_enriched["predicted_probability"] = probs
        df_enriched["prediction"] = predictions
        df_enriched["classification"] = classifications
        df_enriched["risk_tier"] = risk_tiers
        df_enriched["status_label"] = status_labels
        df_enriched["screen_to_sleep_ratio"] = ss_ratios
        df_enriched["primary_intervention"] = primary_interventions

        # 6. Aggregate Response
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


def is_inference_engine_loaded() -> bool:
    """Report whether the LightGBM engine singleton is loaded, without triggering a load."""
    return _engine_instance is not None


def reset_inference_engine() -> None:
    """Reset cached singleton instances (useful for test isolation)."""
    global _engine_instance, _mock_engine_instance
    _engine_instance = None
    _mock_engine_instance = None
