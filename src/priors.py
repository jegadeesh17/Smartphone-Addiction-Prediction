"""Population Priors Cache Loader & Single-Row Feature Vectorizer.

Provides cached access to pre-computed population priors (data/priors.json) and
vectorizes a single BehavioralProfileInput into the exact 74-feature DataFrame
required by LightGBM Fold 1 with zero single-row NaNs (ADR-0002).
"""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from src.features import (
    GENDER_MAP,
    GENDER_STRESS_MAP,
    IMPACT_MAP,
    STRESS_IMPACT_MAP,
    STRESS_MAP,
)
from src.schemas import BehavioralProfileInput

# Default path to population priors
DEFAULT_PRIORS_PATH = Path(__file__).resolve().parent.parent / "data" / "priors.json"


@functools.lru_cache(maxsize=4)
def _load_priors_cached(resolved_path_str: str) -> dict[str, Any]:
    """Internal LRU-cached loader for priors dictionary from resolved path string."""
    path = Path(resolved_path_str)
    if not path.is_file():
        raise FileNotFoundError(
            f"Population priors file not found at '{resolved_path_str}'. "
            "Please generate it by running 'python scripts/generate_priors.py'."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_population_priors(path: Optional[Union[str, Path]] = None) -> dict[str, Any]:
    """Load pre-computed population priors dictionary with in-memory caching.

    Args:
        path: Optional explicit path to priors.json. Defaults to PRIORS_PATH env
              var or data/priors.json relative to repository root.

    Returns:
        dict containing feature_names, numerical_stats, frequencies, and cohort aggregations.

    Raises:
        FileNotFoundError: If the specified or default priors file cannot be found.
    """
    if path is None:
        env_path = os.getenv("PRIORS_PATH")
        target_path = Path(env_path) if env_path else DEFAULT_PRIORS_PATH
    else:
        target_path = Path(path)

    resolved_str = str(target_path.resolve())
    return _load_priors_cached(resolved_str)


def _get_frequency(freq_map: dict[str, Any], val: Any) -> float:
    """Safely lookup frequency value with numeric/string normalization and zero fallback."""
    if not freq_map or val is None:
        return 0.0

    s_val = str(val)
    if s_val in freq_map:
        return float(freq_map[s_val])

    if isinstance(val, (int, float, np.number)):
        try:
            f_val = float(val)
            if f_val.is_integer() and str(int(f_val)) in freq_map:
                return float(freq_map[str(int(f_val))])
            r2 = str(round(f_val, 2))
            if r2 in freq_map:
                return float(freq_map[r2])
            r1 = str(round(f_val, 1))
            if r1 in freq_map:
                return float(freq_map[r1])
            f_str = str(f_val)
            if f_str in freq_map:
                return float(freq_map[f_str])
        except (ValueError, TypeError, OverflowError):
            pass

    return 0.0


def build_single_row_features(
    profile: Union[BehavioralProfileInput, dict[str, Any]],
    priors: dict[str, Any],
) -> pd.DataFrame:
    """Vectorize a single behavioral profile into the exact 74 features matching LightGBM Fold 1.

    Guarantees:
    - Exactly 74 feature columns in the exact order required by LightGBM Fold 1.
    - Zero NaNs across all derived and interaction columns (ADR-0002).
    - Sub-millisecond compute execution time.

    Args:
        profile: BehavioralProfileInput instance or raw dictionary of behavioral inputs.
        priors: Pre-computed population priors dictionary loaded via load_population_priors.

    Returns:
        Single-row pandas DataFrame containing the 74 engineered features.
    """
    # 1. Extract raw field values with priors median fallback
    medians = priors.get("numerical_stats", {}).get("medians", {})

    if isinstance(profile, BehavioralProfileInput):
        age = profile.age
        gender = profile.gender
        stress_level = profile.stress_level
        academic_work_impact = profile.academic_work_impact
        daily_screen_time = profile.daily_screen_time_hours
        social_media_hours = profile.social_media_hours
        gaming_hours = profile.gaming_hours
        work_study_hours = profile.work_study_hours
        weekend_screen_time = profile.weekend_screen_time
        sleep_hours = profile.sleep_hours
        notifications_per_day = profile.notifications_per_day
        app_opens_per_day = profile.app_opens_per_day
    elif isinstance(profile, dict):
        age = profile.get("age", int(medians.get("age", 25)))
        gender = str(profile.get("gender", "Male"))
        stress_level = str(profile.get("stress_level", "Medium"))
        academic_work_impact = str(profile.get("academic_work_impact", "Yes"))
        daily_screen_time = profile.get("daily_screen_time_hours", profile.get("daily_screen_time", medians.get("daily_screen_time_hours", 7.5)))
        social_media_hours = profile.get("social_media_hours", medians.get("social_media_hours", 3.5))
        gaming_hours = profile.get("gaming_hours", medians.get("gaming_hours", 1.2))
        work_study_hours = profile.get("work_study_hours", medians.get("work_study_hours", 2.5))
        weekend_screen_time = profile.get("weekend_screen_time", medians.get("weekend_screen_time", 9.5))
        sleep_hours = profile.get("sleep_hours", medians.get("sleep_hours", 6.8))
        notifications_per_day = profile.get("notifications_per_day", int(medians.get("notifications_per_day", 140)))
        app_opens_per_day = profile.get("app_opens_per_day", int(medians.get("app_opens_per_day", 100)))
    else:
        raise TypeError(f"Expected BehavioralProfileInput or dict, got {type(profile).__name__}")

    # Track missingness if any value was None
    raw_inputs = {
        "age": age,
        "daily_screen_time_hours": daily_screen_time,
        "social_media_hours": social_media_hours,
        "gaming_hours": gaming_hours,
        "work_study_hours": work_study_hours,
        "sleep_hours": sleep_hours,
        "notifications_per_day": notifications_per_day,
        "app_opens_per_day": app_opens_per_day,
        "weekend_screen_time": weekend_screen_time,
        "gender": gender,
        "stress_level": stress_level,
        "academic_work_impact": academic_work_impact,
    }

    isna_flags = {f"{k}_isna": 1 if v is None or (isinstance(v, float) and np.isnan(v)) else 0 for k, v in raw_inputs.items()}
    num_missing = sum(isna_flags.values())

    # Fill None values with population medians / defaults
    soc_f = float(social_media_hours if social_media_hours is not None else 0.0)
    game_f = float(gaming_hours if gaming_hours is not None else 0.0)
    screen_f = float(daily_screen_time if daily_screen_time is not None else medians.get("daily_screen_time_hours", 7.77))
    sleep_f = float(sleep_hours if sleep_hours is not None else medians.get("sleep_hours", 6.80))
    work_f = float(work_study_hours if work_study_hours is not None else medians.get("work_study_hours", 2.20))
    weekend_f = float(weekend_screen_time if weekend_screen_time is not None else medians.get("weekend_screen_time", 9.58))
    opens_f = float(app_opens_per_day if app_opens_per_day is not None else medians.get("app_opens_per_day", 104.0))
    notifs_f = float(notifications_per_day if notifications_per_day is not None else medians.get("notifications_per_day", 150.0))
    age_f = float(age if age is not None else medians.get("age", 27.0))
    gender_s = str(gender if gender is not None else "Male")
    stress_s = str(stress_level if stress_level is not None else "Medium")
    impact_s = str(academic_work_impact if academic_work_impact is not None else "Yes")

    # Composite keys
    gender_stress_s = f"{gender_s}_{stress_s}"
    stress_impact_s = f"{stress_s}_{impact_s}"

    # 2. Categorical Encodings (Standard & Composite)
    gender_code = GENDER_MAP.get(gender_s, -1)
    stress_code = STRESS_MAP.get(stress_s, -1)
    impact_code = IMPACT_MAP.get(impact_s, -1)
    gender_stress_code = GENDER_STRESS_MAP.get(gender_stress_s, -1)
    stress_impact_code = STRESS_IMPACT_MAP.get(stress_impact_s, -1)

    # 3. Frequency Encodings
    freqs = priors.get("frequencies", {})
    age_freq = _get_frequency(freqs.get("age", {}), age_f)
    gender_freq = _get_frequency(freqs.get("gender", {}), gender_s)
    stress_level_freq = _get_frequency(freqs.get("stress_level", {}), stress_s)
    academic_work_impact_freq = _get_frequency(freqs.get("academic_work_impact", {}), impact_s)
    sleep_hours_freq = _get_frequency(freqs.get("sleep_hours", {}), sleep_f)
    daily_screen_time_hours_freq = _get_frequency(freqs.get("daily_screen_time_hours", {}), screen_f)
    gender_stress_freq = _get_frequency(freqs.get("gender_stress", {}), gender_stress_s)
    stress_impact_freq = _get_frequency(freqs.get("stress_impact", {}), stress_impact_s)

    # 4. Domain Time-Budget and Fragmentation Features
    recreational_hours = soc_f + game_f
    recreational_to_screen = recreational_hours / (screen_f + 1e-5)
    non_recreational_screen = max(0.0, screen_f - recreational_hours)
    total_accounted_hours = screen_f + work_f + sleep_f
    unaccounted_hours = 24.0 - total_accounted_hours
    waking_hours = 24.0 - sleep_f
    screen_fraction_of_waking = screen_f / max(1.0, waking_hours)
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
    social_to_gaming_ratio = (soc_f + 1e-5) / (game_f + 1e-5)
    social_share_of_screen = soc_f / (screen_f + 1e-5)
    gaming_share_of_screen = game_f / (screen_f + 1e-5)

    screen_x_social = screen_f * soc_f
    screen_x_weekend = screen_f * weekend_f
    screen_x_opens = screen_f * (opens_f / 100.0)

    # 5. Group Aggregations
    fallbacks = priors.get("population_fallbacks", {})

    # Cohort 1: (Age, Gender) Screen time
    cohort_ag = priors.get("cohort_ag", {})
    ag_stats = (
        cohort_ag.get(f"{int(age_f)}_{gender_s}")
        or cohort_ag.get(f"{float(age_f)}_{gender_s}")
        or {}
    )
    screen_by_ag_mean = ag_stats.get("screen_by_ag_mean", fallbacks.get("screen_by_ag_mean", 7.64))
    screen_by_ag_std = ag_stats.get("screen_by_ag_std", fallbacks.get("screen_by_ag_std", 2.72))
    screen_diff_ag_mean = screen_f - screen_by_ag_mean
    screen_zscore_ag = screen_diff_ag_mean / (screen_by_ag_std + 1e-5)

    # Cohort 2: (Stress, Academic Impact) Sleep
    cohort_stress = priors.get("cohort_stress", {})
    stress_stats = cohort_stress.get(stress_impact_s, {})
    sleep_by_stress_mean = stress_stats.get("sleep_by_stress_mean", fallbacks.get("sleep_by_stress_mean", 6.80))
    sleep_by_stress_std = stress_stats.get("sleep_by_stress_std", fallbacks.get("sleep_by_stress_std", 1.23))
    sleep_diff_stress_mean = sleep_f - sleep_by_stress_mean
    sleep_zscore_stress = sleep_diff_stress_mean / (sleep_by_stress_std + 1e-5)

    # Cohort 3: (Gender, Stress) Behavioral Means
    cohort_gs = priors.get("cohort_gs", {})
    gs_stats = cohort_gs.get(gender_stress_s, {})
    opens_by_gs_mean = gs_stats.get("opens_by_gs_mean", fallbacks.get("opens_by_gs_mean", 102.64))
    notifs_by_gs_mean = gs_stats.get("notifs_by_gs_mean", fallbacks.get("notifs_by_gs_mean", 145.89))
    screen_by_gs_mean = gs_stats.get("screen_by_gs_mean", fallbacks.get("screen_by_gs_mean", 7.64))
    opens_diff_gs_mean = opens_f - opens_by_gs_mean
    notifs_diff_gs_mean = notifs_f - notifs_by_gs_mean
    screen_diff_gs_mean = screen_f - screen_by_gs_mean

    # 6. Assemble complete feature dictionary matching exact 74 columns
    feature_row = {
        "age": age_f,
        "daily_screen_time_hours": screen_f,
        "social_media_hours": soc_f,
        "gaming_hours": game_f,
        "work_study_hours": work_f,
        "sleep_hours": sleep_f,
        "notifications_per_day": notifs_f,
        "app_opens_per_day": opens_f,
        "weekend_screen_time": weekend_f,
        "age_isna": np.int8(isna_flags["age_isna"]),
        "daily_screen_time_hours_isna": np.int8(isna_flags["daily_screen_time_hours_isna"]),
        "social_media_hours_isna": np.int8(isna_flags["social_media_hours_isna"]),
        "gaming_hours_isna": np.int8(isna_flags["gaming_hours_isna"]),
        "work_study_hours_isna": np.int8(isna_flags["work_study_hours_isna"]),
        "sleep_hours_isna": np.int8(isna_flags["sleep_hours_isna"]),
        "notifications_per_day_isna": np.int8(isna_flags["notifications_per_day_isna"]),
        "app_opens_per_day_isna": np.int8(isna_flags["app_opens_per_day_isna"]),
        "weekend_screen_time_isna": np.int8(isna_flags["weekend_screen_time_isna"]),
        "gender_isna": np.int8(isna_flags["gender_isna"]),
        "stress_level_isna": np.int8(isna_flags["stress_level_isna"]),
        "academic_work_impact_isna": np.int8(isna_flags["academic_work_impact_isna"]),
        "num_missing": np.int64(num_missing),
        "gender_code": np.int8(gender_code),
        "stress_code": np.int64(stress_code),
        "impact_code": np.int64(impact_code),
        "gender_stress_code": np.int8(gender_stress_code),
        "stress_impact_code": np.int8(stress_impact_code),
        "age_freq": np.float64(age_freq),
        "gender_freq": np.float64(gender_freq),
        "stress_level_freq": np.float64(stress_level_freq),
        "academic_work_impact_freq": np.float64(academic_work_impact_freq),
        "sleep_hours_freq": np.float64(sleep_hours_freq),
        "daily_screen_time_hours_freq": np.float64(daily_screen_time_hours_freq),
        "gender_stress_freq": np.float64(gender_stress_freq),
        "stress_impact_freq": np.float64(stress_impact_freq),
        "recreational_hours": np.float64(recreational_hours),
        "recreational_to_screen": np.float64(recreational_to_screen),
        "non_recreational_screen": np.float64(non_recreational_screen),
        "total_accounted_hours": np.float64(total_accounted_hours),
        "unaccounted_hours": np.float64(unaccounted_hours),
        "waking_hours": np.float64(waking_hours),
        "screen_fraction_of_waking": np.float64(screen_fraction_of_waking),
        "screen_fraction_of_day": np.float64(screen_fraction_of_day),
        "weekend_vs_weekday_diff": np.float64(weekend_vs_weekday_diff),
        "weekend_to_weekday_ratio": np.float64(weekend_to_weekday_ratio),
        "weighted_weekly_screen": np.float64(weighted_weekly_screen),
        "app_opens_per_screen_hour": np.float64(app_opens_per_screen_hour),
        "avg_unlock_minutes": np.float64(avg_unlock_minutes),
        "notifications_per_app_open": np.float64(notifications_per_app_open),
        "notifications_per_screen_hour": np.float64(notifications_per_screen_hour),
        "interaction_density": np.float64(interaction_density),
        "screen_to_sleep_ratio": np.float64(screen_to_sleep_ratio),
        "work_to_sleep_ratio": np.float64(work_to_sleep_ratio),
        "work_to_screen_ratio": np.float64(work_to_screen_ratio),
        "social_to_gaming_ratio": np.float64(social_to_gaming_ratio),
        "social_share_of_screen": np.float64(social_share_of_screen),
        "gaming_share_of_screen": np.float64(gaming_share_of_screen),
        "screen_x_social": np.float64(screen_x_social),
        "screen_x_weekend": np.float64(screen_x_weekend),
        "screen_x_opens": np.float64(screen_x_opens),
        "screen_by_ag_mean": np.float64(screen_by_ag_mean),
        "screen_by_ag_std": np.float64(screen_by_ag_std),
        "screen_diff_ag_mean": np.float64(screen_diff_ag_mean),
        "screen_zscore_ag": np.float64(screen_zscore_ag),
        "sleep_by_stress_mean": np.float64(sleep_by_stress_mean),
        "sleep_by_stress_std": np.float64(sleep_by_stress_std),
        "sleep_diff_stress_mean": np.float64(sleep_diff_stress_mean),
        "sleep_zscore_stress": np.float64(sleep_zscore_stress),
        "opens_by_gs_mean": np.float64(opens_by_gs_mean),
        "notifs_by_gs_mean": np.float64(notifs_by_gs_mean),
        "screen_by_gs_mean": np.float64(screen_by_gs_mean),
        "opens_diff_gs_mean": np.float64(opens_diff_gs_mean),
        "notifs_diff_gs_mean": np.float64(notifs_diff_gs_mean),
        "screen_diff_gs_mean": np.float64(screen_diff_gs_mean),
    }

    # Ensure ordering matches exact feature names in priors
    feat_cols = priors.get("feature_names", list(feature_row.keys()))
    df = pd.DataFrame([feature_row])[feat_cols]

    # Guarantee zero NaNs
    if df.isna().any().any():
        df = df.fillna(0.0)

    return df
