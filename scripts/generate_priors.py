"""Generate Population Priors Cache for Single-Row Tabular Inference.

Extracts empirical population statistics, categorical frequency distributions,
and multi-cohort aggregations from data/train.csv without modifying the raw dataset (REG-3).
Saves the serialized artifacts to data/priors.json (ADR-0002).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

# Ensure project root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.features import (
    GENDER_CATEGORIES,
    GENDER_MAP,
    GENDER_STRESS_CATEGORIES,
    GENDER_STRESS_MAP,
    IMPACT_CATEGORIES,
    IMPACT_MAP,
    STRESS_CATEGORIES,
    STRESS_IMPACT_CATEGORIES,
    STRESS_IMPACT_MAP,
    STRESS_MAP,
)


def compute_priors(train_df: pd.DataFrame) -> dict[str, Any]:
    """Compute all required priors from the raw training dataset."""
    raw_num = [
        "age",
        "daily_screen_time_hours",
        "social_media_hours",
        "gaming_hours",
        "work_study_hours",
        "sleep_hours",
        "notifications_per_day",
        "app_opens_per_day",
        "weekend_screen_time",
    ]
    raw_cat = ["gender", "stress_level", "academic_work_impact"]

    # 1. Feature column names matching LightGBM Fold 1 exactly (74 columns)
    feat_cols = [
        "age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
        "work_study_hours", "sleep_hours", "notifications_per_day", "app_opens_per_day",
        "weekend_screen_time", "age_isna", "daily_screen_time_hours_isna",
        "social_media_hours_isna", "gaming_hours_isna", "work_study_hours_isna",
        "sleep_hours_isna", "notifications_per_day_isna", "app_opens_per_day_isna",
        "weekend_screen_time_isna", "gender_isna", "stress_level_isna",
        "academic_work_impact_isna", "num_missing", "gender_code", "stress_code",
        "impact_code", "gender_stress_code", "stress_impact_code", "age_freq",
        "gender_freq", "stress_level_freq", "academic_work_impact_freq",
        "sleep_hours_freq", "daily_screen_time_hours_freq", "gender_stress_freq",
        "stress_impact_freq", "recreational_hours", "recreational_to_screen",
        "non_recreational_screen", "total_accounted_hours", "unaccounted_hours",
        "waking_hours", "screen_fraction_of_waking", "screen_fraction_of_day",
        "weekend_vs_weekday_diff", "weekend_to_weekday_ratio", "weighted_weekly_screen",
        "app_opens_per_screen_hour", "avg_unlock_minutes", "notifications_per_app_open",
        "notifications_per_screen_hour", "interaction_density", "screen_to_sleep_ratio",
        "work_to_sleep_ratio", "work_to_screen_ratio", "social_to_gaming_ratio",
        "social_share_of_screen", "gaming_share_of_screen", "screen_x_social",
        "screen_x_weekend", "screen_x_opens", "screen_by_ag_mean", "screen_by_ag_std",
        "screen_diff_ag_mean", "screen_zscore_ag", "sleep_by_stress_mean",
        "sleep_by_stress_std", "sleep_diff_stress_mean", "sleep_zscore_stress",
        "opens_by_gs_mean", "notifs_by_gs_mean", "screen_by_gs_mean",
        "opens_diff_gs_mean", "notifs_diff_gs_mean", "screen_diff_gs_mean",
    ]

    # 2. Numerical summary statistics (medians, means, stds)
    medians: dict[str, float] = {}
    means: dict[str, float] = {}
    stds: dict[str, float] = {}

    for c in raw_num:
        series = train_df[c].dropna()
        medians[c] = float(series.median())
        means[c] = float(series.mean())
        stds[c] = float(series.std())

    # 3. Categorical frequency encodings
    frequencies: dict[str, dict[str, float]] = {}

    # Standard columns
    for c in ["age", "gender", "stress_level", "academic_work_impact", "sleep_hours", "daily_screen_time_hours"]:
        vc = train_df[c].value_counts(normalize=True).to_dict()
        freq_map: dict[str, float] = {}
        for val, prob in vc.items():
            freq_map[str(val)] = float(prob)
            if isinstance(val, (int, float)):
                # Store int string representation if applicable
                if float(val).is_integer():
                    freq_map[str(int(val))] = float(prob)
                # Store rounded representations
                freq_map[str(round(float(val), 2))] = float(prob)
        frequencies[c] = freq_map

    # Composite columns
    gender_stress = train_df["gender"].astype(str) + "_" + train_df["stress_level"].astype(str)
    vc_gs = gender_stress.value_counts(normalize=True).to_dict()
    frequencies["gender_stress"] = {str(k): float(v) for k, v in vc_gs.items()}

    stress_impact = train_df["stress_level"].astype(str) + "_" + train_df["academic_work_impact"].astype(str)
    vc_si = stress_impact.value_counts(normalize=True).to_dict()
    frequencies["stress_impact"] = {str(k): float(v) for k, v in vc_si.items()}

    # 4. Multi-Cohort GroupBy Aggregations
    # Cohort 1: (Age, Gender) -> Screen time distribution
    grp_ag = train_df.groupby(["age", "gender"])["daily_screen_time_hours"].agg(["mean", "std"]).reset_index()
    cohort_ag: dict[str, dict[str, float]] = {}
    for _, row in grp_ag.iterrows():
        age_val = row["age"]
        gender_val = row["gender"]
        stats = {
            "screen_by_ag_mean": float(row["mean"]),
            "screen_by_ag_std": float(row["std"]) if pd.notna(row["std"]) else 1.0,
        }
        # Keys with float and int representations
        cohort_ag[f"{int(age_val)}_{gender_val}"] = stats
        cohort_ag[f"{float(age_val)}_{gender_val}"] = stats

    # Cohort 2: (Stress Level, Academic Impact) -> Sleep distribution
    grp_stress = train_df.groupby(["stress_level", "academic_work_impact"])["sleep_hours"].agg(["mean", "std"]).reset_index()
    cohort_stress: dict[str, dict[str, float]] = {}
    for _, row in grp_stress.iterrows():
        stress_val = row["stress_level"]
        impact_val = row["academic_work_impact"]
        cohort_stress[f"{stress_val}_{impact_val}"] = {
            "sleep_by_stress_mean": float(row["mean"]),
            "sleep_by_stress_std": float(row["std"]) if pd.notna(row["std"]) else 1.0,
        }

    # Cohort 3: (Gender, Stress Level) -> Behavioral Means
    grp_gs = (
        train_df.groupby(["gender", "stress_level"])[
            ["app_opens_per_day", "notifications_per_day", "daily_screen_time_hours"]
        ]
        .agg("mean")
        .reset_index()
    )
    cohort_gs: dict[str, dict[str, float]] = {}
    for _, row in grp_gs.iterrows():
        gender_val = row["gender"]
        stress_val = row["stress_level"]
        cohort_gs[f"{gender_val}_{stress_val}"] = {
            "opens_by_gs_mean": float(row["app_opens_per_day"]),
            "notifs_by_gs_mean": float(row["notifications_per_day"]),
            "screen_by_gs_mean": float(row["daily_screen_time_hours"]),
        }

    # 5. Global Population Fallbacks
    population_fallbacks: dict[str, float] = {
        "screen_by_ag_mean": float(train_df["daily_screen_time_hours"].dropna().mean()),
        "screen_by_ag_std": float(train_df["daily_screen_time_hours"].dropna().std()),
        "sleep_by_stress_mean": float(train_df["sleep_hours"].dropna().mean()),
        "sleep_by_stress_std": float(train_df["sleep_hours"].dropna().std()),
        "opens_by_gs_mean": float(train_df["app_opens_per_day"].dropna().mean()),
        "notifs_by_gs_mean": float(train_df["notifications_per_day"].dropna().mean()),
        "screen_by_gs_mean": float(train_df["daily_screen_time_hours"].dropna().mean()),
    }

    # 6. Categorical Encodings Reference
    categorical_maps = {
        "gender_map": GENDER_MAP,
        "stress_map": STRESS_MAP,
        "impact_map": IMPACT_MAP,
        "gender_stress_map": GENDER_STRESS_MAP,
        "stress_impact_map": STRESS_IMPACT_MAP,
    }

    return {
        "version": "1.0.0",
        "dataset_rows": len(train_df),
        "feature_names": feat_cols,
        "numerical_stats": {
            "medians": medians,
            "means": means,
            "stds": stds,
        },
        "frequencies": frequencies,
        "cohort_ag": cohort_ag,
        "cohort_stress": cohort_stress,
        "cohort_gs": cohort_gs,
        "population_fallbacks": population_fallbacks,
        "categorical_maps": categorical_maps,
    }


def generate_priors(train_path: Path, output_path: Path) -> dict[str, Any]:
    """Load train.csv (read-only), compute priors, and write output JSON."""
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found at {train_path}")

    # Read-only load of data/train.csv without modifying raw data (REG-3)
    train_df = pd.read_csv(train_path)

    priors = compute_priors(train_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(priors, f, indent=2)

    return priors


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract population priors from train.csv for single-row inference.")
    parser.add_argument("--train-path", type=str, default="data/train.csv", help="Path to raw train.csv")
    parser.add_argument("--output", type=str, default="data/priors.json", help="Path to output priors.json")
    args = parser.parse_args()

    train_path = Path(args.train_path).resolve()
    output_path = Path(args.output).resolve()

    print(f"Generating priors from {train_path}...")
    priors = generate_priors(train_path, output_path)
    print(f"Saved population priors to {output_path} ({len(priors['feature_names'])} features, {priors['dataset_rows']:,} rows processed).")


if __name__ == "__main__":
    main()
