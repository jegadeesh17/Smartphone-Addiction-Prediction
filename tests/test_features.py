"""Unit and regression tests for population priors and single-row 74-feature vectorizer.

Validates SPEC REG-1 (create_features pipeline integrity and 74 columns),
REG-3 (raw train.csv dataset immutability), and ADR-0002 (vectorized single-row
transformer generates exact 74 features without single-row NaNs or group leakage).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest

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
    create_features,
)
from src.priors import build_single_row_features, load_population_priors
from src.schemas import BehavioralProfileInput

EXPECTED_TRAIN_HEADER = (
    "id,age,daily_screen_time_hours,social_media_hours,gaming_hours,work_study_hours,"
    "sleep_hours,notifications_per_day,app_opens_per_day,weekend_screen_time,gender,"
    "stress_level,academic_work_impact,addicted_label\n"
)
EXPECTED_TRAIN_LINE_COUNT = 691370  # 1 header + 691,369 participant rows
EXPECTED_TRAIN_SIZE_BYTES = 44855546


# ==============================================================================
# 1. Population Priors Loader Tests
# ==============================================================================


def test_load_population_priors_success(data_dir: Path) -> None:
    """Verify that population priors load cleanly and contain all expected top-level schema keys."""
    priors_file = data_dir / "priors.json"
    assert priors_file.exists(), f"priors.json not found at {priors_file}"

    priors = load_population_priors(priors_file)
    assert isinstance(priors, dict)

    required_keys = [
        "version",
        "dataset_rows",
        "feature_names",
        "numerical_stats",
        "frequencies",
        "cohort_ag",
        "cohort_stress",
        "cohort_gs",
        "population_fallbacks",
        "categorical_maps",
    ]
    for key in required_keys:
        assert key in priors, f"Key '{key}' missing from priors.json"

    assert len(priors["feature_names"]) == 74
    assert priors["dataset_rows"] == 691369

    # Verify numerical stats medians
    medians = priors["numerical_stats"]["medians"]
    for col in [
        "daily_screen_time_hours",
        "sleep_hours",
        "work_study_hours",
        "weekend_screen_time",
        "app_opens_per_day",
        "notifications_per_day",
    ]:
        assert col in medians
        assert isinstance(medians[col], (int, float))


def test_load_population_priors_caching(data_dir: Path) -> None:
    """Verify that load_population_priors caches results in memory across calls."""
    priors_file = data_dir / "priors.json"
    priors_1 = load_population_priors(priors_file)
    priors_2 = load_population_priors(priors_file)
    assert priors_1 is priors_2


def test_load_population_priors_missing_file_raises(tmp_path: Path) -> None:
    """Verify that passing a non-existent priors path raises FileNotFoundError."""
    missing_path = tmp_path / "non_existent_priors.json"
    with pytest.raises(FileNotFoundError):
        load_population_priors(missing_path)


# ==============================================================================
# 2. Single-Row 74-Feature Vectorizer Tests (ADR-0002)
# ==============================================================================


def test_single_row_feature_transformation_dimensions_and_names() -> None:
    """Verify that build_single_row_features produces exactly 74 features matching LightGBM."""
    priors = load_population_priors()
    profile = BehavioralProfileInput()

    df_feats = build_single_row_features(profile, priors)

    assert isinstance(df_feats, pd.DataFrame)
    assert df_feats.shape == (1, 74)
    assert list(df_feats.columns) == priors["feature_names"]

    # Assert zero NaNs across all columns (ADR-0002)
    nan_counts = df_feats.isna().sum().sum()
    assert nan_counts == 0, f"Found {nan_counts} NaNs in single-row feature transformation"

    # Assert no infinite values
    assert not np.isinf(df_feats.values).any(), "Found infinite values in single-row features"

    # Assert valid numeric types
    for col in df_feats.columns:
        assert pd.api.types.is_numeric_dtype(df_feats[col]), f"Column '{col}' is not numeric"


def test_single_row_feature_transformation_from_dict(sample_valid_profile: dict[str, Any]) -> None:
    """Verify that build_single_row_features works with a raw profile dictionary."""
    priors = load_population_priors()
    df_feats = build_single_row_features(sample_valid_profile, priors)

    assert isinstance(df_feats, pd.DataFrame)
    assert df_feats.shape == (1, 74)
    assert list(df_feats.columns) == priors["feature_names"]
    assert df_feats.isna().sum().sum() == 0


def test_single_row_matches_lightgbm_fold_1_feature_list(models_dir: Path) -> None:
    """Verify that single-row transformer columns match LightGBM fold 1 feature_name_."""
    priors = load_population_priors()
    profile = BehavioralProfileInput()
    df_feats = build_single_row_features(profile, priors)

    model_path = models_dir / "lgb_fold_1.joblib"
    assert model_path.exists(), f"Model not found at {model_path}"
    model = joblib.load(model_path)

    assert hasattr(model, "feature_name_")
    assert list(df_feats.columns) == model.feature_name_
    assert len(df_feats.columns) == len(model.feature_name_) == 74


# ==============================================================================
# 3. Categorical Constants and create_features Integrity (REG-1)
# ==============================================================================


def test_categorical_mapping_constants_exposed() -> None:
    """Verify that categorical mapping constants are cleanly exposed in src/features.py."""
    assert isinstance(GENDER_MAP, dict)
    assert GENDER_MAP["Female"] == 0
    assert GENDER_MAP["Male"] == 1
    assert GENDER_MAP["Other"] == 2

    assert isinstance(STRESS_MAP, dict)
    assert STRESS_MAP["Low"] == 0
    assert STRESS_MAP["Medium"] == 1
    assert STRESS_MAP["High"] == 2

    assert isinstance(IMPACT_MAP, dict)
    assert IMPACT_MAP["No"] == 0
    assert IMPACT_MAP["Yes"] == 1

    assert isinstance(GENDER_STRESS_MAP, dict)
    assert len(GENDER_STRESS_CATEGORIES) == 16
    assert GENDER_STRESS_MAP["Female_High"] == 0
    assert GENDER_STRESS_MAP["Male_High"] == 4

    assert isinstance(STRESS_IMPACT_MAP, dict)
    assert len(STRESS_IMPACT_CATEGORIES) == 12
    assert STRESS_IMPACT_MAP["High_No"] == 0


def test_create_features_pipeline_integrity_reg1(data_dir: Path) -> None:
    """Verify create_features preserves signature and produces 74 columns without unexpected NaNs (REG-1)."""
    train_path = data_dir / "train.csv"
    assert train_path.exists()

    # Test with clean non-missing sample: all 74 features must have zero NaNs
    rows = []
    for i in range(10):
        rows.append({
            "id": i,
            "age": 25.0,
            "daily_screen_time_hours": 7.5 + i * 0.1,
            "social_media_hours": 3.0,
            "gaming_hours": 1.0,
            "work_study_hours": 2.0,
            "sleep_hours": 7.0,
            "notifications_per_day": 100.0,
            "app_opens_per_day": 50.0,
            "weekend_screen_time": 9.0,
            "gender": "Male",
            "stress_level": "Medium",
            "academic_work_impact": "Yes",
            "addicted_label": 0,
        })
    df_clean = pd.DataFrame(rows)

    tr_fe, te_fe, feat_cols = create_features(df_clean)

    assert isinstance(tr_fe, pd.DataFrame)
    assert te_fe is None
    assert isinstance(feat_cols, list)
    assert len(feat_cols) == 74

    # On clean input without missing values, transformed output has 0 NaNs
    assert tr_fe[feat_cols].isna().sum().sum() == 0

    # Check column match with priors
    priors = load_population_priors()
    assert feat_cols == priors["feature_names"]


def test_create_features_with_test_split(data_dir: Path) -> None:
    """Verify create_features handles combined train and test splits."""
    rows_train = []
    for i in range(6):
        rows_train.append({
            "id": i,
            "age": 24.0,
            "daily_screen_time_hours": 6.0 + i * 0.2,
            "social_media_hours": 2.0,
            "gaming_hours": 1.0,
            "work_study_hours": 3.0,
            "sleep_hours": 7.5,
            "notifications_per_day": 120.0,
            "app_opens_per_day": 60.0,
            "weekend_screen_time": 8.0,
            "gender": "Female",
            "stress_level": "Low",
            "academic_work_impact": "No",
            "addicted_label": 0,
        })
    rows_test = []
    for i in range(4):
        rows_test.append({
            "id": 100 + i,
            "age": 28.0,
            "daily_screen_time_hours": 8.0 + i * 0.2,
            "social_media_hours": 3.5,
            "gaming_hours": 1.5,
            "work_study_hours": 2.5,
            "sleep_hours": 6.5,
            "notifications_per_day": 150.0,
            "app_opens_per_day": 80.0,
            "weekend_screen_time": 10.0,
            "gender": "Male",
            "stress_level": "High",
            "academic_work_impact": "Yes",
        })

    df_train_sample = pd.DataFrame(rows_train)
    df_test_sample = pd.DataFrame(rows_test)

    tr_fe, te_fe, feat_cols = create_features(df_train_sample, df_test_sample)

    assert isinstance(tr_fe, pd.DataFrame)
    assert isinstance(te_fe, pd.DataFrame)
    assert len(tr_fe) == 6
    assert len(te_fe) == 4
    assert len(feat_cols) == 74
    assert tr_fe[feat_cols].isna().sum().sum() == 0
    assert te_fe[feat_cols].isna().sum().sum() == 0


def test_create_features_domain_arithmetic_safe_on_raw_sample(data_dir: Path) -> None:
    """Verify safe domain arithmetic columns never produce cascading NaNs even with raw missingness."""
    train_path = data_dir / "train.csv"
    df_sample = pd.read_csv(train_path, nrows=50)

    tr_fe, _, feat_cols = create_features(df_sample)
    assert len(feat_cols) == 74

    # Domain features computed from filled variables must have ZERO NaNs
    domain_cols = [
        "recreational_hours",
        "recreational_to_screen",
        "non_recreational_screen",
        "total_accounted_hours",
        "unaccounted_hours",
        "waking_hours",
        "screen_fraction_of_waking",
        "screen_fraction_of_day",
        "weekend_vs_weekday_diff",
        "weekend_to_weekday_ratio",
        "weighted_weekly_screen",
        "app_opens_per_screen_hour",
        "avg_unlock_minutes",
        "notifications_per_app_open",
        "notifications_per_screen_hour",
        "interaction_density",
        "screen_to_sleep_ratio",
        "work_to_sleep_ratio",
        "work_to_screen_ratio",
        "social_to_gaming_ratio",
        "social_share_of_screen",
        "gaming_share_of_screen",
    ]
    for c in domain_cols:
        assert tr_fe[c].isna().sum() == 0, f"Unexpected NaN in domain feature '{c}'"


# ==============================================================================
# 4. Immutability of Raw Competition Datasets (REG-3)
# ==============================================================================


def test_data_train_csv_unmodified_reg3(data_dir: Path) -> None:
    """Verify data/train.csv remains unmodified in byte size, line count, and header (REG-3)."""
    train_path = data_dir / "train.csv"
    assert train_path.exists()

    # Exact byte size verification
    assert (
        train_path.stat().st_size == EXPECTED_TRAIN_SIZE_BYTES
    ), f"Train dataset modified! Expected {EXPECTED_TRAIN_SIZE_BYTES} bytes, got {train_path.stat().st_size}"

    # Header verification
    with open(train_path, "r", encoding="utf-8") as f:
        header = f.readline()
        assert header == EXPECTED_TRAIN_HEADER, f"Train header mutated: {header!r}"

    # Row count verification
    with open(train_path, "rb") as f:
        line_count = sum(1 for _ in f)
    assert (
        line_count == EXPECTED_TRAIN_LINE_COUNT
    ), f"Train line count mutated! Expected {EXPECTED_TRAIN_LINE_COUNT}, got {line_count}"


# ==============================================================================
# 5. Edge-Case Inputs Validation
# ==============================================================================


@pytest.mark.parametrize(
    "edge_case_params",
    [
        {"age": 18, "gender": "Female"},
        {"age": 35, "gender": "Other"},
        {"daily_screen_time_hours": 0.0, "social_media_hours": 0.0, "gaming_hours": 0.0},
        {"daily_screen_time_hours": 24.0, "social_media_hours": 12.0, "gaming_hours": 10.0},
        {"sleep_hours": 1.0, "work_study_hours": 0.0},
        {"sleep_hours": 18.0, "work_study_hours": 14.0},
        {"app_opens_per_day": 0, "notifications_per_day": 0},
        {"app_opens_per_day": 500, "notifications_per_day": 500},
        {"stress_level": "Low", "academic_work_impact": "No"},
        {"stress_level": "High", "academic_work_impact": "Yes"},
        {"weekend_screen_time": 0.0},
        {"weekend_screen_time": 24.0},
    ],
)
def test_edge_case_inputs(edge_case_params: dict[str, Any]) -> None:
    """Verify build_single_row_features handles boundary and extreme values with zero NaNs."""
    priors = load_population_priors()
    profile = BehavioralProfileInput(**edge_case_params)

    df_feats = build_single_row_features(profile, priors)

    assert df_feats.shape == (1, 74)
    assert df_feats.isna().sum().sum() == 0
    assert not np.isinf(df_feats.values).any()


def test_out_of_bounds_and_none_fallbacks() -> None:
    """Verify dictionary input with out-of-bounds values or missing fields uses fallbacks safely."""
    priors = load_population_priors()

    # Pass profile with unexpected / out-of-range values and missing keys
    raw_dict: dict[str, Any] = {
        "age": 50,  # Out of normal 18-35 range
        "gender": "UnknownGender",  # Unseen gender
        "stress_level": "Severe",  # Unseen stress
        "daily_screen_time_hours": 30.0,  # Extreme screen time
        "sleep_hours": 0.5,
    }

    df_feats = build_single_row_features(raw_dict, priors)

    assert df_feats.shape == (1, 74)
    assert df_feats.isna().sum().sum() == 0
    assert not np.isinf(df_feats.values).any()


# ==============================================================================
# 6. Model Evaluation on Single-Row Features
# ==============================================================================


@pytest.mark.slow
def test_model_inference_on_single_row_features(models_dir: Path) -> None:
    """Verify LightGBM Fold 1 model successfully scores vectorized single-row features."""
    priors = load_population_priors()
    profile = BehavioralProfileInput()
    df_feats = build_single_row_features(profile, priors)

    model_path = models_dir / "lgb_fold_1.joblib"
    model = joblib.load(model_path)

    proba = model.predict_proba(df_feats)
    assert proba.shape == (1, 2)
    assert 0.0 <= proba[0, 1] <= 1.0

    # At median population baseline, risk probability is near ~52.8%
    assert 0.40 <= proba[0, 1] <= 0.65
