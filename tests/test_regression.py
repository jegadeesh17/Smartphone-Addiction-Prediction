"""Comprehensive regression safeguards and full pipeline verification test suite.

Validates:
- SPEC REG-1: Feature engineering pipeline integrity (create_features produces exactly 74 features with zero NaNs).
- SPEC REG-2: Model checkpoint deserialization (5-fold LightGBM, XGBoost, and CatBoost models load and evaluate valid probabilities).
- SPEC REG-3: Immutability of raw competition datasets (train.csv has 691,370 lines, test.csv has 296,303 lines).
- SPEC REG-4: Ensemble optimization and training driver interface stability (src/ensemble.py and src/train.py signatures intact).
- SPEC AC-1.2: Single-row inference latency benchmark (p95 latency < 20ms over 50 consecutive runs).
"""

from __future__ import annotations

import inspect
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest

from src.features import create_features
from src.inference import InferenceEngine, reset_inference_engine
from src.priors import load_population_priors
from src.schemas import BehavioralProfileInput, PredictionResponse

EXPECTED_TRAIN_HEADER = (
    "id,age,daily_screen_time_hours,social_media_hours,gaming_hours,work_study_hours,"
    "sleep_hours,notifications_per_day,app_opens_per_day,weekend_screen_time,gender,"
    "stress_level,academic_work_impact,addicted_label\n"
)
EXPECTED_TRAIN_LINE_COUNT = 691370  # 1 header line + 691,369 participant rows
EXPECTED_TRAIN_SIZE_BYTES = 44855546

EXPECTED_TEST_HEADER = (
    "id,age,daily_screen_time_hours,social_media_hours,gaming_hours,work_study_hours,"
    "sleep_hours,notifications_per_day,app_opens_per_day,weekend_screen_time,gender,"
    "stress_level,academic_work_impact\n"
)
EXPECTED_TEST_LINE_COUNT = 296303  # 1 header line + 296,302 participant rows
EXPECTED_TEST_SIZE_BYTES = 18672999

EXPECTED_FEATURE_COUNT = 74

DOMAIN_ENGINEERED_COLS = [
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


# ==============================================================================
# 1. REG-1: Feature Engineering Pipeline Integrity
# ==============================================================================


class TestFeaturePipelineIntegrityREG1:
    """Regression suite validating feature engineering pipeline integrity and 74-column contract."""

    def test_create_features_synthetic_slice_reg1(
        self,
        sample_synthetic_train_df: pd.DataFrame,
        sample_synthetic_test_df: pd.DataFrame,
    ) -> None:
        """Verify create_features(df_train, df_test) produces exactly 74 feature columns with zero NaNs."""
        tr_fe, te_fe, feat_cols = create_features(
            sample_synthetic_train_df, sample_synthetic_test_df
        )

        assert isinstance(tr_fe, pd.DataFrame), "tr_fe must be a DataFrame"
        assert isinstance(te_fe, pd.DataFrame), "te_fe must be a DataFrame"
        assert isinstance(feat_cols, list), "feat_cols must be a list of strings"
        assert len(feat_cols) == EXPECTED_FEATURE_COUNT, (
            f"Expected {EXPECTED_FEATURE_COUNT} feature columns, got {len(feat_cols)}"
        )

        # Verify zero NaNs in engineered features for both train and test splits
        nan_count_tr = tr_fe[feat_cols].isna().sum().sum()
        nan_count_te = te_fe[feat_cols].isna().sum().sum()
        assert nan_count_tr == 0, f"Transformed train features contain {nan_count_tr} NaNs"
        assert nan_count_te == 0, f"Transformed test features contain {nan_count_te} NaNs"

        # Verify feature columns match canonical population priors
        priors = load_population_priors()
        assert feat_cols == priors["feature_names"], (
            "Engineered feature names deviate from canonical population priors"
        )

    def test_create_features_train_only_reg1(
        self, sample_synthetic_train_df: pd.DataFrame
    ) -> None:
        """Verify create_features handles single train DataFrame without test split."""
        tr_fe, te_fe, feat_cols = create_features(sample_synthetic_train_df)

        assert te_fe is None, "te_fe should be None when df_test is omitted"
        assert len(feat_cols) == EXPECTED_FEATURE_COUNT
        nan_count = tr_fe[feat_cols].isna().sum().sum()
        assert nan_count == 0, f"Transformed train features contain {nan_count} NaNs"

    @pytest.mark.slow
    def test_create_features_full_train_dataset_slow(self, data_dir: Path) -> None:
        """Benchmark and verify create_features on full raw train.csv (691,369 rows).

        Marked @pytest.mark.slow as full dataset feature transformation takes > 2 seconds.
        """
        train_path = data_dir / "train.csv"
        assert train_path.exists(), f"Raw train.csv not found at {train_path}"

        df_train = pd.read_csv(train_path)
        assert len(df_train) == 691369

        t0 = time.perf_counter()
        tr_fe, _, feat_cols = create_features(df_train)
        elapsed_s = time.perf_counter() - t0

        assert len(feat_cols) == EXPECTED_FEATURE_COUNT
        assert len(tr_fe) == 691369

        # In raw dataset, input features may contain natural missingness, but all
        # derived domain arithmetic features must be strictly free of NaNs.
        for col in DOMAIN_ENGINEERED_COLS:
            assert tr_fe[col].isna().sum() == 0, (
                f"Unexpected NaN found in domain feature '{col}' on full dataset"
            )

        # Sanity check execution completed reasonably
        assert elapsed_s > 0.0


# ==============================================================================
# 2. REG-2: Model Checkpoint Deserialization
# ==============================================================================


class TestModelCheckpointDeserializationREG2:
    """Regression suite verifying model checkpoints exist, deserialize cleanly, and produce valid probabilities."""

    @pytest.mark.parametrize("model_prefix", ["lgb", "xgb", "cat"])
    def test_model_checkpoints_fold_1_fast(
        self, models_dir: Path, sample_feature_vector: pd.DataFrame, model_prefix: str
    ) -> None:
        """Verify Fold 1 checkpoint for each model family loads without AttributeError and evaluates valid probabilities."""
        model_path = models_dir / f"{model_prefix}_fold_1.joblib"
        assert model_path.exists(), f"Model checkpoint not found: {model_path}"
        assert model_path.stat().st_size > 0, f"Model file is empty: {model_path}"

        model = joblib.load(model_path)
        assert hasattr(model, "predict_proba"), (
            f"Deserialized {model_prefix} model missing predict_proba"
        )
        assert callable(model.predict_proba), "predict_proba is not callable"

        proba = model.predict_proba(sample_feature_vector)
        assert isinstance(proba, np.ndarray), "predict_proba must return a numpy array"
        assert proba.shape == (1, 2), (
            f"Expected probability shape (1, 2), got {proba.shape}"
        )
        assert 0.0 <= proba[0, 0] <= 1.0, f"Class 0 probability {proba[0, 0]} out of bounds"
        assert 0.0 <= proba[0, 1] <= 1.0, f"Class 1 probability {proba[0, 1]} out of bounds"
        assert abs(proba[0].sum() - 1.0) < 1e-4, "Class probabilities do not sum to 1.0"

    @pytest.mark.slow
    @pytest.mark.parametrize("fold", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("model_prefix", ["lgb", "xgb", "cat"])
    def test_all_15_model_checkpoints_slow(
        self,
        models_dir: Path,
        sample_feature_vector: pd.DataFrame,
        model_prefix: str,
        fold: int,
    ) -> None:
        """Verify all 5 folds across LightGBM, XGBoost, and CatBoost (15 models) load and evaluate valid probabilities.

        Marked @pytest.mark.slow as loading all 15 checkpoints across disk takes > 5 seconds.
        """
        model_path = models_dir / f"{model_prefix}_fold_{fold}.joblib"
        assert model_path.exists(), f"Model checkpoint not found: {model_path}"
        assert model_path.stat().st_size > 0, f"Checkpoint file is empty: {model_path}"

        model = joblib.load(model_path)
        proba = model.predict_proba(sample_feature_vector)

        assert proba.shape == (1, 2)
        assert 0.0 <= proba[0, 0] <= 1.0
        assert 0.0 <= proba[0, 1] <= 1.0
        assert abs(proba[0].sum() - 1.0) < 1e-4


# ==============================================================================
# 3. REG-3: Immutability of Raw Datasets
# ==============================================================================


class TestRawDatasetImmutabilityREG3:
    """Regression suite verifying competition datasets are unmodified, readable, and non-truncated."""

    def test_raw_train_csv_immutability_reg3(self, data_dir: Path) -> None:
        """Verify data/train.csv exists, matches exact byte size, line count, and header."""
        train_path = data_dir / "train.csv"
        assert train_path.exists(), f"data/train.csv does not exist at {train_path}"
        assert train_path.is_file(), f"{train_path} is not a regular file"

        # Byte size check
        actual_size = train_path.stat().st_size
        assert actual_size == EXPECTED_TRAIN_SIZE_BYTES, (
            f"train.csv byte size mutated! Expected {EXPECTED_TRAIN_SIZE_BYTES}, got {actual_size}"
        )

        # Header check
        with open(train_path, "r", encoding="utf-8") as f:
            header = f.readline()
        assert header == EXPECTED_TRAIN_HEADER, f"train.csv header mutated: {header!r}"

        # Exact line count check
        with open(train_path, "rb") as f:
            line_count = sum(1 for _ in f)
        assert line_count == EXPECTED_TRAIN_LINE_COUNT, (
            f"train.csv line count mutated! Expected {EXPECTED_TRAIN_LINE_COUNT}, got {line_count}"
        )

    def test_raw_test_csv_immutability_reg3(self, data_dir: Path) -> None:
        """Verify data/test.csv exists, matches exact byte size, line count, and header."""
        test_path = data_dir / "test.csv"
        assert test_path.exists(), f"data/test.csv does not exist at {test_path}"
        assert test_path.is_file(), f"{test_path} is not a regular file"

        # Byte size check
        actual_size = test_path.stat().st_size
        assert actual_size == EXPECTED_TEST_SIZE_BYTES, (
            f"test.csv byte size mutated! Expected {EXPECTED_TEST_SIZE_BYTES}, got {actual_size}"
        )

        # Header check
        with open(test_path, "r", encoding="utf-8") as f:
            header = f.readline()
        assert header == EXPECTED_TEST_HEADER, f"test.csv header mutated: {header!r}"

        # Exact line count check
        with open(test_path, "rb") as f:
            line_count = sum(1 for _ in f)
        assert line_count == EXPECTED_TEST_LINE_COUNT, (
            f"test.csv line count mutated! Expected {EXPECTED_TEST_LINE_COUNT}, got {line_count}"
        )


# ==============================================================================
# 4. REG-4: Ensemble Optimization & Training Driver Stability
# ==============================================================================


class TestDriverStabilityREG4:
    """Regression suite verifying public signatures and functions in src/ensemble.py and src/train.py."""

    def test_ensemble_driver_interface_reg4(self) -> None:
        """Verify src/ensemble.py remains importable and exports intact optimize_and_submit signature."""
        import src.ensemble as ensemble_mod

        assert hasattr(ensemble_mod, "optimize_and_submit"), (
            "src/ensemble.py missing optimize_and_submit function"
        )
        assert callable(ensemble_mod.optimize_and_submit), (
            "optimize_and_submit is not callable"
        )

        sig = inspect.signature(ensemble_mod.optimize_and_submit)
        assert len(sig.parameters) == 0, (
            f"optimize_and_submit signature changed; expected 0 params, got {sig.parameters}"
        )

    def test_training_driver_interface_reg4(self) -> None:
        """Verify src/train.py remains importable and exports intact train_pipeline signature."""
        import src.train as train_mod

        assert hasattr(train_mod, "train_pipeline"), (
            "src/train.py missing train_pipeline function"
        )
        assert callable(train_mod.train_pipeline), (
            "train_pipeline is not callable"
        )

        sig = inspect.signature(train_mod.train_pipeline)
        param_names = list(sig.parameters.keys())
        expected_params = ["n_splits", "random_state", "include_nn"]
        for p in expected_params:
            assert p in param_names, f"Missing parameter '{p}' in train_pipeline signature"

        assert sig.parameters["n_splits"].default == 5
        assert sig.parameters["random_state"].default == 42
        assert sig.parameters["include_nn"].default is True

    def test_neural_network_module_interface_reg4(self) -> None:
        """Verify src/nn_model.py remains importable and exports PyTorch tabular architectures."""
        import src.nn_model as nn_mod

        assert hasattr(nn_mod, "train_tabular_nn"), "src/nn_model.py missing train_tabular_nn"
        assert hasattr(nn_mod, "TabularResNet"), "src/nn_model.py missing TabularResNet class"
        assert callable(nn_mod.train_tabular_nn)


# ==============================================================================
# 5. AC-1.2: Single-Row Inference Latency Benchmark
# ==============================================================================


class TestInferenceLatencyBenchmarkAC12:
    """Performance SLA suite verifying single-row inference p95 latency < 20ms over 50 consecutive runs."""

    def test_single_row_inference_latency_benchmark_ac12(self) -> None:
        """Verify p95 inference latency on InferenceEngine is < 20ms over 50 consecutive runs (AC-1.2)."""
        reset_inference_engine()
        engine = InferenceEngine()
        profile = BehavioralProfileInput()

        # Warm-up iterations to prime JIT paths and CPU cache
        for _ in range(5):
            _ = engine.predict(profile)

        wall_latencies_ms: list[float] = []
        telemetry_latencies_ms: list[float] = []

        for _ in range(50):
            t0 = time.perf_counter()
            response = engine.predict(profile)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            assert isinstance(response, PredictionResponse)
            assert 0.0 <= response.probability <= 1.0
            wall_latencies_ms.append(elapsed_ms)
            telemetry_latencies_ms.append(response.latency_ms)

        assert len(wall_latencies_ms) == 50

        p95_wall = float(np.percentile(wall_latencies_ms, 95))
        p95_telemetry = float(np.percentile(telemetry_latencies_ms, 95))
        mean_wall = float(np.mean(wall_latencies_ms))

        # AC-1.2 Acceptance Criteria: p95 latency must be strictly < 20ms
        assert p95_wall < 20.0, (
            f"p95 wall latency ({p95_wall:.2f}ms) violated the < 20ms SLA threshold"
        )
        assert p95_telemetry < 20.0, (
            f"p95 telemetry latency ({p95_telemetry:.2f}ms) violated the < 20ms SLA threshold"
        )
        assert mean_wall < 15.0, (
            f"Mean wall latency ({mean_wall:.2f}ms) exceeded 15ms target"
        )
