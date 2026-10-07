"""Baseline test suite verifying test runner execution, asset presence, and data/model integrity."""

from pathlib import Path
from typing import Any
import pytest
import joblib


EXPECTED_TRAIN_HEADER = (
    "id,age,daily_screen_time_hours,social_media_hours,gaming_hours,work_study_hours,"
    "sleep_hours,notifications_per_day,app_opens_per_day,weekend_screen_time,gender,"
    "stress_level,academic_work_impact,addicted_label\n"
)
EXPECTED_TRAIN_LINE_COUNT = 691370  # 1 header + 691,369 participant rows
EXPECTED_TRAIN_SIZE_BYTES = 44855546

EXPECTED_TEST_HEADER = (
    "id,age,daily_screen_time_hours,social_media_hours,gaming_hours,work_study_hours,"
    "sleep_hours,notifications_per_day,app_opens_per_day,weekend_screen_time,gender,"
    "stress_level,academic_work_impact\n"
)
EXPECTED_TEST_LINE_COUNT = 296303  # 1 header + 296,302 participant rows
EXPECTED_TEST_SIZE_BYTES = 18672999

EXPECTED_LGB_FOLD_1_SIZE_BYTES = 49581796


def test_pytest_runner_execution(sample_valid_profile: dict[str, Any], root_dir: Path) -> None:
    """Verify that pytest runner executes cleanly, fixtures load, and basic configurations match."""
    assert root_dir.exists()
    assert (root_dir / "pytest.ini").is_file()
    assert isinstance(sample_valid_profile, dict)
    assert sample_valid_profile["age"] == 25
    assert sample_valid_profile["gender"] == "Male"
    assert sample_valid_profile["decision_threshold"] == 0.50
    assert 1.0 <= sample_valid_profile["sleep_hours"] <= 24.0
    assert 0.0 <= sample_valid_profile["daily_screen_time_hours"] <= 24.0


def test_models_lgb_fold_1_exists_and_unmutated(models_dir: Path) -> None:
    """Verify models/lgb_fold_1.joblib is present, non-empty, and unmutated."""
    model_path = models_dir / "lgb_fold_1.joblib"
    assert model_path.exists(), f"Model file not found at {model_path}"
    assert model_path.is_file(), f"Model path {model_path} is not a regular file"
    assert (
        model_path.stat().st_size == EXPECTED_LGB_FOLD_1_SIZE_BYTES
    ), f"Model size changed: expected {EXPECTED_LGB_FOLD_1_SIZE_BYTES}, got {model_path.stat().st_size}"


def test_data_train_csv_exists_and_unmutated(data_dir: Path) -> None:
    """Verify data/train.csv is present, matches exact byte size, header, and row count (REG-3)."""
    train_path = data_dir / "train.csv"
    assert train_path.exists(), f"Train dataset not found at {train_path}"
    assert train_path.is_file(), f"Train path {train_path} is not a regular file"
    assert (
        train_path.stat().st_size == EXPECTED_TRAIN_SIZE_BYTES
    ), f"Train dataset byte size mutated: expected {EXPECTED_TRAIN_SIZE_BYTES}, got {train_path.stat().st_size}"

    with open(train_path, "r", encoding="utf-8") as f:
        header = f.readline()
        assert header == EXPECTED_TRAIN_HEADER, f"Train dataset header changed: {header!r}"

    with open(train_path, "rb") as f:
        line_count = sum(1 for _ in f)
    assert (
        line_count == EXPECTED_TRAIN_LINE_COUNT
    ), f"Train line count mutated: expected {EXPECTED_TRAIN_LINE_COUNT}, got {line_count}"


def test_raw_baseline_test_csv_integrity(data_dir: Path) -> None:
    """Verify data/test.csv is present, matches exact byte size, header, and row count (REG-3)."""
    test_path = data_dir / "test.csv"
    assert test_path.exists(), f"Test dataset not found at {test_path}"
    assert test_path.is_file(), f"Test path {test_path} is not a regular file"
    assert (
        test_path.stat().st_size == EXPECTED_TEST_SIZE_BYTES
    ), f"Test dataset byte size mutated: expected {EXPECTED_TEST_SIZE_BYTES}, got {test_path.stat().st_size}"

    with open(test_path, "r", encoding="utf-8") as f:
        header = f.readline()
        assert header == EXPECTED_TEST_HEADER, f"Test dataset header changed: {header!r}"

    with open(test_path, "rb") as f:
        line_count = sum(1 for _ in f)
    assert (
        line_count == EXPECTED_TEST_LINE_COUNT
    ), f"Test line count mutated: expected {EXPECTED_TEST_LINE_COUNT}, got {line_count}"


def test_env_example_presence_and_variables(root_dir: Path) -> None:
    """Verify .env.example is present and defines all required platform environment variables."""
    env_file = root_dir / ".env.example"
    assert env_file.exists() and env_file.is_file(), ".env.example file is missing"
    content = env_file.read_text(encoding="utf-8")
    required_vars = [
        "APP_HOST",
        "APP_PORT",
        "APP_DEBUG",
        "MODEL_PATH",
        "PRIORS_PATH",
        "COHORT_CACHE_PATH",
        "USE_MOCK_MODEL",
    ]
    for var in required_vars:
        assert f"{var}=" in content, f"Required environment variable {var} missing from .env.example"


def test_invalid_asset_path_handling(tmp_path: Path) -> None:
    """Verify that attempting to load a non-existent model or data asset raises FileNotFoundError."""
    non_existent_file = tmp_path / "non_existent_model.joblib"
    with pytest.raises(FileNotFoundError):
        joblib.load(non_existent_file)

    non_existent_csv = tmp_path / "missing_dataset.csv"
    with pytest.raises(FileNotFoundError):
        with open(non_existent_csv, "r", encoding="utf-8") as f:
            f.read()


@pytest.mark.slow
def test_models_checkpoint_deserialization(models_dir: Path) -> None:
    """Verify models/lgb_fold_1.joblib deserializes cleanly into a model with predict_proba (REG-2)."""
    model_path = models_dir / "lgb_fold_1.joblib"
    model = joblib.load(model_path)
    assert hasattr(model, "predict_proba"), "Deserialized model missing predict_proba method"
    assert callable(model.predict_proba), "predict_proba attribute is not callable"

