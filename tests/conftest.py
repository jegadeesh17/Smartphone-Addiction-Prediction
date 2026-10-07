import sys
from pathlib import Path
from typing import Any
import pandas as pd
import pytest

# Ensure src directory is on sys.path for direct module imports in tests
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


@pytest.fixture(scope="session")
def root_dir() -> Path:
    """Return the absolute Path to the project root directory."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def models_dir(root_dir: Path) -> Path:
    """Return the Path to the serialized models directory."""
    return root_dir / "models"


@pytest.fixture(scope="session")
def data_dir(root_dir: Path) -> Path:
    """Return the Path to the competition datasets and cached data directory."""
    return root_dir / "data"


@pytest.fixture
def sample_valid_profile() -> dict[str, Any]:
    """Provide a standard baseline BehavioralProfile dictionary centered on population medians."""
    return {
        "age": 25,
        "gender": "Male",
        "stress_level": "Medium",
        "academic_work_impact": "Yes",
        "daily_screen_time_hours": 7.5,
        "social_media_hours": 3.5,
        "gaming_hours": 1.2,
        "work_study_hours": 2.5,
        "weekend_screen_time": 9.5,
        "sleep_hours": 6.8,
        "notifications_per_day": 140,
        "app_opens_per_day": 100,
        "decision_threshold": 0.50,
    }


@pytest.fixture
def sample_synthetic_train_df() -> pd.DataFrame:
    """Provide a balanced, complete synthetic raw training DataFrame slice with >= 2 samples per cohort."""
    rows: list[dict[str, Any]] = []
    for i in range(12):
        rows.append({
            "id": i,
            "age": 20.0 + (i % 2),
            "daily_screen_time_hours": 6.0 + i * 0.2,
            "social_media_hours": 2.0,
            "gaming_hours": 1.0,
            "work_study_hours": 3.0,
            "sleep_hours": 7.0 + (i % 2) * 0.5,
            "notifications_per_day": 100.0 + i * 5,
            "app_opens_per_day": 50.0 + i * 2,
            "weekend_screen_time": 8.0 + i * 0.3,
            "gender": "Female" if (i % 4 < 2) else "Male",
            "stress_level": "Low" if (i % 2 == 0) else "High",
            "academic_work_impact": "No" if (i % 4 < 2) else "Yes",
            "addicted_label": i % 2,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_synthetic_test_df() -> pd.DataFrame:
    """Provide a balanced, complete synthetic raw test DataFrame slice matching competition test format."""
    rows: list[dict[str, Any]] = []
    for i in range(6):
        rows.append({
            "id": 100 + i,
            "age": 20.0 + (i % 2),
            "daily_screen_time_hours": 6.5 + i * 0.2,
            "social_media_hours": 2.2,
            "gaming_hours": 1.1,
            "work_study_hours": 2.8,
            "sleep_hours": 7.2,
            "notifications_per_day": 110.0,
            "app_opens_per_day": 55.0,
            "weekend_screen_time": 8.5,
            "gender": "Female" if (i % 4 < 2) else "Male",
            "stress_level": "Low" if (i % 2 == 0) else "High",
            "academic_work_impact": "No" if (i % 4 < 2) else "Yes",
        })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_feature_vector(sample_valid_profile: dict[str, Any]) -> pd.DataFrame:
    """Provide a single-row feature DataFrame containing the exact 74 engineered features in canonical order."""
    from src.priors import build_single_row_features, load_population_priors
    from src.schemas import BehavioralProfileInput

    priors = load_population_priors()
    profile = BehavioralProfileInput(**sample_valid_profile)
    return build_single_row_features(profile, priors)

