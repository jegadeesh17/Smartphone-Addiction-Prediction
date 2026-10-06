"""Shared pytest fixtures for Smartphone Addiction Prediction platform test suite."""

from pathlib import Path
from typing import Any
import pytest


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
