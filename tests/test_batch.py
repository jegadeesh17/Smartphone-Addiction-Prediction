"""Unit and integration tests for batch CSV diagnostic scoring, validation, and report export.

Validates:
- AC-3.3: Batch CSV Scoring Execution (100 records -> 200 OK, prevalence rate, download_token, 20-row preview).
- AC-3.4: Batch Upload Header Validation Error Handling (Missing column -> HTTP 422 detailing missing headers).
- AC-3.5: Batch Upload Row Limit Enforcement (>10,000 rows -> HTTP 413 Payload Too Large).
- AC-3.6: Full Diagnostic CSV Export Download (GET with token -> 200 OK attachment 'diagnostic_report.csv').
- Mock adapter parity and isolation.
- Error handling for invalid tokens and malformed CSV uploads.
"""

from __future__ import annotations

import io
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import predict as predict_mod
from src.api.predict import clear_batch_export_cache, get_batch_export, store_batch_export
from src.config import Settings, get_settings
from src.inference import get_inference_engine, reset_inference_engine
from src.main import app
from src.mock_adapter import MockInferenceEngine


@pytest.fixture
def client() -> TestClient:
    """Create a FastAPI test client instance."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def clean_state() -> None:
    """Reset singletons, dependency overrides, and export cache before each test."""
    reset_inference_engine()
    clear_batch_export_cache()
    app.dependency_overrides.clear()
    get_settings.cache_clear()
    yield
    reset_inference_engine()
    clear_batch_export_cache()
    app.dependency_overrides.clear()
    get_settings.cache_clear()


def make_sample_cohort_df(n_rows: int = 100) -> pd.DataFrame:
    """Generate a synthetic DataFrame with valid behavioral profiles."""
    rows = []
    for i in range(n_rows):
        is_high = i % 2 == 0
        age = 19 + (i % 14)
        gender = "Female" if i % 3 == 0 else ("Male" if i % 3 == 1 else "Other")
        stress = "High" if is_high else ("Medium" if i % 3 == 0 else "Low")
        impact = "Yes" if is_high else "No"
        screen = 9.0 if is_high else 4.5
        soc = round(screen * 0.4, 2)
        gaming = round(screen * 0.2, 2)
        work = round(screen * 0.3, 2)
        weekend = screen + (3.0 if is_high else 1.0)
        sleep = 5.5 if is_high else 7.5
        notifs = 200 if is_high else 80
        opens = 150 if is_high else 60

        rows.append({
            "participant_id": f"P-{1000 + i}",
            "age": age,
            "gender": gender,
            "stress_level": stress,
            "academic_work_impact": impact,
            "daily_screen_time_hours": screen,
            "social_media_hours": soc,
            "gaming_hours": gaming,
            "work_study_hours": work,
            "weekend_screen_time": weekend,
            "sleep_hours": sleep,
            "notifications_per_day": notifs,
            "app_opens_per_day": opens,
        })
    return pd.DataFrame(rows)


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convert DataFrame to UTF-8 encoded CSV bytes."""
    return df.to_csv(index=False).encode("utf-8")


# ==============================================================================
# 1. Inference Engine Unit Tests
# ==============================================================================


class TestInferenceEngineBatch:
    """Unit tests for InferenceEngine.batch_predict and MockInferenceEngine.batch_predict."""

    def test_real_engine_batch_predict_100_rows(self) -> None:
        """Verify real inference engine accurately vectorizes and scores 100 rows."""
        engine = get_inference_engine(use_mock=False)
        df = make_sample_cohort_df(100)

        df_enriched, response = engine.batch_predict(df, default_threshold=0.50)

        assert len(df_enriched) == 100
        # Preserves original columns
        assert "participant_id" in df_enriched.columns
        assert "age" in df_enriched.columns

        # Enriched diagnostic columns
        assert "predicted_probability" in df_enriched.columns
        assert "prediction" in df_enriched.columns
        assert "classification" in df_enriched.columns
        assert "risk_tier" in df_enriched.columns
        assert "status_label" in df_enriched.columns
        assert "screen_to_sleep_ratio" in df_enriched.columns
        assert "primary_intervention" in df_enriched.columns

        # Verify summary response
        assert response.total_records == 100
        assert response.processed_records == 100
        assert 0 <= response.addiction_count <= 100
        assert 0.0 <= response.addiction_prevalence_pct <= 100.0
        assert bool(response.download_token)
        assert len(response.sample_records) == 20
        assert response.latency_ms > 0.0
        assert response.high_risk_pct is not None
        assert response.mean_screen_to_sleep is not None

        # Verify preview sample record schema
        sample = response.sample_records[0]
        assert sample.row_index == 0
        assert 0.0 <= sample.predicted_probability <= 1.0
        assert sample.classification in ("ADDICTION DETECTED", "HEALTHY")
        assert sample.risk_tier in ("HIGH", "MODERATE", "LOW")
        assert sample.status_label is not None
        assert sample.screen_to_sleep_ratio is not None
        assert sample.primary_intervention is not None
        assert sample.ratios is not None

    def test_mock_engine_batch_predict(self) -> None:
        """Verify MockInferenceEngine.batch_predict produces identical output structure."""
        engine = MockInferenceEngine()
        df = make_sample_cohort_df(50)

        df_enriched, response = engine.batch_predict(df, default_threshold=0.50)

        assert len(df_enriched) == 50
        assert response.total_records == 50
        assert response.processed_records == 50
        assert len(response.sample_records) == 20
        assert bool(response.download_token)
        assert 0.0 <= response.addiction_prevalence_pct <= 100.0

    def test_batch_predict_empty_dataframe(self) -> None:
        """Verify batch_predict handles zero-row DataFrame gracefully."""
        engine = get_inference_engine(use_mock=False)
        df_empty = pd.DataFrame(columns=["age", "gender", "daily_screen_time_hours"])

        df_enriched, response = engine.batch_predict(df_empty)

        assert len(df_enriched) == 0
        assert response.total_records == 0
        assert response.processed_records == 0
        assert response.addiction_count == 0
        assert len(response.sample_records) == 0


# ==============================================================================
# 2. Batch API Integration Tests (AC-3.3 to AC-3.6)
# ==============================================================================


class TestBatchAPI:
    """Integration test suite for POST /api/predict/batch and GET /api/predict/batch/export."""

    def test_batch_upload_100_rows_success(self, client: TestClient) -> None:
        """Verify POST /api/predict/batch successfully processes 100 rows (AC-3.3)."""
        df = make_sample_cohort_df(100)
        csv_bytes = df_to_csv_bytes(df)

        response = client.post(
            "/api/predict/batch",
            files={"file": ("cohort_100.csv", csv_bytes, "text/csv")},
        )

        assert response.status_code == 200
        data = response.json()

        assert data["total_records"] == 100
        assert data["processed_records"] == 100
        assert isinstance(data["addiction_count"], int)
        assert 0 <= data["addiction_count"] <= 100
        assert isinstance(data["addiction_prevalence_pct"], float)
        assert 0.0 <= data["addiction_prevalence_pct"] <= 100.0
        assert "download_token" in data
        assert len(data["download_token"]) > 10
        assert len(data["sample_records"]) == 20
        assert "preview_rows" in data
        assert len(data["preview_rows"]) == 20
        assert data["latency_ms"] > 0.0

    def test_batch_upload_missing_mandatory_column_returns_422(self, client: TestClient) -> None:
        """Verify upload missing daily_screen_time_hours returns HTTP 422 detailing missing headers (AC-3.4)."""
        df = make_sample_cohort_df(20)
        df_missing = df.drop(columns=["daily_screen_time_hours"])
        csv_bytes = df_to_csv_bytes(df_missing)

        response = client.post(
            "/api/predict/batch",
            files={"file": ("missing_screen.csv", csv_bytes, "text/csv")},
        )

        assert response.status_code == 422
        data = response.json()
        assert "daily_screen_time_hours" in str(data["detail"])

    def test_batch_upload_missing_multiple_mandatory_columns_returns_422(self, client: TestClient) -> None:
        """Verify upload missing multiple required columns returns 422 listing all missing names."""
        df = make_sample_cohort_df(10)
        df_missing = df.drop(columns=["sleep_hours", "gender", "notifications_per_day"])
        csv_bytes = df_to_csv_bytes(df_missing)

        response = client.post(
            "/api/predict/batch",
            files={"file": ("missing_multiple.csv", csv_bytes, "text/csv")},
        )

        assert response.status_code == 422
        detail = str(response.json()["detail"])
        assert "sleep_hours" in detail
        assert "gender" in detail
        assert "notifications_per_day" in detail

    def test_batch_upload_exceeds_10000_rows_returns_413(self, client: TestClient) -> None:
        """Verify upload exceeding 10,000 rows returns HTTP 413 Payload Too Large (AC-3.5)."""
        # Create CSV header plus 10,001 simple rows
        header = "age,gender,stress_level,academic_work_impact,daily_screen_time_hours,social_media_hours,gaming_hours,work_study_hours,weekend_screen_time,sleep_hours,notifications_per_day,app_opens_per_day\n"
        row = "25,Male,Medium,Yes,7.5,3.5,1.2,2.5,9.5,6.8,140,100\n"
        content = (header + row * 10_001).encode("utf-8")

        response = client.post(
            "/api/predict/batch",
            files={"file": ("oversized.csv", content, "text/csv")},
        )

        assert response.status_code == 413
        data = response.json()
        assert data["detail"] == "Batch upload exceeds maximum limit of 10,000 rows"

    def test_batch_export_download_success(self, client: TestClient) -> None:
        """Verify GET /api/predict/batch/export?token=... returns 200 OK with diagnostic report attachment (AC-3.6)."""
        df = make_sample_cohort_df(30)
        csv_bytes = df_to_csv_bytes(df)

        # 1. Run batch scoring
        upload_resp = client.post(
            "/api/predict/batch",
            files={"file": ("cohort_30.csv", csv_bytes, "text/csv")},
        )
        assert upload_resp.status_code == 200
        download_token = upload_resp.json()["download_token"]

        # 2. Request export download
        export_resp = client.get(f"/api/predict/batch/export?token={download_token}")
        assert export_resp.status_code == 200
        assert export_resp.headers["content-type"].startswith("text/csv")
        assert 'attachment; filename="diagnostic_report.csv"' in export_resp.headers["content-disposition"]

        # 3. Verify downloaded CSV content
        report_df = pd.read_csv(io.BytesIO(export_resp.content))
        assert len(report_df) == 30
        assert "participant_id" in report_df.columns
        assert "predicted_probability" in report_df.columns
        assert "classification" in report_df.columns
        assert "screen_to_sleep_ratio" in report_df.columns
        assert "primary_intervention" in report_df.columns

    def test_batch_export_invalid_token_returns_404(self, client: TestClient) -> None:
        """Verify invalid or nonexistent export token returns HTTP 404."""
        response = client.get("/api/predict/batch/export?token=nonexistent-uuid-token")
        assert response.status_code == 404
        assert response.json()["detail"] == "Export token not found or expired"

    def test_batch_alias_endpoints(self, client: TestClient) -> None:
        """Verify un-prefixed /predict/batch and /predict/batch/export aliases function identically."""
        df = make_sample_cohort_df(15)
        csv_bytes = df_to_csv_bytes(df)

        upload_resp = client.post(
            "/predict/batch",
            files={"file": ("test_alias.csv", csv_bytes, "text/csv")},
        )
        assert upload_resp.status_code == 200
        token = upload_resp.json()["download_token"]

        export_resp = client.get(f"/predict/batch/export?token={token}")
        assert export_resp.status_code == 200
        assert export_resp.headers["content-type"].startswith("text/csv")

    def test_batch_upload_accepts_column_aliases(self, client: TestClient) -> None:
        """Verify CSV containing column aliases (e.g. daily_screen_time) is accepted without 422 error."""
        df = make_sample_cohort_df(10)
        df_alias = df.rename(columns={
            "daily_screen_time_hours": "daily_screen_time",
            "weekend_screen_time": "weekend_screen_time_hours",
            "sleep_hours": "sleep_duration_hours",
        })
        csv_bytes = df_to_csv_bytes(df_alias)

        response = client.post(
            "/api/predict/batch",
            files={"file": ("aliases.csv", csv_bytes, "text/csv")},
        )
        assert response.status_code == 200
        assert response.json()["total_records"] == 10

    def test_batch_upload_empty_file_returns_422(self, client: TestClient) -> None:
        """Verify uploading an empty CSV file returns HTTP 422."""
        response = client.post(
            "/api/predict/batch",
            files={"file": ("empty.csv", b"", "text/csv")},
        )
        assert response.status_code == 422
        assert "empty" in str(response.json()["detail"]).lower()

    def test_batch_upload_mock_mode_settings(self, client: TestClient) -> None:
        """Verify batch scoring operates successfully under mock model configuration."""
        mock_settings = Settings(USE_MOCK_MODEL=True)
        app.dependency_overrides[get_settings] = lambda: mock_settings

        df = make_sample_cohort_df(25)
        csv_bytes = df_to_csv_bytes(df)

        response = client.post(
            "/api/predict/batch",
            files={"file": ("cohort_mock.csv", csv_bytes, "text/csv")},
        )
        assert response.status_code == 200
        assert response.json()["total_records"] == 25


# ==============================================================================
# 3. Batch Hardening Tests (numeric coercion, bounds, byte cap, formula safety)
# ==============================================================================

BATCH_HEADER = (
    "age,gender,stress_level,academic_work_impact,daily_screen_time_hours,social_media_hours,"
    "gaming_hours,work_study_hours,weekend_screen_time,sleep_hours,notifications_per_day,app_opens_per_day\n"
)
GOOD_ROW = "25,Male,Medium,Yes,7.5,3.5,1.2,2.5,9.5,6.8,140,100\n"


class TestBatchHardening:
    """Regression tests for reviewer defects in the batch CSV path."""

    def _post(self, client: TestClient, text: str):
        return client.post(
            "/api/predict/batch",
            files={"file": ("t.csv", text.encode("utf-8"), "text/csv")},
        )

    @pytest.mark.parametrize("bad", ["abc", "inf", "-inf", "nan", ""])
    def test_non_numeric_cell_returns_422_with_row_and_column(self, client: TestClient, bad: str) -> None:
        bad_row = f"25,Male,Low,No,{bad},1,1,2,8,7,100,100\n"
        response = self._post(client, BATCH_HEADER + GOOD_ROW + bad_row)
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert isinstance(detail, str)
        assert "daily_screen_time_hours" in detail
        assert "row(s) 2" in detail
        assert "[" not in detail

    def test_out_of_bounds_rows_return_422(self, client: TestClient) -> None:
        rows = (
            GOOD_ROW
            + "25,Male,Low,No,5,1,1,2,8,0,100,100\n"
            + "25,Male,Low,No,5,1,1,2,8,-3,100,100\n"
            + "25,Male,Low,No,5,1,1,2,8,7,100,0\n"
            + "17,Male,Low,No,5,1,1,2,8,7,100,10\n"
        )
        response = self._post(client, BATCH_HEADER + rows)
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert "sleep_hours" in detail and "row(s) 2, 3" in detail
        assert "age" in detail and "row(s) 5" in detail
        # app_opens_per_day=0 is within the schema bounds (ge=0) and must be accepted
        assert "app_opens_per_day" not in detail

    def test_zero_app_opens_accepted(self, client: TestClient) -> None:
        response = self._post(client, BATCH_HEADER + "25,Male,Low,No,5,1,1,2,8,7,100,0\n")
        assert response.status_code == 200

    def test_byte_cap_returns_413_before_parsing(self, client: TestClient) -> None:
        content = "x" * (5 * 1024 * 1024 + 1)
        response = self._post(client, content)
        assert response.status_code == 413
        assert "5 MB" in response.json()["detail"]

    def test_missing_columns_message_is_readable(self, client: TestClient) -> None:
        text = "age,gender\n25,Male\n"
        response = self._post(client, text)
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert detail.startswith("Missing required CSV columns: stress_level")
        assert "[" not in detail and "'" not in detail

    def test_export_neutralizes_formula_cells(self, client: TestClient) -> None:
        header = "participant_id," + BATCH_HEADER
        rows = "=SUM(A1),   25,Male,Low,No,5,1,1,2,8,7,100,100\n".replace("   ", "")
        rows += "@cmd,25,Male,Low,No,5,1,1,2,8,7,100,100\n-1+2,25,Male,Low,No,5,1,1,2,8,7,100,100\n"
        upload = self._post(client, header + rows)
        assert upload.status_code == 200
        export = client.get(f"/api/predict/batch/export?token={upload.json()['download_token']}")
        ids = pd.read_csv(io.BytesIO(export.content), dtype=str)["participant_id"].tolist()
        assert ids == ["'=SUM(A1)", "'@cmd", "'-1+2"]


class TestExportCacheByteCap:
    """The export cache evicts oldest entries once total stored bytes exceed the cap."""

    def test_oldest_entries_evicted_when_total_bytes_exceed_cap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(predict_mod, "MAX_CACHE_BYTES", 100)
        store_batch_export("first", b"a" * 40)
        store_batch_export("second", b"b" * 40)
        store_batch_export("third", b"c" * 40)  # 120 bytes total: "first" is evicted

        assert get_batch_export("first") is None
        assert get_batch_export("second") == b"b" * 40
        assert get_batch_export("third") == b"c" * 40
