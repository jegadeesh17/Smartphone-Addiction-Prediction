"""Inference API endpoints for live tabular risk evaluation and diagnostics."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.config import Settings, get_settings
from src.inference import get_inference_engine
from src.schemas import BehavioralProfileInput, PredictionResponse

router = APIRouter(tags=["Inference"])


@router.post("/api/predict", response_model=PredictionResponse)
@router.post("/predict", response_model=PredictionResponse)
def predict(
    profile: BehavioralProfileInput,
    settings: Settings = Depends(get_settings),
) -> PredictionResponse:
    """Evaluate single-row behavioral profile with sub-20ms latency SLA.

    Constructs 74-feature vector using cached population priors and returns
    PredictionResponse populated with probability, classification, 3-tier
    clinical spectrum status_label, behavioral ratios, and recommended interventions.
    """
    engine = get_inference_engine(
        use_mock=settings.USE_MOCK_MODEL,
        model_path=settings.MODEL_PATH,
        priors_path=settings.PRIORS_PATH,
    )
    return engine.predict(profile)
