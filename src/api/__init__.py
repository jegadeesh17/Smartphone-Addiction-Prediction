"""API routers package for Smartphone Addiction Prediction analytical platform."""

from src.api.health import router as health_router
from src.api.predict import router as predict_router

__all__ = ["health_router", "predict_router"]
