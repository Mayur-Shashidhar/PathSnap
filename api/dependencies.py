"""FastAPI dependency injection providers."""

import functools
from typing import Generator
from api.config import settings, Settings
from api.services.model_service import ModelService
from api.services.trip_service import TripService
from api.services.inference_service import InferenceService
from api.services.evaluation_service import EvaluationService
from api.services.map_service import MapService

# Singleton instances created once and reused across requests
_model_service: ModelService | None = None
_trip_service: TripService | None = None
_inference_service: InferenceService | None = None
_evaluation_service: EvaluationService | None = None


def get_settings() -> Settings:
    """Return the application settings."""
    return settings


def get_model_service() -> ModelService:
    """Return the singleton model service."""
    global _model_service
    if _model_service is None:
        _model_service = ModelService(
            model_path=settings.MODEL_PATH,
            feature_cols=settings.FEATURE_COLS
        )
    return _model_service


def get_trip_service() -> TripService:
    """Return the trip service configured with evaluator directory."""
    global _trip_service
    if _trip_service is None or _trip_service.evaluator_dir != settings.EVALUATOR_DATA_DIR:
        _trip_service = TripService(evaluator_dir=settings.EVALUATOR_DATA_DIR)
    return _trip_service


def get_inference_service() -> InferenceService:
    """Return the inference service coordinating P1/P2/P3."""
    global _inference_service
    if _inference_service is None:
        model_svc = get_model_service()
        _inference_service = InferenceService(
            model_service=model_svc,
            feature_cols=settings.FEATURE_COLS,
            w_emit=settings.W_EMIT
        )
    return _inference_service


def get_evaluation_service() -> EvaluationService:
    """Return the server-side evaluation service."""
    global _evaluation_service
    if _evaluation_service is None:
        _evaluation_service = EvaluationService(
            reliable_dist_threshold=settings.RELIABLE_DIST_THRESHOLD
        )
    return _evaluation_service


def get_map_service() -> MapService:
    """Return the map service helper."""
    return MapService()
