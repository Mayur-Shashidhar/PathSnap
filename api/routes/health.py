"""Health endpoint definition."""

import logging
from fastapi import APIRouter, Depends
from api.config import Settings
from api.schemas import HealthResponse
from api.dependencies import get_settings, get_model_service, get_trip_service
from api.services.model_service import ModelService
from api.services.trip_service import TripService

router = APIRouter(tags=["Health"])
logger = logging.getLogger("pathsnap.routes.health")


@router.get("/health", response_model=HealthResponse)
def get_health(
    settings: Settings = Depends(get_settings),
    model_service: ModelService = Depends(get_model_service),
    trip_service: TripService = Depends(get_trip_service)
) -> HealthResponse:
    """Return system health status, model loading state, and evaluator trip availability.
    Truthfully reports component status without disguising missing models or data.
    """
    model_loaded = model_service.is_loaded
    model_type = model_service.model_info.get("model_type") if model_loaded else None

    # Check whether pipeline modules are available
    try:
        import loaders  # noqa: F401
        import candidates  # noqa: F401
        import features  # noqa: F401
        import decoder  # noqa: F401
        pipeline_available = True
    except ImportError as e:
        logger.warning("Pipeline modules import check failed: %s", e)
        pipeline_available = False

    evaluator_dir_exists = trip_service.is_directory_available()
    evaluator_trips = trip_service.list_evaluator_trips() if evaluator_dir_exists else []

    # Overall status assessment
    if model_loaded and pipeline_available:
        status = "healthy"
    elif pipeline_available:
        status = "degraded"  # model missing or failed to load
    else:
        status = "unhealthy"

    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        model_path=settings.MODEL_PATH,
        model_type=model_type,
        pipeline_available=pipeline_available,
        evaluator_dir=settings.EVALUATOR_DATA_DIR,
        evaluator_dir_exists=evaluator_dir_exists,
        evaluator_trips_count=len(evaluator_trips),
        version=settings.API_VERSION
    )
