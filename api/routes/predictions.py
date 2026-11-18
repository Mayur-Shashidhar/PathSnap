"""Prediction endpoint definition."""

import logging
from fastapi import APIRouter, Depends, HTTPException, status
from api.schemas import PredictionResponse, EvaluationMetrics, ErrorResponse
from api.dependencies import (
    get_trip_service,
    get_model_service,
    get_inference_service,
    get_evaluation_service
)
from api.services.trip_service import TripService
from api.services.model_service import ModelService
from api.services.inference_service import InferenceService
from api.services.evaluation_service import EvaluationService

router = APIRouter(tags=["Predictions"])
logger = logging.getLogger("pathsnap.routes.predictions")


@router.get(
    "/predict/{trip_id}",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid trip ID format"},
        404: {"model": ErrorResponse, "description": "Trip not found or missing required files"},
        503: {"model": ErrorResponse, "description": "Trained ML model unavailable"},
        500: {"model": ErrorResponse, "description": "Inference or processing error"}
    }
)
def predict_trip(
    trip_id: str,
    trip_service: TripService = Depends(get_trip_service),
    model_service: ModelService = Depends(get_model_service),
    inference_service: InferenceService = Depends(get_inference_service),
    evaluation_service: EvaluationService = Depends(get_evaluation_service)
) -> PredictionResponse:
    """Execute end-to-end map matching and route reconstruction for an evaluator trip.

    Workflow:
    1. Validate trip ID and ensure no path traversal.
    2. Resolve trip directory in the configured evaluator data directory.
    3. Verify required files (.track, .nodes, .arcs).
    4. Verify ML model readiness.
    5. Run inference WITHOUT ground truth (with_route=False).
    6. Compute server-side evaluation metrics strictly after prediction is complete.
    7. Return predicted route geometry, continuity, and evaluation summary.
    """
    logger.info("Received prediction request for trip_id: %s", trip_id)

    # 1 & 2. Validate trip ID and resolve directory
    trip_dir, err_msg = trip_service.resolve_trip_dir(trip_id)
    if err_msg is not None:
        if "Invalid trip ID" in err_msg or "Access denied" in err_msg:
            logger.warning("Rejected invalid trip_id '%s': %s", trip_id, err_msg)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=err_msg
            )
        else:
            logger.warning("Trip '%s' resolution failed: %s", trip_id, err_msg)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=err_msg
            )

    # 4. Check model availability
    if not model_service.is_loaded:
        load_err = model_service.load_error or "Trained ML model is not available on this server."
        logger.error("Model unavailable during predict request: %s", load_err)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model unavailable: {load_err}"
        )

    # 5. Run inference pipeline (loads trip with with_route=False)
    try:
        prediction_res = inference_service.run_prediction(trip_dir)
    except Exception as exc:
        logger.exception("Inference failed for trip %s: %s", trip_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Map-matching inference failed for trip '{trip_id}'. Check server logs."
        )

    # 6. Post-inference server-side evaluation (only if .route is present)
    eval_dict = None
    try:
        eval_dict = evaluation_service.evaluate(trip_dir, prediction_res)
    except Exception as exc:
        logger.warning("Evaluation failed post-inference for trip %s: %s", trip_id, exc)
        eval_dict = {
            "has_ground_truth": False,
            "details": f"Post-inference evaluation encountered an error: {exc}"
        }

    evaluation_metrics = EvaluationMetrics(**eval_dict) if eval_dict else None

    # 7. Construct final response
    return PredictionResponse(
        trip_id=trip_id,
        status="success",
        n_points=prediction_res.get("n_points", 0),
        n_route_arcs=len(prediction_res.get("route_arcs", [])),
        arc_per_point=prediction_res.get("arc_per_point", []),
        route_arcs=prediction_res.get("route_arcs", []),
        route_coords=prediction_res.get("route_coords", []),
        gps_coords=prediction_res.get("gps_coords", []),
        continuity_pct=prediction_res.get("continuity_pct", 0.0),
        processing_time_s=prediction_res.get("processing_time_s", 0.0),
        evaluation=evaluation_metrics
    )
