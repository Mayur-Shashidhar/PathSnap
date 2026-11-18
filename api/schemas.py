"""Pydantic schemas for the PathSnap API."""

from typing import List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health endpoint response."""
    status: str = Field(..., description="Overall system health status: 'healthy', 'degraded', or 'unhealthy'")
    model_loaded: bool = Field(..., description="Whether the trained ML model is loaded and ready")
    model_path: str = Field(..., description="Configured path to the model artifact")
    model_type: Optional[str] = Field(None, description="Class name of the loaded model")
    pipeline_available: bool = Field(..., description="Whether P1/P3 pipeline modules are importable")
    evaluator_dir: str = Field(..., description="Configured evaluator dataset directory")
    evaluator_dir_exists: bool = Field(..., description="Whether evaluator directory exists on disk")
    evaluator_trips_count: int = Field(..., description="Number of valid evaluator trips discovered")
    version: str = Field(..., description="API version")


class TripsResponse(BaseModel):
    """Trips discovery response."""
    trips: List[str] = Field(default_factory=list, description="Available evaluator trip IDs")
    total: int = Field(..., description="Total count of available evaluator trips")
    evaluator_dir: str = Field(..., description="Configured evaluator directory path")
    status: str = Field(..., description="Status: 'available', 'empty', or 'directory_not_found'")


class EvaluationMetrics(BaseModel):
    """Server-side ground-truth evaluation metrics.
    Computed ONLY if reference route (.route) is available on the server.
    Never exposes raw reference labels or coordinates.
    """
    has_ground_truth: bool = Field(..., description="Whether ground-truth route was available for evaluation")
    top1_accuracy: Optional[float] = Field(None, description="Observation-level Top-1 Accuracy across all points (0-100%)")
    top1_accuracy_reliable: Optional[float] = Field(None, description="Observation-level Top-1 Accuracy on reliable points (0-100%)")
    macro_f1: Optional[float] = Field(None, description="Macro F1-score across road arcs (0-1.0)")
    candidate_miss_rate: Optional[float] = Field(None, description="Fraction of points where ground truth was not among candidates (0-100%)")
    candidate_miss_rate_reliable: Optional[float] = Field(None, description="Miss rate on reliable points subset (0-100%)")
    evaluated_points: Optional[int] = Field(None, description="Total number of evaluated observations")
    reliable_points: Optional[int] = Field(None, description="Number of points meeting the reliability distance threshold")
    details: Optional[str] = Field(None, description="Evaluation notes or summary description")


class PredictionResponse(BaseModel):
    """Inference and route reconstruction response."""
    trip_id: str = Field(..., description="Evaluated Trip identifier")
    status: str = Field("success", description="Prediction status")
    n_points: int = Field(..., description="Number of GPS observations")
    n_route_arcs: int = Field(..., description="Number of distinct consecutive road arcs in predicted route")
    arc_per_point: List[int] = Field(..., description="Predicted road arc index for each GPS observation")
    route_arcs: List[int] = Field(..., description="Ordered list of unique consecutive road arcs forming the route")
    route_coords: List[List[List[float]]] = Field(..., description="Coordinates of predicted route segments: [[[lat1, lon1], [lat2, lon2]], ...]")
    gps_coords: List[List[float]] = Field(..., description="Raw GPS trajectory coordinates: [[lat, lon], ...]")
    continuity_pct: float = Field(..., description="Percentage of consecutive route arcs that share a connecting node (0-100%)")
    processing_time_s: float = Field(..., description="Total inference pipeline duration in seconds")
    evaluation: Optional[EvaluationMetrics] = Field(None, description="Server-side evaluation metrics (if ground truth available)")


class ErrorResponse(BaseModel):
    """Structured error response."""
    detail: str = Field(..., description="Human-readable error description")
    error_type: str = Field(..., description="Machine-readable error category")
    trip_id: Optional[str] = Field(None, description="Trip ID associated with the error, if applicable")
