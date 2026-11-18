"""Trips discovery endpoint definition."""

import logging
from fastapi import APIRouter, Depends
from api.schemas import TripsResponse
from api.dependencies import get_trip_service
from api.services.trip_service import TripService

router = APIRouter(tags=["Trips"])
logger = logging.getLogger("pathsnap.routes.trips")


@router.get("/trips", response_model=TripsResponse)
def get_evaluator_trips(
    trip_service: TripService = Depends(get_trip_service)
) -> TripsResponse:
    """Return available evaluator trip IDs only.

    Strict rules:
    - Never returns training or student-test trips.
    - If the evaluator directory does not exist or contains no valid trips,
      returns an honest empty list with appropriate status.
    - Validates that each trip folder has all required files (.track, .nodes, .arcs).
    """
    if not trip_service.is_directory_available():
        logger.info("Evaluator directory %s does not exist.", trip_service.evaluator_dir)
        return TripsResponse(
            trips=[],
            total=0,
            evaluator_dir=trip_service.evaluator_dir,
            status="directory_not_found"
        )

    trips = trip_service.list_evaluator_trips()
    status = "available" if len(trips) > 0 else "empty"

    logger.info("Discovered %d evaluator trips in %s", len(trips), trip_service.evaluator_dir)

    return TripsResponse(
        trips=trips,
        total=len(trips),
        evaluator_dir=trip_service.evaluator_dir,
        status=status
    )
