"""PathSnap FastAPI Application (Laptop 1).

Serves the map-matching inference pipeline and server-side evaluation.
"""

from contextlib import asynccontextmanager
import logging
import sys
import time
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.config import settings
from api.dependencies import get_model_service, get_trip_service
from api.routes import health, trips, predictions

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("pathsnap.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler: loads model and validates configuration on startup."""
    logger.info("==================================================")
    logger.info("Starting PathSnap API Server (Laptop 1)...")
    logger.info("Host: %s | Port: %d", settings.HOST, settings.PORT)
    logger.info("Model Path: %s", settings.MODEL_PATH)
    logger.info("Evaluator Dir: %s", settings.EVALUATOR_DATA_DIR)
    logger.info("==================================================")

    # Pre-load ML model on startup
    model_service = get_model_service()
    success = model_service.load()
    if success:
        logger.info("Model startup loading: SUCCESS (%s)", model_service.model_info.get("model_type"))
    else:
        logger.warning(
            "Model startup loading: FAILED - %s. API will report degraded status.",
            model_service.load_error
        )

    # Check evaluator directory
    trip_service = get_trip_service()
    if trip_service.is_directory_available():
        count = len(trip_service.list_evaluator_trips())
        logger.info("Evaluator directory mounted: found %d valid evaluator trips.", count)
    else:
        logger.info("Evaluator directory '%s' not present yet (waiting for mount).", settings.EVALUATOR_DATA_DIR)

    yield

    logger.info("PathSnap API Server shutting down...")


app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    lifespan=lifespan
)

# CORS middleware for cross-laptop communication (Laptop 2 -> Laptop 1)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests_middleware(request: Request, call_next):
    """Log incoming requests with timestamp, client IP, path, status, and duration."""
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"
    method = request.method
    path = request.url.path

    # Extract trip_id if this is a prediction request
    trip_id = None
    if path.startswith("/predict/"):
        parts = path.split("/")
        if len(parts) >= 3:
            trip_id = parts[2]

    try:
        response = await call_next(request)
        duration_ms = round((time.time() - start_time) * 1000, 2)
        status_code = response.status_code

        if trip_id:
            logger.info(
                "[PREDICTION REQUEST] IP=%s | TripID=%s | Status=%d | Latency=%.2fms",
                client_ip, trip_id, status_code, duration_ms
            )
        else:
            logger.info(
                "%s %s - %d (%s, %.2fms)",
                method, path, status_code, client_ip, duration_ms
            )

        return response

    except Exception as exc:
        duration_ms = round((time.time() - start_time) * 1000, 2)
        logger.exception(
            "[REQUEST FAILED] %s %s from %s after %.2fms: %s",
            method, path, client_ip, duration_ms, exc
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "Internal server error occurred processing request.",
                "error_type": "internal_error",
                "trip_id": trip_id
            }
        )


# Mount API routes
app.include_router(health.router)
app.include_router(trips.router)
app.include_router(predictions.router)


@app.get("/", tags=["Root"])
def root():
    """Root info endpoint."""
    return {
        "name": settings.API_TITLE,
        "version": settings.API_VERSION,
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "trips": "/trips",
            "predict": "/predict/{trip_id}"
        }
    }
