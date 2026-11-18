"""Inference service integrating the P1, P2, and P3 pipeline."""

import logging
import os
import sys
import time
from typing import Any, Dict, List

# Ensure features and hmm modules are discoverable
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEATURES_DIR = os.path.join(REPO_ROOT, "features")
HMM_DIR = os.path.join(REPO_ROOT, "hmm")

for p in (FEATURES_DIR, HMM_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# Import actual P3 inference function
from decoder import predict_route  # noqa: E402

logger = logging.getLogger("pathsnap.inference_service")


class InferenceService:
    """Coordinates end-to-end inference using the integrated P1/P2/P3 pipeline."""

    def __init__(self, model_service: Any, feature_cols: List[str], w_emit: float = 1.0):
        self.model_service = model_service
        self.feature_cols = feature_cols
        self.w_emit = w_emit

    def run_prediction(self, trip_dir: str) -> Dict[str, Any]:
        """Execute the full end-to-end inference pipeline:
        1. Loads the trip with with_route=False (no ground truth).
        2. Generates road candidates (P1).
        3. Extracts features (P1).
        4. Predicts candidate probabilities with the trained model (P2).
        5. Decodes the most likely connected route via Viterbi (P3).
        6. Returns predicted arc per point, route coordinates, and continuity.

        Uses the actual P3 function: predict_route(trip_dir, model, feature_cols, w_emit).
        """
        if not self.model_service.is_loaded:
            raise RuntimeError("Cannot run prediction: ML model is not loaded.")

        model = self.model_service.get_model()

        logger.info("Executing predict_route on %s with model %s ...", trip_dir, type(model).__name__)
        t0 = time.time()

        # Call the existing P3 function directly
        res = predict_route(
            trip_dir=trip_dir,
            model=model,
            feature_cols=self.feature_cols,
            w_emit=self.w_emit
        )

        total_duration = round(time.time() - t0, 3)
        res["processing_time_s"] = total_duration

        logger.info(
            "Prediction complete for trip %s: %d points, %d route arcs, continuity %.1f%% in %.2fs",
            res.get("trip_id"),
            res.get("n_points", 0),
            len(res.get("route_arcs", [])),
            res.get("continuity_pct", 0.0),
            total_duration
        )

        return res
