"""Server-side ground-truth evaluation service.

Performs post-prediction evaluation strictly on the server side.
Ensures zero data leakage:
- Reference route is never loaded during inference.
- Ground truth is never returned to the frontend or caller.
- Returns only aggregate summary metrics.
"""

import logging
import os
import sys
from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Ensure features module is discoverable
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEATURES_DIR = os.path.join(REPO_ROOT, "features")
if FEATURES_DIR not in sys.path:
    sys.path.insert(0, FEATURES_DIR)

from loaders import load_trip  # noqa: E402
from candidates import generate, align_route  # noqa: E402

logger = logging.getLogger("pathsnap.evaluation_service")


class EvaluationService:
    """Calculates server-side evaluation metrics after prediction is complete."""

    def __init__(self, reliable_dist_threshold: float = 100.0):
        self.reliable_dist_threshold = reliable_dist_threshold

    def evaluate(self, trip_dir: str, prediction_res: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Evaluate predicted route against reference route if .route is available.
        Called strictly AFTER prediction is done.

        Returns EvaluationMetrics dict if .route exists, or unavailable indicator if not.
        Never returns true_arc or route sequences.
        """
        tid = os.path.basename(os.path.normpath(trip_dir))
        route_file = os.path.join(trip_dir, f"{tid}.route")

        if not os.path.isfile(route_file):
            logger.info("No server-side ground truth (.route) found for trip %s. Metrics marked unavailable.", tid)
            return {
                "has_ground_truth": False,
                "top1_accuracy": None,
                "top1_accuracy_reliable": None,
                "macro_f1": None,
                "candidate_miss_rate": None,
                "candidate_miss_rate_reliable": None,
                "evaluated_points": None,
                "reliable_points": None,
                "details": "Server-side ground truth (.route) not available for this evaluator trip."
            }

        try:
            logger.info("Loading reference route for post-prediction evaluation of trip %s ...", tid)
            # Load trip with reference route for independent evaluation
            trip_ref = load_trip(trip_dir, with_route=True)
            if trip_ref.route is None or len(trip_ref.route) == 0:
                return {
                    "has_ground_truth": False,
                    "top1_accuracy": None,
                    "top1_accuracy_reliable": None,
                    "macro_f1": None,
                    "candidate_miss_rate": None,
                    "candidate_miss_rate_reliable": None,
                    "evaluated_points": None,
                    "reliable_points": None,
                    "details": "Reference route file is empty."
                }

            # Align reference route to derive per-point ground truth arcs
            true_arc, route_pos, true_dist = align_route(trip_ref)
            pred_arc = np.array(prediction_res.get("arc_per_point", []))

            n_points = len(pred_arc)
            if n_points != len(true_arc):
                logger.error("Length mismatch: pred_arc has %d, true_arc has %d points.", n_points, len(true_arc))
                return {
                    "has_ground_truth": False,
                    "top1_accuracy": None,
                    "top1_accuracy_reliable": None,
                    "macro_f1": None,
                    "candidate_miss_rate": None,
                    "candidate_miss_rate_reliable": None,
                    "evaluated_points": n_points,
                    "reliable_points": None,
                    "details": f"Point count mismatch between prediction ({n_points}) and ground truth ({len(true_arc)})."
                }

            # Reliable points subset
            reliable = true_dist <= self.reliable_dist_threshold
            n_reliable = int(np.sum(reliable))

            # Observation-level Top-1 Accuracy
            top1_acc = round(float(100.0 * accuracy_score(true_arc, pred_arc)), 2)

            top1_acc_reliable = None
            if n_reliable > 0:
                top1_acc_reliable = round(float(100.0 * accuracy_score(true_arc[reliable], pred_arc[reliable])), 2)

            # Observation-level Macro F1 across arcs
            macro_f1 = round(float(f1_score(true_arc, pred_arc, average="macro", zero_division=0)), 4)

            # Candidate miss rate evaluation
            cand, _ = generate(trip_ref)
            cand_by_point = cand.groupby("point_idx")["arc_id"].apply(set).to_dict()

            hit = np.zeros(n_points, dtype=bool)
            for i in range(n_points):
                cands_for_pt = cand_by_point.get(i, set())
                hit[i] = true_arc[i] in cands_for_pt

            miss_rate = round(float(100.0 * (1.0 - np.mean(hit))), 2)

            miss_rate_reliable = None
            if n_reliable > 0:
                miss_rate_reliable = round(float(100.0 * (1.0 - np.mean(hit[reliable]))), 2)

            logger.info(
                "Evaluation complete for trip %s: Top-1 Acc=%.2f%% (Reliable=%.2f%%), Macro F1=%.4f, Miss Rate=%.2f%%",
                tid,
                top1_acc,
                top1_acc_reliable if top1_acc_reliable is not None else 0.0,
                macro_f1,
                miss_rate
            )

            return {
                "has_ground_truth": True,
                "top1_accuracy": top1_acc,
                "top1_accuracy_reliable": top1_acc_reliable,
                "macro_f1": macro_f1,
                "candidate_miss_rate": miss_rate,
                "candidate_miss_rate_reliable": miss_rate_reliable,
                "evaluated_points": n_points,
                "reliable_points": n_reliable,
                "details": f"Evaluated against reference route ({n_points} observations, {n_reliable} reliable)."
            }

        except Exception as exc:
            logger.exception("Server-side evaluation failed for trip %s: %s", tid, exc)
            return {
                "has_ground_truth": False,
                "top1_accuracy": None,
                "top1_accuracy_reliable": None,
                "macro_f1": None,
                "candidate_miss_rate": None,
                "candidate_miss_rate_reliable": None,
                "evaluated_points": len(prediction_res.get("arc_per_point", [])),
                "reliable_points": None,
                "details": f"Evaluation error: {exc}"
            }
