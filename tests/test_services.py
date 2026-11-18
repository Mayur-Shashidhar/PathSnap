"""Unit tests for individual API services."""

import os
import pytest
from api.services.model_service import ModelService
from api.services.trip_service import TripService
from api.services.evaluation_service import EvaluationService
from api.services.map_service import MapService
from api.config import settings


def test_model_service_load_real_model():
    """Test loading the real LightGBM model artifact."""
    svc = ModelService(model_path=settings.MODEL_PATH, feature_cols=settings.FEATURE_COLS)
    assert svc.load() is True
    assert svc.is_loaded is True
    assert svc.get_model() is not None
    assert svc.model_info["model_type"] == "LGBMClassifier"


def test_model_service_missing_file():
    """Test ModelService fails cleanly with a nonexistent model path."""
    svc = ModelService(model_path="/fake/path/model.joblib", feature_cols=[])
    assert svc.load() is False
    assert svc.is_loaded is False
    assert "not found" in svc.load_error.lower()
    with pytest.raises(RuntimeError):
        svc.get_model()


def test_trip_service_validation():
    """Test TripService validation rules."""
    svc = TripService(evaluator_dir="/tmp")

    assert svc.validate_trip_id("00000001") is True
    assert svc.validate_trip_id("trip_test-01") is True
    assert svc.validate_trip_id("../trip") is False
    assert svc.validate_trip_id("trip/sub") is False
    assert svc.validate_trip_id("trip\\sub") is False
    assert svc.validate_trip_id("") is False
    assert svc.validate_trip_id("   ") is False


def test_map_service_bounds():
    """Test bounding box and center calculation in MapService."""
    coords = [[12.935, 77.535], [12.936, 77.537]]
    min_b, max_b = MapService.calculate_bounds(coords)
    assert min_b == [12.935, 77.535]
    assert max_b == [12.936, 77.537]

    center = MapService.calculate_center(coords)
    assert round(center[0], 4) == 12.9355
    assert round(center[1], 4) == 77.5360


def test_evaluation_service_metrics(temp_evaluator_dir):
    """Test server-side metric calculation with EvaluationService."""
    tmp_dir, tid = temp_evaluator_dir
    trip_dir = os.path.join(tmp_dir, tid)

    eval_svc = EvaluationService(reliable_dist_threshold=100.0)

    # Simulated prediction with matching arcs [0, 0, 1, 1, 2, 2]
    pred_res = {
        "arc_per_point": [0, 0, 1, 1, 2, 2],
        "n_points": 6
    }

    metrics = eval_svc.evaluate(trip_dir, pred_res)
    assert metrics is not None
    assert metrics["has_ground_truth"] is True
    assert metrics["top1_accuracy"] is not None
    assert metrics["macro_f1"] is not None
    assert metrics["evaluated_points"] == 6
