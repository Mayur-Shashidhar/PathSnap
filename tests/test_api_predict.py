"""Integration and unit tests for the /predict/{trip_id} endpoint."""

import pytest
from fastapi.testclient import TestClient
from api.main import app
from api.dependencies import get_trip_service, get_model_service
from api.services.trip_service import TripService
from api.services.model_service import ModelService


def test_predict_invalid_trip_id_format(test_client: TestClient):
    """Test that path traversal attempts or invalid IDs are rejected with HTTP 400."""
    # Path traversal attempt
    resp = test_client.get("/predict/..%2F..%2Fetc%2Fpasswd")
    assert resp.status_code in (400, 404)

    resp = test_client.get("/predict/trip@#$!")
    assert resp.status_code == 400
    assert "Invalid trip ID" in resp.json()["detail"]


def test_predict_trip_not_found(test_client: TestClient, temp_evaluator_dir):
    """Test requesting a nonexistent trip returns HTTP 404."""
    tmp_dir, _ = temp_evaluator_dir
    service = TripService(evaluator_dir=tmp_dir)
    app.dependency_overrides[get_trip_service] = lambda: service

    try:
        resp = test_client.get("/predict/nonexistent_trip_999")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()
    finally:
        app.dependency_overrides.clear()


def test_predict_model_unavailable(test_client: TestClient, temp_evaluator_dir):
    """Test that 503 is returned if model is not loaded."""
    tmp_dir, tid = temp_evaluator_dir
    service = TripService(evaluator_dir=tmp_dir)
    unloaded_model_service = ModelService(model_path="/missing.joblib", feature_cols=[])
    unloaded_model_service.load()

    app.dependency_overrides[get_trip_service] = lambda: service
    app.dependency_overrides[get_model_service] = lambda: unloaded_model_service

    try:
        resp = test_client.get(f"/predict/{tid}")
        assert resp.status_code == 503
        assert "Model unavailable" in resp.json()["detail"]
    finally:
        app.dependency_overrides.clear()


def test_predict_real_pipeline_success(test_client: TestClient, temp_evaluator_dir):
    """Test full real end-to-end prediction exercising P1/P2/P3 with the actual trained model."""
    tmp_dir, tid = temp_evaluator_dir
    service = TripService(evaluator_dir=tmp_dir)
    app.dependency_overrides[get_trip_service] = lambda: service

    try:
        resp = test_client.get(f"/predict/{tid}")
        assert resp.status_code == 200
        data = resp.json()

        assert data["trip_id"] == tid
        assert data["status"] == "success"
        assert data["n_points"] == 6
        assert len(data["arc_per_point"]) == 6
        assert len(data["gps_coords"]) == 6
        assert len(data["route_arcs"]) >= 1
        assert len(data["route_coords"]) >= 1
        assert data["continuity_pct"] > 0.0
        assert data["processing_time_s"] > 0.0

        # Evaluation metrics should be populated because .route was present
        eval_metrics = data["evaluation"]
        assert eval_metrics is not None
        assert eval_metrics["has_ground_truth"] is True
        assert eval_metrics["top1_accuracy"] is not None
        assert eval_metrics["top1_accuracy"] >= 0.0
        assert eval_metrics["macro_f1"] is not None
        assert eval_metrics["candidate_miss_rate"] is not None
    finally:
        app.dependency_overrides.clear()


def test_predict_without_ground_truth(test_client: TestClient, temp_evaluator_dir_no_route):
    """Test prediction on evaluator trip where reference route is unavailable."""
    tmp_dir, tid = temp_evaluator_dir_no_route
    service = TripService(evaluator_dir=tmp_dir)
    app.dependency_overrides[get_trip_service] = lambda: service

    try:
        resp = test_client.get(f"/predict/{tid}")
        assert resp.status_code == 200
        data = resp.json()

        assert data["trip_id"] == tid
        assert data["n_points"] == 3
        assert len(data["arc_per_point"]) == 3

        # Evaluation should honestly report ground truth not available
        eval_metrics = data["evaluation"]
        assert eval_metrics is not None
        assert eval_metrics["has_ground_truth"] is False
        assert eval_metrics["top1_accuracy"] is None
        assert eval_metrics["macro_f1"] is None
    finally:
        app.dependency_overrides.clear()
