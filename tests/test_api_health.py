"""Unit tests for the /health endpoint."""

import pytest
from fastapi.testclient import TestClient
from api.services.model_service import ModelService
from api.dependencies import get_model_service


def test_health_success(test_client: TestClient):
    """Test that /health returns 200 and accurate system information."""
    resp = test_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()

    assert "status" in data
    assert "model_loaded" in data
    assert "pipeline_available" in data
    assert "evaluator_dir" in data
    assert "evaluator_trips_count" in data
    assert "version" in data

    # Model and pipeline should be loaded with the actual LightGBM baseline artifact
    assert data["model_loaded"] is True
    assert data["pipeline_available"] is True
    assert data["status"] == "healthy"
    assert data["model_type"] == "LGBMClassifier"


def test_health_reports_degraded_when_model_missing(test_client: TestClient):
    """Test that /health truthfully reports degraded when the model is not loaded."""
    # Create a dummy model service with invalid path
    unloaded_service = ModelService(model_path="/nonexistent/model.joblib", feature_cols=[])
    unloaded_service.load()

    from api.main import app
    app.dependency_overrides[get_model_service] = lambda: unloaded_service

    try:
        resp = test_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_loaded"] is False
        assert data["status"] == "degraded"
        assert data["model_type"] is None
    finally:
        app.dependency_overrides.clear()
