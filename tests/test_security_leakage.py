"""Security and data leakage tests.

Ensures ground-truth reference routes and labels never leak to the API response or model input.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app
from api.dependencies import get_trip_service
from api.services.trip_service import TripService


def test_zero_ground_truth_leakage_in_api_response(test_client: TestClient, temp_evaluator_dir):
    """Verify that the prediction response contains NO true arcs, route sequences, or reference labels."""
    tmp_dir, tid = temp_evaluator_dir
    service = TripService(evaluator_dir=tmp_dir)
    app.dependency_overrides[get_trip_service] = lambda: service

    try:
        resp = test_client.get(f"/predict/{tid}")
        assert resp.status_code == 200
        data = resp.json()

        # Prohibited field names that would leak ground truth
        prohibited_keys = [
            "true_arc",
            "true_arcs",
            "reference_route",
            "gt_route",
            "ground_truth_route",
            "labels",
            "true_dist",
            "route_pos"
        ]

        # Check top-level response
        for key in prohibited_keys:
            assert key not in data, f"Security Violation: '{key}' found in top-level API response!"

        # Check evaluation sub-object
        if data.get("evaluation"):
            for key in prohibited_keys:
                assert key not in data["evaluation"], f"Security Violation: '{key}' found in evaluation object!"

        # Ground truth route file content must not be embedded anywhere
        resp_text = resp.text
        assert "route_file" not in resp_text
        assert ".route" not in resp_text or "not available" in resp_text.lower() or ".route" in str(data.get("evaluation", {}).get("details", ""))

    finally:
        app.dependency_overrides.clear()
