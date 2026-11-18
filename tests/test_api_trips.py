"""Unit tests for the /trips endpoint."""

import os
import pytest
from fastapi.testclient import TestClient
from api.main import app
from api.dependencies import get_trip_service
from api.services.trip_service import TripService


def test_trips_nonexistent_directory(test_client: TestClient):
    """Test /trips when evaluator directory does not exist."""
    fake_service = TripService(evaluator_dir="/path/does/not/exist")
    app.dependency_overrides[get_trip_service] = lambda: fake_service

    try:
        resp = test_client.get("/trips")
        assert resp.status_code == 200
        data = resp.json()
        assert data["trips"] == []
        assert data["total"] == 0
        assert data["status"] == "directory_not_found"
    finally:
        app.dependency_overrides.clear()


def test_trips_with_valid_directory(test_client: TestClient, temp_evaluator_dir):
    """Test /trips returns valid evaluator trips when present."""
    tmp_dir, tid = temp_evaluator_dir
    service = TripService(evaluator_dir=tmp_dir)
    app.dependency_overrides[get_trip_service] = lambda: service

    try:
        resp = test_client.get("/trips")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert tid in data["trips"]
        assert data["status"] == "available"
    finally:
        app.dependency_overrides.clear()


def test_trips_skips_invalid_folders(test_client: TestClient, temp_evaluator_dir):
    """Test /trips skips incomplete folders (e.g. missing .arcs)."""
    tmp_dir, tid = temp_evaluator_dir
    # Create an incomplete trip folder
    bad_dir = os.path.join(tmp_dir, "bad_trip_99")
    os.makedirs(bad_dir)
    with open(os.path.join(bad_dir, "bad_trip_99.track"), "w") as f:
        f.write("0 0 0\n")

    service = TripService(evaluator_dir=tmp_dir)
    app.dependency_overrides[get_trip_service] = lambda: service

    try:
        resp = test_client.get("/trips")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert "bad_trip_99" not in data["trips"]
        assert tid in data["trips"]
    finally:
        app.dependency_overrides.clear()
