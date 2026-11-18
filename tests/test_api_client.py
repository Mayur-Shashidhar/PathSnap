"""Unit tests for the Streamlit UI API client."""

import pytest
from unittest.mock import patch, MagicMock
from ui.api_client import PathSnapApiClient


def test_api_client_url_handling():
    """Test client URL formatting and setter."""
    client = PathSnapApiClient(base_url="http://192.168.1.50:8000/")
    assert client.base_url == "http://192.168.1.50:8000"

    client.set_base_url("http://laptop1:8080/")
    assert client.base_url == "http://laptop1:8080"


@patch("requests.get")
def test_api_client_get_health(mock_get):
    """Test get_health method parses response properly."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"status": "healthy", "model_loaded": True}
    mock_get.return_value = mock_resp

    client = PathSnapApiClient(base_url="http://localhost:8000")
    ok, data, err = client.get_health()

    assert ok is True
    assert data["status"] == "healthy"
    assert err is None


@patch("requests.get")
def test_api_client_get_trips(mock_get):
    """Test get_trips method handles trip list."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"trips": ["00000001", "00000002"], "total": 2}
    mock_get.return_value = mock_resp

    client = PathSnapApiClient(base_url="http://localhost:8000")
    ok, trips, err = client.get_trips()

    assert ok is True
    assert trips == ["00000001", "00000002"]
    assert err is None


@patch("requests.get")
def test_api_client_predict_error(mock_get):
    """Test predict handles error responses gracefully."""
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_resp.json.return_value = {"detail": "Trip not found"}
    mock_get.return_value = mock_resp

    client = PathSnapApiClient(base_url="http://localhost:8000")
    ok, data, err = client.predict("fake_trip")

    assert ok is False
    assert "Trip not found" in err
