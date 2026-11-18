"""HTTP API client for communicating with the PathSnap backend on Laptop 1."""

import os
from typing import Any, Dict, List, Optional, Tuple
import requests


class PathSnapApiClient:
    """Client for the PathSnap FastAPI backend on Laptop 1."""

    def __init__(self, base_url: Optional[str] = None, timeout: int = 45):
        # Determine base URL: explicitly passed, or from env var, or default
        self.base_url = (
            base_url
            or os.getenv("PATHSNAP_API_URL", "http://localhost:8000")
        ).rstrip("/")
        self.timeout = timeout

    def set_base_url(self, base_url: str):
        """Update the base URL dynamically from the UI."""
        self.base_url = base_url.rstrip("/")

    def get_health(self) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """Call GET /health.
        Returns (success, response_dict, error_message).
        """
        url = f"{self.base_url}/health"
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                return True, resp.json(), None
            else:
                return False, {}, f"Server returned HTTP {resp.status_code}: {resp.text}"
        except requests.exceptions.ConnectionError:
            return (
                False,
                {},
                f"Connection refused to {self.base_url}. Ensure Laptop 1 backend is running and reachable on port 8000."
            )
        except requests.exceptions.Timeout:
            return False, {}, f"Health check timed out after 5s at {self.base_url}."
        except Exception as exc:
            return False, {}, f"Unexpected connection error: {exc}"

    def get_trips(self) -> Tuple[bool, List[str], Optional[str]]:
        """Call GET /trips.
        Returns (success, trips_list, error_message).
        """
        url = f"{self.base_url}/trips"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                return True, data.get("trips", []), None
            else:
                return False, [], f"Server returned HTTP {resp.status_code}: {resp.text}"
        except requests.exceptions.ConnectionError:
            return False, [], f"Cannot connect to {self.base_url} to fetch trips."
        except requests.exceptions.Timeout:
            return False, [], "Trips request timed out."
        except Exception as exc:
            return False, [], f"Error fetching trips: {exc}"

    def predict(self, trip_id: str) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """Call GET /predict/{trip_id}.
        Returns (success, prediction_dict, error_message).
        """
        url = f"{self.base_url}/predict/{trip_id}"
        try:
            resp = requests.get(url, timeout=self.timeout)
            if resp.status_code == 200:
                return True, resp.json(), None
            else:
                err_detail = resp.text
                try:
                    err_json = resp.json()
                    err_detail = err_json.get("detail", resp.text)
                except Exception:
                    pass
                return False, {}, f"Prediction failed (HTTP {resp.status_code}): {err_detail}"
        except requests.exceptions.ConnectionError:
            return False, {}, f"Connection lost to backend at {self.base_url} during prediction."
        except requests.exceptions.Timeout:
            return False, {}, f"Prediction timed out after {self.timeout}s."
        except Exception as exc:
            return False, {}, f"Error during prediction request: {exc}"
