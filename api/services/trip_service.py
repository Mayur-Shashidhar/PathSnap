"""Trip service for discovering and validating evaluator trips."""

import logging
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger("pathsnap.trip_service")

# Trip ID validation: alphanumeric string, underscores or hyphens, typically 8 hex digits or characters
TRIP_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-]+$")


class TripService:
    """Manages discovery, validation, and resolution of evaluator trips."""

    def __init__(self, evaluator_dir: str):
        self.evaluator_dir = os.path.abspath(evaluator_dir)

    def is_directory_available(self) -> bool:
        """Check if the evaluator directory exists and is accessible."""
        return os.path.isdir(self.evaluator_dir)

    def validate_trip_id(self, trip_id: str) -> bool:
        """Check whether the trip ID matches safe naming patterns and prevents path traversal."""
        if not trip_id or not isinstance(trip_id, str):
            return False
        if ".." in trip_id or "/" in trip_id or "\\" in trip_id:
            return False
        return bool(TRIP_ID_PATTERN.match(trip_id))

    def validate_trip_files(self, trip_dir: str, trip_id: str) -> Tuple[bool, List[str]]:
        """Validate that the trip directory contains the required P1 files.
        Required: .track, .nodes, .arcs
        Optional: .route (ground truth)
        """
        required_exts = ["track", "nodes", "arcs"]
        missing = []
        for ext in required_exts:
            file_path = os.path.join(trip_dir, f"{trip_id}.{ext}")
            if not os.path.isfile(file_path):
                missing.append(f"{trip_id}.{ext}")

        return len(missing) == 0, missing

    def has_ground_truth(self, trip_id: str) -> bool:
        """Check if server-side ground truth (.route) is available for this trip."""
        trip_dir = os.path.join(self.evaluator_dir, trip_id)
        route_path = os.path.join(trip_dir, f"{trip_id}.route")
        return os.path.isfile(route_path)

    def list_evaluator_trips(self) -> List[str]:
        """Scan the configured evaluator directory and return all valid trip IDs.
        Returns empty list if directory is missing or empty.
        Never substitutes training or test trips.
        """
        if not self.is_directory_available():
            logger.warning(
                "Evaluator data directory does not exist: %s",
                self.evaluator_dir
            )
            return []

        valid_trips = []
        try:
            entries = sorted(os.listdir(self.evaluator_dir))
        except OSError as exc:
            logger.error("Failed to read evaluator directory %s: %exc", self.evaluator_dir, exc)
            return []

        for entry in entries:
            trip_path = os.path.join(self.evaluator_dir, entry)
            if not os.path.isdir(trip_path):
                continue

            if not self.validate_trip_id(entry):
                continue

            is_valid, missing = self.validate_trip_files(trip_path, entry)
            if is_valid:
                valid_trips.append(entry)
            else:
                logger.debug(
                    "Trip %s skipped in evaluator dir: missing required files %s",
                    entry,
                    missing
                )

        return sorted(valid_trips)

    def resolve_trip_dir(self, trip_id: str) -> Tuple[Optional[str], Optional[str]]:
        """Resolve and validate the directory for a requested trip ID.
        Returns (trip_dir, error_message).
        If resolution fails, trip_dir is None and error_message explains why.
        """
        if not self.validate_trip_id(trip_id):
            return None, f"Invalid trip ID format: '{trip_id}'. Must be alphanumeric without path separators."

        if not self.is_directory_available():
            return None, f"Evaluator dataset directory not found at: {self.evaluator_dir}"

        trip_dir = os.path.abspath(os.path.join(self.evaluator_dir, trip_id))

        # Enforce that resolved path stays within evaluator_dir
        if not trip_dir.startswith(self.evaluator_dir):
            return None, f"Access denied: resolved path is outside evaluator directory."

        if not os.path.isdir(trip_dir):
            return None, f"Trip '{trip_id}' not found in evaluator directory."

        is_valid, missing = self.validate_trip_files(trip_dir, trip_id)
        if not is_valid:
            return None, f"Trip '{trip_id}' is missing required files: {', '.join(missing)}"

        return trip_dir, None
