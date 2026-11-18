"""Map and geometry helper service."""

from typing import Any, Dict, List, Tuple


class MapService:
    """Helper for geometry bounding boxes and coordinate summaries."""

    @staticmethod
    def calculate_bounds(coords: List[List[float]]) -> Tuple[List[float], List[float]]:
        """Calculate [[min_lat, min_lon], [max_lat, max_lon]] from a list of [lat, lon]."""
        if not coords:
            return [0.0, 0.0], [0.0, 0.0]
        lats = [c[0] for c in coords]
        lons = [c[1] for c in coords]
        return [min(lats), min(lons)], [max(lats), max(lons)]

    @staticmethod
    def calculate_center(coords: List[List[float]]) -> List[float]:
        """Calculate [mean_lat, mean_lon] for initial map centering."""
        if not coords:
            return [0.0, 0.0]
        mean_lat = sum(c[0] for c in coords) / len(coords)
        mean_lon = sum(c[1] for c in coords) / len(coords)
        return [round(mean_lat, 6), round(mean_lon, 6)]
