"""Pytest configuration and shared fixtures for PathSnap tests."""

import os
import shutil
import sys
import tempfile
import numpy as np
import pytest
from fastapi.testclient import TestClient

# Ensure root, features, hmm, api, and ui are on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("", "features", "hmm", "api", "ui"):
    p = os.path.join(REPO_ROOT, sub) if sub else REPO_ROOT
    if p not in sys.path:
        sys.path.insert(0, p)

from api.main import app
from api.config import settings


@pytest.fixture(scope="session")
def test_client():
    """FastAPI TestClient fixture."""
    with TestClient(app) as client:
        yield client


@pytest.fixture
def temp_evaluator_dir():
    """Create a temporary evaluator directory with a valid synthetic trip."""
    tmp_dir = tempfile.mkdtemp(prefix="pathsnap_eval_")
    tid = "eval_trip_01"
    trip_dir = os.path.join(tmp_dir, tid)
    os.makedirs(trip_dir)

    # Coordinates near Bangalore (~12.935, 77.535)
    # 4 road nodes along a connected path: 0 -> 1 -> 2 -> 3
    nodes = np.array([
        [77.5350, 12.9350],
        [77.5360, 12.9350],
        [77.5370, 12.9350],
        [77.5380, 12.9350]
    ])
    # 3 arcs: 0->1 (0), 1->2 (1), 2->3 (2)
    arcs = np.array([
        [0, 1],
        [1, 2],
        [2, 3]
    ])
    # 6 GPS track fixes along this corridor
    track = np.array([
        [77.5351, 12.93501, 1000.0],
        [77.5356, 12.93502, 1001.0],
        [77.5361, 12.93500, 1002.0],
        [77.5366, 12.93501, 1003.0],
        [77.5371, 12.93499, 1004.0],
        [77.5378, 12.93501, 1005.0]
    ])
    # Reference route: arcs 0, 1, 2
    route = np.array([0, 1, 2])

    np.savetxt(os.path.join(trip_dir, f"{tid}.nodes"), nodes, fmt="%.6f")
    np.savetxt(os.path.join(trip_dir, f"{tid}.arcs"), arcs, fmt="%d")
    np.savetxt(os.path.join(trip_dir, f"{tid}.track"), track, fmt=["%.6f", "%.6f", "%.1f"])
    np.savetxt(os.path.join(trip_dir, f"{tid}.route"), route, fmt="%d")

    yield tmp_dir, tid

    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture
def temp_evaluator_dir_no_route():
    """Create a temporary evaluator trip WITHOUT reference route (.route)."""
    tmp_dir = tempfile.mkdtemp(prefix="pathsnap_eval_noroute_")
    tid = "eval_noroute_02"
    trip_dir = os.path.join(tmp_dir, tid)
    os.makedirs(trip_dir)

    nodes = np.array([
        [77.5350, 12.9350],
        [77.5360, 12.9350],
        [77.5370, 12.9350]
    ])
    arcs = np.array([[0, 1], [1, 2]])
    track = np.array([
        [77.5351, 12.93501, 1000.0],
        [77.5355, 12.93502, 1001.0],
        [77.5365, 12.93501, 1002.0]
    ])

    np.savetxt(os.path.join(trip_dir, f"{tid}.nodes"), nodes, fmt="%.6f")
    np.savetxt(os.path.join(trip_dir, f"{tid}.arcs"), arcs, fmt="%d")
    np.savetxt(os.path.join(trip_dir, f"{tid}.track"), track, fmt=["%.6f", "%.6f", "%.1f"])

    yield tmp_dir, tid

    shutil.rmtree(tmp_dir, ignore_errors=True)
