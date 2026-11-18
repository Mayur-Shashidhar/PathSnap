import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

EARTH_R = 6371008.8
RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")


@dataclass
class Trip:
    trip_id: str
    track: pd.DataFrame      # lon, lat, t, x, y
    nodes: pd.DataFrame      # lon, lat, x, y
    arcs: pd.DataFrame       # u, v, length, bearing
    route: np.ndarray | None
    lon0: float
    lat0: float


def _read(path, names, dtype=float):
    return pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype=dtype, engine="c")


def to_xy(lon, lat, lon0, lat0):
    # equirectangular around the trip centre; trips are a few km wide so error is negligible
    x = np.radians(np.asarray(lon) - lon0) * np.cos(np.radians(lat0)) * EARTH_R
    y = np.radians(np.asarray(lat) - lat0) * EARTH_R
    return x, y


def to_lonlat(x, y, lon0, lat0):
    lon = lon0 + np.degrees(np.asarray(x) / (EARTH_R * np.cos(np.radians(lat0))))
    lat = lat0 + np.degrees(np.asarray(y) / EARTH_R)
    return lon, lat


def list_trips(split):
    d = os.path.join(RAW_DIR, split)
    return sorted(t for t in os.listdir(d) if os.path.isdir(os.path.join(d, t)))


def load_trip(trip_dir, with_route=True):
    trip_dir = trip_dir.rstrip("/\\")
    tid = os.path.basename(trip_dir)
    p = lambda ext: os.path.join(trip_dir, f"{tid}.{ext}")

    track = _read(p("track"), ["lon", "lat", "t"])
    nodes = _read(p("nodes"), ["lon", "lat"])
    arcs = _read(p("arcs"), ["u", "v"], dtype=np.int64)

    route = None
    if with_route and os.path.exists(p("route")):
        route = _read(p("route"), ["arc"], dtype=np.int64)["arc"].to_numpy()

    # drop invalid / duplicate fixes, keep time order
    track = track.dropna()
    track = track[track.lon.between(-180, 180) & track.lat.between(-90, 90)]
    track = track.sort_values("t", kind="stable").drop_duplicates("t").reset_index(drop=True)

    lon0, lat0 = track.lon.mean(), track.lat.mean()
    track["x"], track["y"] = to_xy(track.lon, track.lat, lon0, lat0)
    nodes["x"], nodes["y"] = to_xy(nodes.lon, nodes.lat, lon0, lat0)

    ux, uy = nodes.x.values[arcs.u], nodes.y.values[arcs.u]
    vx, vy = nodes.x.values[arcs.v], nodes.y.values[arcs.v]
    arcs["length"] = np.hypot(vx - ux, vy - uy)
    # compass bearing, 0 = north, clockwise
    arcs["bearing"] = np.degrees(np.arctan2(vx - ux, vy - uy)) % 360

    return Trip(tid, track, nodes, arcs, route, lon0, lat0)


def load_split(split, with_route=True):
    for tid in list_trips(split):
        yield load_trip(os.path.join(RAW_DIR, split, tid), with_route)
