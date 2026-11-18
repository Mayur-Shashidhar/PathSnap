import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

RADIUS = 50.0      # metres
MAX_CANDS = 10
FALLBACK_RADIUS = 200.0
DENSIFY_STEP = 10.0


def point_seg_dist(px, py, ax, ay, bx, by):
    """Distance from points to segments (all arrays broadcast). Also returns projection t in [0, 1]."""
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    t = np.where(l2 > 0, ((px - ax) * dx + (py - ay) * dy) / np.where(l2 > 0, l2, 1), 0.0)
    t = np.clip(t, 0.0, 1.0)
    cx, cy = ax + t * dx, ay + t * dy
    return np.hypot(px - cx, py - cy), t


def arc_coords(trip):
    n = trip.nodes
    u, v = trip.arcs.u.values, trip.arcs.v.values
    return n.x.values[u], n.y.values[u], n.x.values[v], n.y.values[v]


class ArcIndex:
    """KD-tree over points sampled every DENSIFY_STEP metres along each arc."""

    def __init__(self, trip, step=DENSIFY_STEP):
        self.ax, self.ay, self.bx, self.by = arc_coords(trip)
        lengths = trip.arcs.length.values
        n_samp = np.maximum(np.ceil(lengths / step).astype(int), 1) + 1
        arc_of = np.repeat(np.arange(len(lengths)), n_samp)
        # fractional position of each sample along its arc
        starts = np.cumsum(n_samp) - n_samp
        frac = (np.arange(n_samp.sum()) - np.repeat(starts, n_samp)) / np.repeat(n_samp - 1, n_samp)
        sx = self.ax[arc_of] + frac * (self.bx - self.ax)[arc_of]
        sy = self.ay[arc_of] + frac * (self.by - self.ay)[arc_of]
        self.arc_of = arc_of
        self.step = step
        self.tree = cKDTree(np.column_stack([sx, sy]))

    def query(self, px, py, radius):
        """Unique (point_idx, arc_id) pairs with at least one arc sample within radius + step/2."""
        hits = self.tree.query_ball_point(np.column_stack([px, py]), r=radius + self.step / 2)
        sizes = np.fromiter((len(h) for h in hits), dtype=np.int64, count=len(hits))
        if sizes.sum() == 0:
            return np.empty(0, int), np.empty(0, int)
        pi = np.repeat(np.arange(len(hits)), sizes)
        ai = self.arc_of[np.concatenate([np.asarray(h, dtype=np.int64) for h in hits if h])]
        key = np.unique(pi * (len(self.arc_of) + 1) + ai)
        return key // (len(self.arc_of) + 1), key % (len(self.arc_of) + 1)

    def dist(self, px, py, arc_ids):
        return point_seg_dist(px, py, self.ax[arc_ids], self.ay[arc_ids], self.bx[arc_ids], self.by[arc_ids])


def generate(trip, index=None, radius=RADIUS, k=MAX_CANDS, fallback=FALLBACK_RADIUS):
    """Candidate arcs for every GPS point: up to k nearest arcs within radius.
    Points with nothing inside radius get the k nearest within the fallback radius."""
    index = index or ArcIndex(trip)
    px, py = trip.track.x.values, trip.track.y.values

    pi, ai = index.query(px, py, radius)
    d, _ = index.dist(px[pi], py[pi], ai)
    keep = d <= radius
    pi, ai, d = pi[keep], ai[keep], d[keep]

    empty = np.setdiff1d(np.arange(len(px)), pi)
    if len(empty):
        fpi, fai = index.query(px[empty], py[empty], fallback)
        fd, _ = index.dist(px[empty][fpi], py[empty][fpi], fai)
        ok = fd <= fallback
        pi = np.concatenate([pi, empty[fpi[ok]]])
        ai = np.concatenate([ai, fai[ok]])
        d = np.concatenate([d, fd[ok]])

    cand = pd.DataFrame({"point_idx": pi, "arc_id": ai, "dist": d})
    cand = cand.sort_values(["point_idx", "dist", "arc_id"], kind="stable")
    cand["rank"] = cand.groupby("point_idx").cumcount() + 1
    cand = cand[cand["rank"] <= k].reset_index(drop=True)
    return cand, index


def align_route(trip, cap=150.0):
    """Assign each GPS point to one arc of the reference route.

    The route has no timestamps, so points are aligned to the arc sequence with a
    monotone DP (the vehicle can only move forward along the route) minimising the
    total point-to-arc distance. Distances are capped so a few bad fixes cannot
    drag the alignment around.
    """
    route = trip.route
    ax, ay, bx, by = arc_coords(trip)
    ax, ay, bx, by = ax[route], ay[route], bx[route], by[route]
    px, py = trip.track.x.values, trip.track.y.values
    n, m = len(px), len(route)

    back = np.empty((n, m), dtype=np.int16 if m < 32767 else np.int32)
    dists = np.empty((n, m), dtype=np.float32)
    prev = None
    idx = np.arange(m)
    for i in range(n):
        d, _ = point_seg_dist(px[i], py[i], ax, ay, bx, by)
        dists[i] = d
        c = np.minimum(d, cap)
        if prev is None:
            cur = c
            back[i] = idx
        else:
            # running min over k <= j of prev[k], and where it came from
            run_min = np.minimum.accumulate(prev)
            is_new = prev <= run_min
            arg = np.maximum.accumulate(np.where(is_new, idx, 0))
            cur = c + run_min
            back[i] = arg
        prev = cur

    path = np.empty(n, dtype=np.int64)
    j = int(np.argmin(prev))
    for i in range(n - 1, -1, -1):
        path[i] = j
        if i > 0:
            j = int(back[i, j])
    true_arc = route[path]
    true_dist = dists[np.arange(n), path].astype(float)
    return true_arc, path, true_dist
