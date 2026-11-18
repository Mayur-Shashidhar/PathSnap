"""P3 decoder on P1's candidates (same roads and point_idx that P2's model scores).

decode_trip(cands, arcs)            -> {point_idx: arc_id}    Viterbi over one trip
predict_route(trip_dir, model, cols)-> dict for the API       full pipeline, no ground truth
"""
import os, sys, time
from collections import deque
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "features"))

SIGMA = 10.0      # GPS noise (m) for the distance emission
MAX_HOPS = 4      # arcs a car can move between two consecutive 1 Hz points
GAP_S = 3.0       # a time gap above this counts as a GPS gap
HOPS_PER_S = 0.5  # extra hops allowed per second of gap
MAX_GAP_HOPS = 20
HOP_COST = 1.0    # log-penalty per arc hop
FAR_COST = 20.0   # log-penalty for an unconnected jump (soft, decoding never breaks)
MIN_PROB = 1e-6


def build_graph(arcs):
    """arc -> arcs you can drive onto next (arc.to == next.from)."""
    by_from = {}
    for i, (f, _) in enumerate(arcs):
        by_from.setdefault(int(f), []).append(i)
    return [by_from.get(int(t), []) for _, t in arcs]


def hops_upto(a, max_hops, nxt, cache):
    """BFS: {arc: hops} reachable from arc a within max_hops."""
    key = (a, max_hops)
    if key in cache:
        return cache[key]
    seen, q = {a: 0}, deque([a])
    while q:
        u = q.popleft()
        if seen[u] >= max_hops:
            continue
        for v in nxt[u]:
            if v not in seen:
                seen[v] = seen[u] + 1
                q.append(v)
    cache[key] = seen
    return seen


def allowed_hops(dt):
    """More hops when the GPS skipped time (gap handling)."""
    if not np.isfinite(dt) or dt <= GAP_S:
        return MAX_HOPS
    return int(min(MAX_GAP_HOPS, MAX_HOPS + HOPS_PER_S * dt))


def decode_trip(c, arcs, score_col=None, w_emit=1.0):
    """c: candidate rows of ONE trip (P1 format). Returns {point_idx: arc_id}."""
    nxt, cache = build_graph(arcs), {}
    pts = []
    for pidx, g in c.groupby("point_idx", sort=True):
        ids = g.arc_id.to_numpy()
        if score_col:
            e = np.log(np.maximum(g[score_col].to_numpy(), MIN_PROB))
        else:
            e = -0.5 * (g.dist.to_numpy() / SIGMA) ** 2
        dt = float(g.dt_prev.iloc[0]) if "dt_prev" in g else 1.0
        pts.append((pidx, ids, w_emit * e, dt))
    if not pts:
        return {}
    score, back = pts[0][2], []
    for (_, pa, _, _), (_, ca, e, dt) in zip(pts, pts[1:]):
        h_max = allowed_hops(dt)
        T = np.full((len(pa), len(ca)), -FAR_COST)
        for i, a in enumerate(pa):
            h = hops_upto(int(a), h_max, nxt, cache)
            for k, b in enumerate(ca):
                if int(b) in h:
                    T[i, k] = -HOP_COST * h[int(b)]
        tot = score[:, None] + T
        back.append(tot.argmax(0))
        score = tot.max(0) + e
    path = [int(score.argmax())]
    for bp in reversed(back):
        path.append(int(bp[path[-1]]))
    path.reverse()
    return {int(p[0]): int(p[1][k]) for p, k in zip(pts, path)}


def continuity(seq, arcs):
    """% of consecutive distinct predicted arcs that are actually connected."""
    seq = [a for i, a in enumerate(seq) if i == 0 or a != seq[i - 1]]
    if len(seq) < 2:
        return 100.0
    return float(100 * np.mean([arcs[a, 1] == arcs[b, 0] for a, b in zip(seq, seq[1:])]))


def predict_route(trip_dir, model=None, feature_cols=None, w_emit=1.0):
    """API entry point. Loads the trip WITHOUT its route, runs P1 candidates/features,
    the optional classifier (sklearn-style predict_proba) and Viterbi."""
    from loaders import load_trip
    from candidates import generate
    from features import build_features

    t0 = time.time()
    trip = load_trip(trip_dir, with_route=False)
    cand, _ = generate(trip)
    feats = build_features(trip, cand)
    score_col = None
    if model is not None:
        feats["prob"] = model.predict_proba(feats[feature_cols])[:, 1]
        score_col = "prob"
    arcs = np.c_[trip.arcs.u.values, trip.arcs.v.values]
    pred = decode_trip(feats, arcs, score_col, w_emit)

    n = len(trip.track)
    arc_per_point = [pred.get(i, -1) for i in range(n)]
    route_arcs = [a for i, a in enumerate(arc_per_point) if a >= 0 and (i == 0 or a != arc_per_point[i - 1])]
    lon, lat = trip.nodes.lon.values, trip.nodes.lat.values
    return {
        "trip_id": os.path.basename(os.path.normpath(trip_dir)),
        "n_points": n,
        "arc_per_point": arc_per_point,
        "route_arcs": route_arcs,
        "route_coords": [[[float(lat[u]), float(lon[u])], [float(lat[v]), float(lon[v])]]
                         for u, v in (arcs[a] for a in route_arcs)],
        "gps_coords": trip.track[["lat", "lon"]].values.tolist(),
        "continuity_pct": round(continuity(route_arcs, arcs), 2),
        "seconds": round(time.time() - t0, 2),
    }


def score_against_route(res, true_arc):
    """Server-side only: Top-1 vs P1-style per-point true arcs (array, len n_points)."""
    pred = np.array(res["arc_per_point"])
    return round(float(100 * np.mean(pred == np.asarray(true_arc))), 2)
