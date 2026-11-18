"""Candidate miss rate vs search radius and K on the training trips.
Used to pick RADIUS / MAX_CANDS; writes results/candidate_sweep.csv.
The 200 m fallback is switched off here, so numbers are a bit higher than build_candidates."""
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loaders import RAW_DIR, list_trips, load_trip  # noqa: E402
from candidates import generate, align_route  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RADII = [20, 30, 50, 75, 100]
KS = [3, 5, 10, 20]


def one_trip(tid):
    trip = load_trip(os.path.join(RAW_DIR, "train", tid))
    cand, _ = generate(trip, radius=max(RADII), k=max(KS), fallback=0)
    true_arc, _, true_dist = align_route(trip)
    cand["pos"] = cand.arc_id.values == true_arc[cand.point_idx.values]
    reliable = true_dist <= 100
    rows = []
    for r in RADII:
        for k in KS:
            c = cand[(cand.dist <= r) & (cand["rank"] <= k)]
            hit = np.zeros(len(true_arc), bool)
            hit[c.point_idx.values[c.pos.values]] = True
            rows.append({"trip_id": tid, "radius": r, "k": k, "n_points": len(hit),
                         "hits": hit.sum(), "hits_reliable": hit[reliable].sum(),
                         "n_reliable": reliable.sum(), "rows": len(c)})
    return rows


if __name__ == "__main__":
    with Pool() as pool:
        out = pool.map(one_trip, list_trips("train"), chunksize=1)
    df = pd.DataFrame([r for rows in out for r in rows])
    g = df.groupby(["radius", "k"])[["n_points", "hits", "hits_reliable", "n_reliable", "rows"]].sum()
    res = pd.DataFrame({
        "miss_rate": 1 - g.hits / g.n_points,
        "miss_rate_reliable": 1 - g.hits_reliable / g.n_reliable,
        "cands_per_point": g.rows / g.n_points,
    }).round(4).reset_index()
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    res.to_csv(os.path.join(ROOT, "results", "candidate_sweep.csv"), index=False)
    print(res.to_string(index=False))
