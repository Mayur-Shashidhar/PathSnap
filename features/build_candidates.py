"""Build the candidate/feature tables for one split.

    python features/build_candidates.py --split train
    python features/build_candidates.py --split student_test
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loaders import RAW_DIR, list_trips, load_trip  # noqa: E402
from candidates import generate, align_route, RADIUS, MAX_CANDS  # noqa: E402
from features import build_features  # noqa: E402

# reference routes are not perfect (trips 33 and 54 drift ~130 m away from a road the
# GPS clearly follows), so points whose aligned arc is this far away are flagged
RELIABLE_DIST = 100.0

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_NAME = {"train": "candidates_train.csv", "student_test": "candidates_test.csv"}

FEATURE_COLS = [
    "dist", "rank", "dist_minus_min", "dist_ratio", "n_cands", "proj_t",
    "heading_diff", "heading_diff_undirected", "heading_valid", "speed",
    "dt_prev", "dt_next", "arc_length", "out_degree", "in_degree",
    "dist_prev_pt", "dist_next_pt", "in_prev_cands", "in_next_cands",
]


def process(args):
    split, tid, radius, k = args
    t0 = time.time()
    trip = load_trip(os.path.join(RAW_DIR, split, tid))
    cand, _ = generate(trip, radius=radius, k=k)
    feats = build_features(trip, cand)
    feats.insert(0, "trip_id", tid)

    pts = trip.track[["t", "lon", "lat"]].copy()
    pts.insert(0, "point_idx", np.arange(len(pts)))
    pts.insert(0, "trip_id", tid)
    pts["n_cands"] = np.bincount(cand.point_idx, minlength=len(pts))
    nearest = cand[cand["rank"] == 1].set_index("point_idx").dist
    pts["fallback"] = (nearest.reindex(pts.point_idx).fillna(np.inf).values > radius).astype(int)

    if trip.route is not None:
        true_arc, route_pos, true_dist = align_route(trip)
        feats["label"] = (feats.arc_id.values == true_arc[feats.point_idx.values]).astype(int)
        pts["true_arc"] = true_arc
        pts["route_pos"] = route_pos
        pts["true_dist"] = true_dist
        hit = np.zeros(len(pts), dtype=bool)
        hit[feats.point_idx.values[feats.label.values == 1]] = True
        pts["hit"] = hit.astype(int)
        pts["reliable"] = (true_dist <= RELIABLE_DIST).astype(int)
        feats["reliable"] = pts["reliable"].values[feats.point_idx.values]
    else:
        feats["label"] = -1
        feats["reliable"] = 0

    stats = {
        "trip_id": tid, "n_points": len(pts), "n_arcs": len(trip.arcs),
        "n_cand_rows": len(feats), "mean_cands": len(feats) / max(len(pts), 1),
        "empty_points": int((pts.n_cands == 0).sum()),
        "fallback_points": int(pts.fallback.sum()),
        "miss_rate": float(1 - pts.hit.mean()) if "hit" in pts else np.nan,
        "reliable_frac": float(pts.reliable.mean()) if "hit" in pts else np.nan,
        "miss_rate_reliable": float(1 - pts.hit[pts.reliable == 1].mean()) if "hit" in pts else np.nan,
        "seconds": round(time.time() - t0, 2),
    }
    return feats, pts, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "student_test"])
    ap.add_argument("--radius", type=float, default=RADIUS)
    ap.add_argument("--k", type=int, default=MAX_CANDS)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    trips = list_trips(args.split)
    jobs = [(args.split, t, args.radius, args.k) for t in trips]
    with Pool(args.workers) as pool:
        out = pool.map(process, jobs, chunksize=1)

    feats = pd.concat([o[0] for o in out], ignore_index=True)
    pts = pd.concat([o[1] for o in out], ignore_index=True)
    stats = pd.DataFrame([o[2] for o in out])

    cols = ["trip_id", "point_idx", "t", "arc_id"] + FEATURE_COLS + ["label", "reliable"]
    feats = feats[cols]
    float_cols = feats.select_dtypes("float").columns
    feats[float_cols] = feats[float_cols].round(3)

    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    tag = "train" if args.split == "train" else "test"
    feats.to_csv(os.path.join(ROOT, "data", OUT_NAME[args.split]), index=False)
    pts.round({"true_dist": 2}).to_csv(os.path.join(ROOT, "data", f"points_{tag}.csv"), index=False)
    stats.round(4).to_csv(os.path.join(ROOT, "results", f"candidate_stats_{tag}.csv"), index=False)

    if "hit" in pts:
        print(f"{args.split}: {len(trips)} trips, {len(pts)} points, {len(feats)} candidate rows")
        print(f"mean candidates/point {len(feats) / len(pts):.2f}")
        print(f"candidate miss rate {1 - pts.hit.mean():.4f}")
        rel = pts.reliable == 1
        print(f"miss rate on reliable points {1 - pts.hit[rel].mean():.4f} ({rel.mean():.1%} of points)")
        print(f"points with nothing within {args.radius:.0f} m (fallback used) {pts.fallback.mean():.4f}")
        print(f"points with no candidate at all {(pts.n_cands == 0).mean():.4f}")
        print(f"positive rate {feats.label.mean():.4f}")


if __name__ == "__main__":
    main()
