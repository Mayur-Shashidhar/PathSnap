"""Run the HMM/Viterbi decoder on P1's candidate CSV and report Top-1 accuracy.

Same candidates and point_idx as the classifier, so we can compare:
    - rank-1 (nearest arc) only
    - classifier only            (argmax of model prob per point)   -> needs --score
    - HMM with distance emission
    - HMM with classifier emission                                   -> needs --score

Usage:
    py hmm/hmm_eval.py --split student_test
    py hmm/hmm_eval.py --split student_test --preds results/preds_test.csv --score prob
      (preds csv: trip_id, point_idx, arc_id, <score col>)
"""
import os, sys, argparse, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from decoder import decode_trip, continuity, MIN_PROB

CSV = {"train": "data/candidates_train.csv.gz", "student_test": "data/candidates_test.csv"}
PTS = {"train": "data/points_train.csv", "student_test": "data/points_test.csv"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="student_test")
    ap.add_argument("--preds", default=None, help="csv with trip_id, point_idx, arc_id, <score>")
    ap.add_argument("--score", default="prob")
    ap.add_argument("--w_emit", type=float, default=1.0, help="weight of emission vs transition")
    a = ap.parse_args()

    c = pd.read_csv(CSV[a.split], dtype={"trip_id": str})
    pts = pd.read_csv(PTS[a.split], dtype={"trip_id": str}).set_index(["trip_id", "point_idx"])
    score_col = None
    if a.preds:
        p = pd.read_csv(a.preds, dtype={"trip_id": str})
        c = c.merge(p[["trip_id", "point_idx", "arc_id", a.score]], on=["trip_id", "point_idx", "arc_id"], how="left")
        c[a.score] = c[a.score].fillna(MIN_PROB)
        score_col = a.score

    rows = []
    for tid, ct in c.groupby("trip_id", sort=True):
        arcs = np.loadtxt(os.path.join("data", "raw", a.split, tid, f"{tid}.arcs"), dtype=int, ndmin=2)
        t0 = time.time()
        hmm_d = decode_trip(ct, arcs)
        sec = time.time() - t0
        rank1 = ct[ct["rank"] == 1].set_index("point_idx").arc_id.to_dict()
        methods = {"rank1": rank1, "hmm_dist": hmm_d}
        if score_col:
            methods["clf_only"] = ct.loc[ct.groupby("point_idx")[score_col].idxmax()].set_index("point_idx").arc_id.to_dict()
            methods["hmm_clf"] = decode_trip(ct, arcs, score_col, a.w_emit)
        truth = pts.loc[tid]
        for name, pred in methods.items():
            pi = np.array(sorted(pred))
            y = truth.loc[pi, "true_arc"].to_numpy()
            rel = truth.loc[pi, "reliable"].to_numpy() == 1
            yhat = np.array([pred[i] for i in pi])
            rows.append(dict(trip_id=tid, method=name, n=len(pi),
                             top1=100 * np.mean(yhat == y),
                             top1_reliable=100 * np.mean(yhat[rel] == y[rel]) if rel.any() else np.nan,
                             continuity=continuity(list(yhat), arcs),
                             seconds=round(sec, 2) if name == "hmm_dist" else np.nan))
    r = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    out = f"results/hmm_eval_{a.split}.csv"
    r.to_csv(out, index=False)

    # point-weighted overall numbers
    def overall(g):
        return pd.Series({"top1": np.average(g.top1, weights=g.n),
                          "top1_reliable": np.average(g.top1_reliable.fillna(0), weights=g.n),
                          "continuity": g.continuity.mean()})
    print(r.groupby("method").apply(overall).round(2))
    print("saved", out)


if __name__ == "__main__":
    main()
