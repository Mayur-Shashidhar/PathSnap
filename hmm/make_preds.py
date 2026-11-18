"""Score P1's candidates with P2's saved model -> preds csv for hmm_eval.py.
Usage: py hmm/make_preds.py --model models/lightgbm_baseline.joblib --split student_test --out results/preds_test_lgbm.csv
"""
import argparse, joblib, numpy as np, pandas as pd

FEATURES = ["dist", "rank", "dist_minus_min", "dist_ratio", "n_cands", "proj_t", "heading_diff",
            "heading_diff_undirected", "heading_valid", "speed", "dt_prev", "dt_next", "arc_length",
            "out_degree", "in_degree", "dist_prev_pt", "dist_next_pt", "in_prev_cands", "in_next_cands"]
CSV = {"train": "data/candidates_train.csv.gz", "student_test": "data/candidates_test.csv"}

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/lightgbm_baseline.joblib")
ap.add_argument("--split", default="student_test")
ap.add_argument("--out", default="results/preds_test_lgbm.csv")
a = ap.parse_args()

c = pd.read_csv(CSV[a.split], dtype={"trip_id": str})
model = joblib.load(a.model)
c["prob"] = model.predict_proba(c[FEATURES].astype(np.float32))[:, 1]
c[["trip_id", "point_idx", "arc_id", "prob"]].to_csv(a.out, index=False)
print("saved", a.out, len(c), "rows")
