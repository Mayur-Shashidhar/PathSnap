"""Run the HMM baseline on every trip in a split and save results to CSV.
Usage: python hmm/run_all.py train
       python hmm/run_all.py student_test
"""
import os, sys, time, csv
sys.path.insert(0, os.path.dirname(__file__))
from p3_hmm import load_trip, get_candidates, build_graph, viterbi, evaluate

split = sys.argv[1] if len(sys.argv) > 1 else "train"
root = os.path.join("data", "raw", split)
os.makedirs("results", exist_ok=True)
out_path = os.path.join("results", f"hmm_baseline_{split}.csv")

rows = []
for tid in sorted(os.listdir(root)):
    folder = os.path.join(root, tid)
    if not os.path.isdir(folder):
        continue
    t0 = time.time()
    try:
        track, nodes, arcs, route = load_trip(folder)
        cands, dists = get_candidates(track, nodes, arcs)
        pred = viterbi(cands, dists, build_graph(arcs))
        r = {k: float(v) for k, v in evaluate(pred, cands, route).items()}
        r["points"] = int(r["points"])
        r.update(trip_id=tid, n_arcs=len(arcs), seconds=round(time.time() - t0, 2), error="")
    except Exception as e:
        r = dict(trip_id=tid, error=str(e))
    rows.append(r)
    print(tid, r)

cols = ["trip_id", "points", "n_arcs", "no_candidate_pct", "nearest_road_in_route_pct",
        "viterbi_in_route_pct", "route_arcs_recovered_pct", "seconds", "error"]
with open(out_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)

ok = [r for r in rows if not r.get("error")]
if ok:
    avg = lambda k: sum(r[k] for r in ok) / len(ok)
    print(f"\n{len(ok)}/{len(rows)} trips OK | nearest-road {avg('nearest_road_in_route_pct'):.2f}% "
          f"| viterbi {avg('viterbi_in_route_pct'):.2f}% | route recovered {avg('route_arcs_recovered_pct'):.2f}%")
print("saved", out_path)
