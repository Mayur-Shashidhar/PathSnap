# features (P1)

```bash
python features/build_candidates.py --split train
python features/build_candidates.py --split student_test
python features/sweep_radius.py      # optional, radius/K miss-rate table
```

Takes ~25 s for train on 4 cores. Raw data must be in `data/raw/train/` and `data/raw/student_test/`.

| file | what |
|---|---|
| `loaders.py` | reads `.track/.nodes/.arcs/.route`, projects lon/lat to metres around the trip centre, arc length + bearing |
| `candidates.py` | KD-tree over arcs densified every 10 m, candidate generation, route alignment (labels) |
| `features.py` | per-candidate features |
| `build_candidates.py` | runs everything for a split, writes the CSVs |

## Outputs

- `data/candidates_train.csv.gz` (pandas reads it directly; the plain csv is 160 MB so it is not committed)
- `data/candidates_test.csv`
- `data/points_train.csv`, `data/points_test.csv`: one row per GPS point
- `results/candidate_stats_{train,test}.csv`: per-trip counts and miss rate
- `results/candidate_sweep.csv`: miss rate for radius 20–100 m and K 3–20

Read with `dtype={"trip_id": str}`, otherwise the leading zeros are lost.

### candidates_*.csv (one row per GPS point × candidate arc)

| column | meaning |
|---|---|
| trip_id, point_idx, t | which GPS point (point_idx is the row in the cleaned track) |
| arc_id | candidate arc (row index in `.arcs`) |
| dist | perpendicular distance point → arc segment (m) |
| rank | 1 = nearest candidate for this point |
| dist_minus_min, dist_ratio | dist relative to the nearest candidate (`dist - min`, `dist / (min + 1)`) |
| n_cands | number of candidates for this point |
| proj_t | where the projection lands on the arc, 0 = start node, 1 = end node |
| heading_diff | GPS heading vs arc bearing, 0–180°. Separates the two directions of a two-way road |
| heading_diff_undirected | same but 0–90°, ignores direction |
| heading_valid | 0 if the vehicle barely moved (< 2 m over the 3-point window); heading is then carried over from the last moving point |
| speed | m/s from the neighbouring points |
| dt_prev, dt_next | time gap to previous / next fix (s), for gap handling |
| arc_length | m |
| out_degree, in_degree | arcs leaving the arc's end node / entering its start node |
| dist_prev_pt, dist_next_pt | distance of the previous / next GPS point to this same arc |
| in_prev_cands, in_next_cands | 1 if this arc is also a candidate of the previous / next point |
| label | 1 if this is the reference arc for the point, else 0 (-1 when there is no route) |
| reliable | 0 if the point's reference arc is > 100 m away (see below) |

Use `trip_id` for GroupKFold. Don't feed `trip_id, point_idx, t, arc_id, label, reliable` to the model.

### points_*.csv

`trip_id, point_idx, t, lon, lat, n_cands, fallback, true_arc, route_pos, true_dist, hit, reliable`

`true_arc` is the reference arc for the point, `route_pos` its index in the `.route` file, `hit` whether it made it into the candidate list. P3 can use `lon/lat` and `true_arc` for maps and HMM evaluation.

## How it works

**Projection.** Equirectangular projection around the mean of each trip's GPS points. Trips are at most a few tens of km across, so the error vs a proper UTM projection is well under a metre and everything after that is plain Euclidean geometry.

**Cleaning.** Drop NaN/out-of-range coordinates, sort by timestamp, drop duplicate timestamps. Gaps are kept (not interpolated) and exposed through `dt_prev/dt_next`.

**Candidates.** Each arc is sampled every 10 m and the samples go into a KD-tree. For each point we take the arcs with a sample within R + 5 m, compute the exact point-to-segment distance, keep those within R = 50 m, and keep the K = 10 nearest. If a point has nothing within 50 m it gets up to 10 arcs within 200 m instead (`fallback = 1`) so the HMM always has a state.

**Labels.** The `.route` file is just an ordered list of arcs with no timestamps, so we can't directly say which arc a GPS point belongs to. Each point is matched to one route arc with a monotone DP: the matched position along the route can only stay the same or move forward, and the total point-to-arc distance (capped at 150 m per point) is minimised. Taking the plain nearest route arc instead breaks wherever the route passes the same road twice or uses both directions of a road.

**Reference route problems.** In trips 33 and 54 large parts of the route are ~130 m away from where the GPS is, while the GPS sits right on another road (nearest arc ~6 m in trip 33). Trip 2 has the GPS continuing for ~500 points past the end of the route, plus a short detour around points 757–967. These points are marked `reliable = 0` (true_dist > 100 m). They are about 10% of train points, almost all from those three trips. Probably best to drop them when training and report metrics both ways.

## Numbers (R = 50 m, K = 10)

| | train | student_test |
|---|---|---|
| trips / points | 80 / 208,417 | 10 / 19,160 |
| candidate rows | 1,407,484 | 131,630 |
| candidates per point | 6.75 | 6.87 |
| positive rate | 13.7% | 14.5% |
| candidate miss rate | 7.78% | 0.26% |
| miss rate, reliable points only | 1.25% | 0.17% |
| nearest-arc (rank 1) top-1 accuracy | 49.9% | 53.6% |

Rank 1 alone is only ~50% right, mostly because two-way roads give two arcs at exactly the same distance and only the heading tells them apart.

Radius sweep on train (reliable points, no fallback), from `results/candidate_sweep.csv`:

| R \ K | 5 | 10 | 20 |
|---|---|---|---|
| 30 m | 4.40% | 3.19% | 2.94% |
| 50 m | 3.44% | 2.09% | 1.67% |
| 75 m | 3.01% | 1.56% | 1.11% |
| 100 m | 2.82% | 1.17% | 0.70% |

50 m / 10 was picked as the trade-off: going to 100 m / 20 roughly doubles the candidates per point (5.9 → 12.5) for about 1.4 points less miss rate, and every extra candidate is an extra HMM state.
