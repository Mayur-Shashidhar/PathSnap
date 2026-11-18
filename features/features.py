import numpy as np
import pandas as pd

from candidates import point_seg_dist, arc_coords

MIN_MOVE = 2.0   # metres; below this the GPS heading is just noise


def track_kinematics(track):
    """Heading (deg, 0 = north) and speed (m/s) per point from a centred difference."""
    x, y, t = track.x.values, track.y.values, track.t.values
    n = len(x)
    i0 = np.clip(np.arange(n) - 1, 0, n - 1)
    i1 = np.clip(np.arange(n) + 1, 0, n - 1)
    dx, dy = x[i1] - x[i0], y[i1] - y[i0]
    dt = np.maximum(t[i1] - t[i0], 1e-6)
    move = np.hypot(dx, dy)

    speed = move / dt
    heading = pd.Series(np.where(move >= MIN_MOVE, np.degrees(np.arctan2(dx, dy)) % 360, np.nan))
    valid = heading.notna().to_numpy()
    # stationary points inherit the last real heading
    heading = heading.ffill().bfill().fillna(0).to_numpy()

    gaps = np.diff(t)
    dt_prev = np.concatenate([[0.0], gaps])
    dt_next = np.concatenate([gaps, [0.0]])
    return pd.DataFrame({
        "heading": heading, "heading_valid": valid.astype(int),
        "speed": speed, "dt_prev": dt_prev, "dt_next": dt_next,
    })


def angle_diff(a, b):
    d = np.abs(a - b) % 360
    return np.minimum(d, 360 - d)


def build_features(trip, cand):
    track = trip.track
    kin = track_kinematics(track)
    arcs = trip.arcs
    ax, ay, bx, by = arc_coords(trip)
    px, py = track.x.values, track.y.values
    n = len(track)

    p = cand.point_idx.values
    a = cand.arc_id.values
    f = cand.copy()

    f["t"] = track.t.values[p]
    _, proj = point_seg_dist(px[p], py[p], ax[a], ay[a], bx[a], by[a])
    f["proj_t"] = proj

    grp = f.groupby("point_idx")["dist"]
    dmin = grp.transform("min")
    f["n_cands"] = grp.transform("size")
    f["dist_minus_min"] = f["dist"] - dmin
    f["dist_ratio"] = f["dist"] / (dmin + 1.0)

    f["heading_diff"] = angle_diff(kin.heading.values[p], arcs.bearing.values[a])
    # 0..90: how parallel the road is, ignoring direction
    f["heading_diff_undirected"] = np.minimum(f["heading_diff"], 180 - f["heading_diff"])
    f["heading_valid"] = kin.heading_valid.values[p]
    f["speed"] = kin.speed.values[p]
    f["dt_prev"] = kin.dt_prev.values[p]
    f["dt_next"] = kin.dt_next.values[p]

    f["arc_length"] = arcs.length.values[a]
    out_deg = np.bincount(arcs.u.values, minlength=len(trip.nodes))
    in_deg = np.bincount(arcs.v.values, minlength=len(trip.nodes))
    f["out_degree"] = out_deg[arcs.v.values[a]]
    f["in_degree"] = in_deg[arcs.u.values[a]]

    # how well the same arc explains the neighbouring fixes
    for name, off in (("dist_prev_pt", -1), ("dist_next_pt", 1)):
        q = np.clip(p + off, 0, n - 1)
        d, _ = point_seg_dist(px[q], py[q], ax[a], ay[a], bx[a], by[a])
        f[name] = d

    # is the arc also a candidate for the neighbouring points
    key = set(zip(p, a))
    f["in_prev_cands"] = [(i - 1, j) in key for i, j in zip(p, a)]
    f["in_next_cands"] = [(i + 1, j) in key for i, j in zip(p, a)]
    f[["in_prev_cands", "in_next_cands"]] = f[["in_prev_cands", "in_next_cands"]].astype(int)
    return f
