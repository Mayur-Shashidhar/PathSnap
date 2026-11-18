"""P3 starter: candidates + connectivity + Viterbi (distance-only emissions for now).
Later: replace emission_logp() with the classifier's probabilities from P2.
"""
import sys, os, numpy as np
from collections import deque
from scipy.spatial import cKDTree

SIGMA = 10.0      # GPS noise in metres (tune later)
RADIUS = 50.0     # candidate search radius in metres
K = 8             # max candidates per point
MAX_HOPS = 4      # how far ahead (in arcs) a car can move between two points
HOP_COST = 1.0    # log-penalty per arc hop
FAR_COST = 20.0   # log-penalty for unconnected jump (soft, so decoding never breaks)


def load_trip(folder):
    tid = os.path.basename(os.path.normpath(folder))
    p = lambda ext: os.path.join(folder, f"{tid}.{ext}")
    track = np.loadtxt(p("track"), ndmin=2)            # lon lat t
    nodes = np.loadtxt(p("nodes"), ndmin=2)            # lon lat
    arcs = np.loadtxt(p("arcs"), dtype=int, ndmin=2)   # from to
    route = np.loadtxt(p("route"), dtype=int, ndmin=1) if os.path.exists(p("route")) else None
    return track, nodes, arcs, route


def project(lon, lat, lon0, lat0):
    """Local equirectangular projection to metres (fine at city scale)."""
    x = (lon - lon0) * np.cos(np.radians(lat0)) * 111320.0
    y = (lat - lat0) * 110540.0
    return x, y


def point_seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = np.where(L2 > 0, ((px - ax) * dx + (py - ay) * dy) / np.where(L2 > 0, L2, 1), 0)
    t = np.clip(t, 0, 1)
    cx, cy = ax + t * dx, ay + t * dy
    return np.hypot(px - cx, py - cy)


def get_candidates(track, nodes, arcs):
    lon0, lat0 = track[:, 0].mean(), track[:, 1].mean()
    nx, ny = project(nodes[:, 0], nodes[:, 1], lon0, lat0)
    gx, gy = project(track[:, 0], track[:, 1], lon0, lat0)
    ax, ay, bx, by = nx[arcs[:, 0]], ny[arcs[:, 0]], nx[arcs[:, 1]], ny[arcs[:, 1]]

    # densify arcs every ~10 m so a KD-tree can find nearby arcs fast
    pts, owner = [], []
    for i in range(len(arcs)):
        n = max(2, int(np.hypot(bx[i] - ax[i], by[i] - ay[i]) // 10) + 2)
        s = np.linspace(0, 1, n)
        pts.append(np.c_[ax[i] + s * (bx[i] - ax[i]), ay[i] + s * (by[i] - ay[i])])
        owner.append(np.full(n, i))
    tree = cKDTree(np.vstack(pts)); owner = np.concatenate(owner)

    cands, dists = [], []
    for j in range(len(track)):
        ids = np.unique(owner[tree.query_ball_point([gx[j], gy[j]], RADIUS + 10)])
        if len(ids) == 0:
            cands.append(np.array([], int)); dists.append(np.array([])); continue
        d = point_seg_dist(gx[j], gy[j], ax[ids], ay[ids], bx[ids], by[ids])
        keep = np.argsort(d)[:K]
        keep = keep[d[keep] <= RADIUS]
        cands.append(ids[keep]); dists.append(d[keep])
    return cands, dists


def build_graph(arcs):
    """arc -> list of arcs you can drive onto next (arc.to == next.from)."""
    by_from = {}
    for i, (f, _) in enumerate(arcs):
        by_from.setdefault(f, []).append(i)
    return [by_from.get(t, []) for _, t in arcs]


def hops_from(a, nxt, cache):
    """BFS: number of arc hops from arc a to reachable arcs (<= MAX_HOPS)."""
    if a in cache: return cache[a]
    seen, q = {a: 0}, deque([a])
    while q:
        u = q.popleft()
        if seen[u] >= MAX_HOPS: continue
        for v in nxt[u]:
            if v not in seen:
                seen[v] = seen[u] + 1; q.append(v)
    cache[a] = seen
    return seen


def emission_logp(d):
    return -0.5 * (d / SIGMA) ** 2          # swap for log(classifier prob) later


def viterbi(cands, dists, nxt):
    cache = {}
    idx = [j for j in range(len(cands)) if len(cands[j])]  # skip points with no candidates
    score = emission_logp(dists[idx[0]]); back = []
    for prev, cur in zip(idx, idx[1:]):
        pa, ca = cands[prev], cands[cur]
        T = np.full((len(pa), len(ca)), -FAR_COST)
        for i, a in enumerate(pa):
            h = hops_from(a, nxt, cache)
            for k, b in enumerate(ca):
                if b in h: T[i, k] = -HOP_COST * h[b]
        tot = score[:, None] + T
        back.append(tot.argmax(0))
        score = tot.max(0) + emission_logp(dists[cur])
    path = [int(score.argmax())]
    for bp in reversed(back): path.append(int(bp[path[-1]]))
    path.reverse()
    out = np.full(len(cands), -1)
    for j, p in zip(idx, path): out[j] = cands[j][p]
    return out


def evaluate(pred, cands, route):
    rs = set(route.tolist())
    valid = pred >= 0
    seq = [a for i, a in enumerate(pred) if a >= 0 and (i == 0 or a != pred[i - 1])]
    nn = np.array([c[0] if len(c) else -1 for c in cands])  # nearest-road baseline
    return {
        "points": len(pred),
        "no_candidate_pct": round(100 * (~valid).mean(), 2),
        "nearest_road_in_route_pct": round(100 * np.isin(nn[valid], list(rs)).mean(), 2),
        "viterbi_in_route_pct": round(100 * np.isin(pred[valid], list(rs)).mean(), 2),
        "route_arcs_recovered_pct": round(100 * len(rs & set(seq)) / len(rs), 2),
    }


if __name__ == "__main__":
    track, nodes, arcs, route = load_trip(sys.argv[1])
    cands, dists = get_candidates(track, nodes, arcs)
    pred = viterbi(cands, dists, build_graph(arcs))
    print(evaluate(pred, cands, route) if route is not None else pred)
