"""Folium map: raw GPS, road network, true route, predicted route.
Usage: py hmm/make_map.py data/raw/train/00000005
Saves hmm/maps/<trip_id>.html
"""
import os, sys
import folium
sys.path.insert(0, os.path.dirname(__file__))
from p3_hmm import load_trip, get_candidates, build_graph, viterbi, evaluate

MAX_NET_ARCS = 4000   # cap road-network layer so the HTML stays small


def arc_lines(arc_ids, nodes, arcs):
    # folium wants [lat, lon]
    return [[[nodes[arcs[a, 0], 1], nodes[arcs[a, 0], 0]],
             [nodes[arcs[a, 1], 1], nodes[arcs[a, 1], 0]]] for a in arc_ids]


def make_map(folder):
    tid = os.path.basename(os.path.normpath(folder))
    track, nodes, arcs, route = load_trip(folder)
    cands, dists = get_candidates(track, nodes, arcs)
    pred = viterbi(cands, dists, build_graph(arcs))
    stats = evaluate(pred, cands, route) if route is not None else {}

    lat, lon = track[:, 1], track[:, 0]
    m = folium.Map(tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",attr="Esri World Street Map")
    m.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])

    # road network near the trip (arcs that were candidates for any point)
    near = sorted({a for c in cands for a in c})[:MAX_NET_ARCS]
    net = folium.FeatureGroup(name="Road network (nearby)", show=False)
    folium.PolyLine(arc_lines(near, nodes, arcs), color="#999999", weight=1, opacity=0.6).add_to(net)
    net.add_to(m)

    gps = folium.FeatureGroup(name="Raw GPS points")
    step = max(1, len(track) // 2000)  # thin out very long tracks
    for x, y in zip(lon[::step], lat[::step]):
        folium.CircleMarker([y, x], radius=2, color="#1f77b4", fill=True, weight=0).add_to(gps)
    gps.add_to(m)

    if route is not None:
        gt = folium.FeatureGroup(name="Reference route")
        folium.PolyLine(arc_lines(route, nodes, arcs), color="#2ca02c", weight=6, opacity=0.6).add_to(gt)
        gt.add_to(m)

    seq = [a for i, a in enumerate(pred) if a >= 0 and (i == 0 or a != pred[i - 1])]
    pr = folium.FeatureGroup(name="Predicted route (HMM)")
    folium.PolyLine(arc_lines(seq, nodes, arcs), color="#d62728", weight=3).add_to(pr)
    pr.add_to(m)

    folium.Marker([lat[0], lon[0]], tooltip="Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker([lat[-1], lon[-1]], tooltip="End", icon=folium.Icon(color="red")).add_to(m)

    legend = f"""
    <div style="position:fixed;bottom:20px;left:20px;z-index:9999;background:white;
                padding:8px 12px;border:1px solid #ccc;font-size:13px;">
      <b>Trip {tid}</b><br>
      <span style="color:#1f77b4">&#9679;</span> Raw GPS<br>
      <span style="color:#2ca02c">&#9644;</span> Reference route<br>
      <span style="color:#d62728">&#9644;</span> Predicted route<br>
      <span style="color:#999">&#9644;</span> Road network<br>
      {"<br>".join(f"{k}: {float(v)}" for k, v in stats.items())}
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)

    os.makedirs(os.path.join("hmm", "maps"), exist_ok=True)
    out = os.path.join("hmm", "maps", f"{tid}.html")
    m.save(out)
    print("saved", out, stats)


if __name__ == "__main__":
    for f in sys.argv[1:]:
        make_map(f)
