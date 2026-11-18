"""Route visualization component using Folium."""

from typing import Any, Dict, List
import folium
import streamlit as st

# Check for streamlit_folium
try:
    from streamlit_folium import folium_static
    HAS_FOLIUM_STATIC = True
except ImportError:
    HAS_FOLIUM_STATIC = False


def render_route_map(pred_data: Dict[str, Any], height: int = 540):
    """Render interactive Folium map displaying GPS track and predicted route."""
    gps_coords: List[List[float]] = pred_data.get("gps_coords", [])
    route_coords: List[List[List[float]]] = pred_data.get("route_coords", [])
    trip_id = pred_data.get("trip_id", "Unknown")
    continuity = pred_data.get("continuity_pct", 0.0)

    if not gps_coords:
        st.warning("No GPS coordinates available to render map.")
        return

    # Calculate map bounds and center
    lats = [pt[0] for pt in gps_coords]
    lons = [pt[1] for pt in gps_coords]
    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)

    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    # Initialize Folium Map with clean, free OpenStreetMap and Esri tiles (no API key watermark)
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles="OpenStreetMap",
        control_scale=True
    )

    # Optional satellite / street tiles without watermarks
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Street Map",
        name="Esri Streets"
    ).add_to(m)

    # Layer 1: Predicted Road Route (Bold Red/Coral Polyline)
    if route_coords:
        route_group = folium.FeatureGroup(name="Predicted Route (HMM Viterbi)", show=True)
        folium.PolyLine(
            locations=route_coords,
            color="#E63946",  # vivid coral-red
            weight=5,
            opacity=0.85,
            tooltip=f"Predicted Route ({len(route_coords)} segments)"
        ).add_to(route_group)
        route_group.add_to(m)

    # Layer 2: Raw GPS Observations (Deep Blue / Cyan Dots)
    gps_group = folium.FeatureGroup(name="Raw GPS Trajectory", show=True)
    # Downsample visually if more than 2000 points to keep browser responsive
    step = max(1, len(gps_coords) // 2000)
    for idx, (lat, lon) in enumerate(gps_coords[::step]):
        folium.CircleMarker(
            location=[lat, lon],
            radius=2.5,
            color="#1D3557",
            weight=1,
            fill=True,
            fill_color="#457B9D",
            fill_opacity=0.75,
            tooltip=f"GPS #{idx * step}"
        ).add_to(gps_group)
    gps_group.add_to(m)

    # Start and End Markers
    start_pt = gps_coords[0]
    end_pt = gps_coords[-1]

    folium.Marker(
        location=start_pt,
        popup=f"Start: {start_pt[0]:.5f}, {start_pt[1]:.5f}",
        tooltip="Trip Origin",
        icon=folium.Icon(color="green", icon="play", prefix="fa")
    ).add_to(m)

    folium.Marker(
        location=end_pt,
        popup=f"Destination: {end_pt[0]:.5f}, {end_pt[1]:.5f}",
        tooltip="Trip Destination",
        icon=folium.Icon(color="red", icon="stop", prefix="fa")
    ).add_to(m)

    # Auto-fit bounds with padding
    m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]], padding=(20, 20))

    # Clean embedded legend
    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 25px;
        right: 25px;
        z-index: 1000;
        background-color: rgba(255, 255, 255, 0.95);
        padding: 10px 14px;
        border-radius: 6px;
        border: 1px solid #ddd;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 12px;
        line-height: 1.6;
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    ">
        <b style="font-size: 13px; color: #111;">Trip {trip_id}</b><br>
        <span style="color: #E63946; font-size: 16px; font-weight: bold;">&#9644;&#9644;</span> Predicted Route (HMM)<br>
        <span style="color: #457B9D; font-size: 14px;">&#9679;</span> Raw GPS Trajectory<br>
        <span style="color: #2A9D8F; font-size: 14px;">&#9658;</span> Origin / <span style="color: #E76F51; font-size: 14px;">&#9632;</span> Destination<br>
        <span style="color: #666; font-size: 11px;">Continuity: {continuity:.1f}%</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(position="topright", collapsed=False).add_to(m)

    # Render in Streamlit (use HTML component for responsive full-width display)
    st.components.v1.html(m._repr_html_(), height=height + 20)
