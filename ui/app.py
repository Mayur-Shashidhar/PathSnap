"""PathSnap Evaluator Dashboard (Laptop 2).

Streamlit user interface connecting exclusively via HTTP to the FastAPI backend on Laptop 1.
"""

import os
import sys
import streamlit as st

# Ensure ui package modules are discoverable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui.api_client import PathSnapApiClient
from ui.components.status import render_system_status
from ui.components.metrics import render_prediction_metrics
from ui.components.route_map import render_route_map

# Set Streamlit page configuration
st.set_page_config(
    page_title="PathSnap | Route Reconstruction",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, professional geospatial dashboard styling
st.markdown("""
<style>
    /* Clean, modern typography and dark slate header accents */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #E2E8F0;
        margin-bottom: 12px;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .status-online {
        background-color: #DCFCE7;
        color: #166534;
    }
    .status-offline {
        background-color: #FEE2E2;
        color: #991B1B;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize Streamlit session state variables."""
    if "api_url" not in st.session_state:
        st.session_state.api_url = os.getenv("PATHSNAP_API_URL", "http://localhost:8000")
    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None
    if "selected_trip_id" not in st.session_state:
        st.session_state.selected_trip_id = None
    if "health_data" not in st.session_state:
        st.session_state.health_data = None
    if "trips_list" not in st.session_state:
        st.session_state.trips_list = []


def main():
    init_session_state()

    # ------------------ SIDEBAR CONTROLS ------------------
    st.sidebar.title("🗺️ PathSnap Control")
    st.sidebar.markdown("**Evaluator Interface (Laptop 2)**")
    st.sidebar.divider()

    # API Connection Configuration
    st.sidebar.subheader("Backend Configuration")
    api_url_input = st.sidebar.text_input(
        "Laptop 1 API URL",
        value=st.session_state.api_url,
        help="HTTP address of Laptop 1 running FastAPI (e.g. http://192.168.1.100:8000 or http://localhost:8000)"
    )

    if api_url_input != st.session_state.api_url:
        st.session_state.api_url = api_url_input
        st.session_state.prediction_result = None

    client = PathSnapApiClient(base_url=st.session_state.api_url)

    # Refresh Connection button
    if st.sidebar.button("🔄 Check Backend & Sync Trips", use_container_width=True):
        st.session_state.health_data = None
        st.session_state.trips_list = []

    # Fetch Health
    health_ok, health_data, health_err = client.get_health()
    st.session_state.health_data = health_data if health_ok else None

    # Fetch Evaluator Trips
    trips_ok, trips_list, trips_err = client.get_trips()
    if trips_ok:
        st.session_state.trips_list = trips_list
    else:
        st.session_state.trips_list = []

    st.sidebar.divider()

    # Trip Selection & Prediction Trigger
    st.sidebar.subheader("Trip Selection")
    eval_trips = st.session_state.trips_list

    if eval_trips:
        selected_trip = st.sidebar.selectbox(
            "Select Evaluator Trip",
            options=eval_trips,
            index=0,
            help="Available organizer-provided evaluator trips discovered on Laptop 1."
        )
        st.session_state.selected_trip_id = selected_trip

        run_disabled = not health_ok or not health_data.get("model_loaded", False)
        run_btn = st.sidebar.button(
            "🚀 Run Route Prediction",
            type="primary",
            use_container_width=True,
            disabled=run_disabled
        )

        if run_disabled:
            st.sidebar.caption("⚠️ Prediction disabled: backend offline or model not loaded.")
    else:
        st.sidebar.warning(
            "No evaluator trips found on backend. Mount the evaluator dataset on Laptop 1 into `data/raw/evaluator/`."
        )
        run_btn = False

    st.sidebar.divider()
    st.sidebar.markdown(
        "<small style='color: #888;'>PathSnap ML Hackathon &bull; PES University<br>P4: API, UI & Integration</small>",
        unsafe_allow_html=True
    )

    # ------------------ MAIN DASHBOARD ------------------
    st.markdown('<div class="main-header">PathSnap</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Distributed Map-Matching & Route Reconstruction System</div>',
        unsafe_allow_html=True
    )

    # System Status Bar
    render_system_status(st.session_state.health_data, error=health_err)
    st.divider()

    # Prediction Action Handling
    if run_btn and st.session_state.selected_trip_id:
        trip_id = st.session_state.selected_trip_id
        with st.spinner(f"Running inference pipeline on Laptop 1 for trip '{trip_id}' ..."):
            success, pred_data, err_msg = client.predict(trip_id)

        if success:
            st.session_state.prediction_result = pred_data
            st.success(
                f"Prediction succeeded for trip '{trip_id}' in {pred_data.get('processing_time_s', 0):.2f} seconds!"
            )
        else:
            st.session_state.prediction_result = None
            st.error(f"Inference request failed: {err_msg}")

    # Display Results if available
    res = st.session_state.prediction_result
    if res:
        tab_map, tab_metrics = st.tabs(["🗺️ Route Reconstruction Map", "📊 Detailed Metrics & Summary"])

        with tab_map:
            st.markdown(f"### Reconstructed Route for Trip `{res.get('trip_id')}`")
            render_route_map(res, height=560)

        with tab_metrics:
            render_prediction_metrics(res)
    else:
        if eval_trips:
            st.info("👈 Select an evaluator trip from the sidebar and click **Run Route Prediction** to start.")


if __name__ == "__main__":
    main()
