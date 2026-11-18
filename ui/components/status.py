"""Status indicators and system overview component."""

from typing import Any, Dict, Optional
import streamlit as st


def render_system_status(health_data: Optional[Dict[str, Any]], error: Optional[str] = None):
    """Render system status overview cards."""
    col1, col2, col3, col4 = st.columns(4)

    if error or not health_data:
        with col1:
            st.metric(label="Backend API", value="Offline", delta="Disconnected", delta_color="inverse")
        with col2:
            st.metric(label="ML Model", value="Unavailable", delta=None)
        with col3:
            st.metric(label="Pipeline", value="Unavailable", delta=None)
        with col4:
            st.metric(label="Evaluator Trips", value="0", delta=None)

        st.error(f"Cannot reach backend server. {error or 'Check backend status.'}")
        return

    # Extract status fields
    status = health_data.get("status", "unknown")
    model_loaded = health_data.get("model_loaded", False)
    model_type = health_data.get("model_type", "None")
    pipeline_available = health_data.get("pipeline_available", False)
    trips_count = health_data.get("evaluator_trips_count", 0)

    with col1:
        if status == "healthy":
            st.metric(label="Backend API", value="Healthy", delta="Active")
        elif status == "degraded":
            st.metric(label="Backend API", value="Degraded", delta="Warning", delta_color="off")
        else:
            st.metric(label="Backend API", value="Unhealthy", delta="Error", delta_color="inverse")

    with col2:
        if model_loaded:
            st.metric(label="Trained ML Model", value="Ready", delta=model_type)
        else:
            st.metric(label="Trained ML Model", value="Not Loaded", delta="Missing Artifact", delta_color="inverse")

    with col3:
        if pipeline_available:
            st.metric(label="P1-P3 Pipeline", value="Connected", delta="Viterbi Ready")
        else:
            st.metric(label="P1-P3 Pipeline", value="Error", delta="Modules Missing", delta_color="inverse")

    with col4:
        st.metric(label="Evaluator Trips", value=str(trips_count), delta=f"{trips_count} Available")

    if not model_loaded:
        st.warning(
            "ML model is not loaded on the backend. Inference will fail until a compatible model artifact is present."
        )
