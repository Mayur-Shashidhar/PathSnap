"""Metrics display component."""

from typing import Any, Dict, Optional
import streamlit as st


def render_prediction_metrics(pred_data: Dict[str, Any]):
    """Render KPI metrics and summary details for a prediction response."""
    evaluation = pred_data.get("evaluation") or {}
    has_ground_truth = evaluation.get("has_ground_truth", False)

    st.markdown("### Performance & Evaluation Metrics")

    # Primary Row: Key Operational and Accuracy Metrics
    m1, m2, m3, m4, m5 = st.columns(5)

    # Route Continuity
    continuity = pred_data.get("continuity_pct", 0.0)
    with m1:
        st.metric(
            label="Route Continuity",
            value=f"{continuity:.1f}%",
            help="Percentage of consecutive distinct predicted arcs that share a connecting road node."
        )

    # Inference Time
    proc_time = pred_data.get("processing_time_s", 0.0)
    with m2:
        st.metric(
            label="Inference Latency",
            value=f"{proc_time:.2f}s",
            help="Full end-to-end processing time from GPS raw data to decoded route."
        )

    # Top-1 Accuracy (from server-side evaluation)
    top1 = evaluation.get("top1_accuracy")
    with m3:
        if has_ground_truth and top1 is not None:
            st.metric(
                label="Top-1 Accuracy",
                value=f"{top1:.1f}%",
                help="Observation-level Top-1 accuracy against server-side reference route."
            )
        else:
            st.metric(
                label="Top-1 Accuracy",
                value="N/A",
                help="Ground-truth route not available on server for this evaluator trip."
            )

    # Macro F1
    macro_f1 = evaluation.get("macro_f1")
    with m4:
        if has_ground_truth and macro_f1 is not None:
            st.metric(
                label="Macro F1-Score",
                value=f"{macro_f1:.4f}",
                help="Macro-averaged F1 score across predicted road arcs."
            )
        else:
            st.metric(
                label="Macro F1-Score",
                value="N/A",
                help="Ground truth not available for F1 calculation."
            )

    # Candidate Miss Rate
    miss_rate = evaluation.get("candidate_miss_rate")
    with m5:
        if has_ground_truth and miss_rate is not None:
            st.metric(
                label="Candidate Miss Rate",
                value=f"{miss_rate:.2f}%",
                help="Percentage of points where reference road arc was missing from candidate shortlist."
            )
        else:
            st.metric(
                label="Candidate Miss Rate",
                value="N/A",
                help="Requires reference route to evaluate candidate coverage."
            )

    # Secondary details: Reliable points accuracy and counts
    if has_ground_truth:
        top1_rel = evaluation.get("top1_accuracy_reliable")
        n_rel = evaluation.get("reliable_points")
        n_eval = evaluation.get("evaluated_points", pred_data.get("n_points"))
        miss_rel = evaluation.get("candidate_miss_rate_reliable")

        st.info(
            f"**Server-Side Evaluation Summary**: Evaluated {n_eval} points ({n_rel} reliable, ≤100m from route). "
            + (f"Top-1 Accuracy on reliable points: **{top1_rel:.2f}%**. " if top1_rel is not None else "")
            + (f"Reliable Miss Rate: **{miss_rel:.2f}%**." if miss_rel is not None else "")
        )
    else:
        st.info(
            "ℹ️ **Evaluator Trip Mode**: Reference route ground truth is held server-side and was not provided with this trip. "
            "Pipeline successfully produced candidate predictions, feature representations, classifier probabilities, and Viterbi-decoded route."
        )

    # Prediction details summary table/card
    st.markdown("#### Trip Execution Details")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.write(f"**Trip ID:** `{pred_data.get('trip_id')}`")
    with d2:
        st.write(f"**GPS Points:** `{pred_data.get('n_points')}`")
    with d3:
        st.write(f"**Predicted Route Arcs:** `{pred_data.get('n_route_arcs')}`")
    with d4:
        st.write(f"**Status:** `{pred_data.get('status')}`")
