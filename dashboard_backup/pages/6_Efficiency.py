"""Collaboration efficiency page — thesis-level charts."""

import streamlit as st
import json
import pandas as pd
from pathlib import Path
from dashboard.components.styling import render_header, render_agent_section
from dashboard.ui import section_header, kpi_card
from dashboard.components.charts import plot_agent_timeline

if "run_id" not in st.session_state and "selected_run_id" not in st.session_state:
    st.error("Please select a run from the sidebar.")
    st.stop()

run_id = st.session_state.get("selected_run_id") or st.session_state.get("run_id")
from dashboard.components.layout import get_run_dir_path, resolve_paths
render_header("Collaboration Efficiency", "")
render_agent_section("model_agent", "#238b45")
st.caption("Per-agent execution times (each agent has a distinct color in the pipeline).")

run_dir = get_run_dir_path(run_id)
paths = resolve_paths(run_dir)

# Load collaboration data: try evaluation report first, then raw metrics (resolved paths)
collab_path = paths["reports_dir"] / "collaboration_evaluation.json"
if not collab_path.exists():
    collab_path = paths["logs_dir"] / "collaboration_metrics.json"

if not collab_path.exists():
    st.warning("Collaboration metrics not found for this run. Complete a full pipeline run to generate them.")
    st.caption("Expected: reports/collaboration_evaluation.json or logs/collaboration_metrics.json")
    st.stop()

with open(collab_path, "r") as f:
    collab_data = json.load(f)

# Normalize structure: evaluation has execution_times.per_agent; raw metrics have agent_execution_times
agent_times = None
if "execution_times" in collab_data and "per_agent" in collab_data["execution_times"]:
    agent_times = collab_data["execution_times"]["per_agent"]
elif "agent_execution_times" in collab_data:
    agent_times = collab_data["agent_execution_times"]

if agent_times and len(agent_times) > 0:
    section_header("Agent Execution Times", caption=None)
    fig = plot_agent_timeline(agent_times)
    st.plotly_chart(fig, use_container_width=True)
    times_df = pd.DataFrame(
        list(agent_times.items()), columns=["Agent", "Time (s)"]
    )
    total = times_df["Time (s)"].sum()
    times_df["Share (%)"] = (times_df["Time (s)"] / total * 100).round(1)
    st.dataframe(times_df, use_container_width=True)
else:
    st.info("No per-agent execution times in this report.")

st.markdown("---")
section_header("Handover Analysis", caption=None)

handover = collab_data.get("handover_analysis", {})
if not handover and "handovers" in collab_data:
    handovers = collab_data["handovers"]
    handover = {
        "num_handovers": len(handovers),
        "avg_handover_time": 0,
        "handovers": handovers,
    }

col1, col2 = st.columns(2)
with col1:
    kpi_card("Number of handovers", handover.get("num_handovers", 0))
with col2:
    kpi_card("Avg handover time", f"{handover.get('avg_handover_time', 0):.3f}s")

st.markdown("---")
section_header("Error Analysis", caption=None)

errors = collab_data.get("error_analysis", {})
if not errors and "errors" in collab_data:
    err_list = collab_data["errors"]
    errors = {"num_errors": len(err_list), "error_rate": 0, "errors": err_list}

col1, col2 = st.columns(2)
with col1:
    kpi_card("Number of errors", errors.get("num_errors", 0))
with col2:
    kpi_card("Error rate", f"{errors.get('error_rate', 0):.3%}")

if errors.get("num_errors", 0) > 0 and "errors" in errors:
    with st.expander("Error details"):
        for err in errors["errors"]:
            st.json(err)

st.markdown("---")
section_header("Efficiency Metrics", caption=None)

eff = collab_data.get("efficiency_metrics", {})
if not eff and "total_execution_time" in collab_data:
    # Fallback when only raw collaboration_metrics.json exists (no evaluation report yet)
    raw_errors = collab_data.get("errors", [])
    success_rate = 1.0 if len(raw_errors) == 0 else max(0.0, 1.0 - len(raw_errors) / max(1, len(collab_data.get("handovers", []))))
    total_time = collab_data.get("total_execution_time", 0)
    agent_times_for_overhead = agent_times or {}
    sum_agent = sum(agent_times_for_overhead.values())
    overhead = ((total_time - sum_agent) / total_time * 100) if total_time > 0 else 0
    eff = {
        "pipeline_status": collab_data.get("pipeline_status", "success" if len(raw_errors) == 0 else "unknown"),
        "total_time": total_time,
        "success_rate": success_rate,
        "total_agents": max(4, len(agent_times_for_overhead)) or 4,
        "collaboration_overhead": overhead,
    }

if eff:
    # When pipeline status is success, success rate must not show 0% (evaluator may have saved 0 if handovers was empty)
    success_rate = eff.get("success_rate", 0)
    if (eff.get("pipeline_status") == "success" and success_rate == 0):
        success_rate = 1.0
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        kpi_card("Pipeline status", str(eff.get("pipeline_status", "—")), max_value_len=12)
    with col2:
        kpi_card("Success rate", f"{success_rate:.3%}")
    with col3:
        # Pipeline has 4 agents: Data, Model, Explainability, Evaluation (thesis architecture)
        total_agents = eff.get("total_agents") or (len(agent_times) if agent_times else 0)
        kpi_card("Total agents", max(4, total_agents) if total_agents else 4)
    with col4:
        kpi_card("Overhead", f"{eff.get('collaboration_overhead', 0):.3f}%")
else:
    total_time = collab_data.get("total_execution_time")
    if total_time is not None:
        kpi_card("Total execution time", f"{total_time:.3f}s")