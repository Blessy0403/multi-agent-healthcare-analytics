"""Explainability page — SHAP, LIME, and natural language explanations. Dataset-aware: only shows artifacts for the selected dataset and model."""

import streamlit as st
import json
import re
from pathlib import Path
from typing import List, Optional

from dashboard.components.styling import render_header, render_agent_section
from dashboard.ui import section_header

if "run_id" not in st.session_state and "selected_run_id" not in st.session_state:
    st.error("Please select a run from the sidebar.")
    st.stop()

run_id = st.session_state.get("selected_run_id") or st.session_state.get("run_id")
from dashboard.components.layout import get_run_dir_path, resolve_paths

dataset = st.session_state.get("selected_dataset_key") or st.session_state.get("dataset", "diabetes")
model = st.session_state.get("selected_model_key") or st.session_state.get("model", "logistic_regression")

dataset_key = str(dataset).strip().lower().replace(" ", "_")
if dataset_key != "diabetes":
    render_header("Model Explainability", "")
    render_agent_section("explainability_agent", "#6a51a3")
    st.caption("Outputs from the **Explainability Agent**: SHAP, LIME, and natural language explanations.")
    st.warning("Explainability visualizations are currently enabled only for the Diabetes dataset demonstration run. Please select Diabetes to view SHAP, LIME, waterfall, and local explanation outputs.")
    st.caption(f"Explainability guard active for dataset: {dataset_key}")
    st.stop()

def _normalize_model_key(key: str) -> str:
    """Use model key as-is for filenames (e.g. logistic_regression, gradient_boosting)."""
    return (key or "logistic_regression").strip().lower().replace(" ", "_")

dataset = dataset_key
model = _normalize_model_key(model)

run_dir = get_run_dir_path(run_id)
paths = resolve_paths(run_dir)
explainability_dir = paths["explainability_dir"]
figures_dir = paths["figures_dir"]


def _parse_dataset_prefix_from_shap_filename(filename: str) -> Optional[str]:
    """Parse dataset prefix from SHAP artifact filename. diabetes_* -> diabetes; heart_disease_* -> heart_disease."""
    if not filename:
        return None
    name = str(filename).strip().lower()
    if name.startswith("diabetes_"):
        return "diabetes"
    if name.startswith("heart_disease_"):
        return "heart_disease"
    return None


def find_dataset_model_plot(figures_dir: Path, dataset_key: str, model_key: str, suffix: str) -> Optional[Path]:
    """
    Find a plot file that explicitly contains BOTH dataset and model in the filename.
    Only accepts: {dataset}_{model}_{suffix}. No fallback to model-only files.
    """
    if not figures_dir.exists():
        return None
    path = figures_dir / f"{dataset_key}_{model_key}_{suffix}"
    if path.exists() and path.is_file():
        return path
    return None


def find_dataset_model_files(
    base_dir: Path, dataset_key: str, model_key: str, pattern: str
) -> List[Path]:
    """
    Find files under base_dir whose name contains both dataset_key and model_key.
    pattern is a glob pattern (e.g. '*shap_waterfall_*.png'). Only returns files
    that include both dataset_key and model_key in the stem/filename.
    """
    if not base_dir.exists() or not base_dir.is_dir():
        return []
    prefix = f"{dataset_key}_{model_key}_"
    collected = []
    for f in base_dir.glob(pattern):
        if f.is_file() and f.name.startswith(prefix):
            collected.append(f)
    return sorted(collected)


# ----- Debug expander (top of page) -----
with st.expander("Debug: dataset & artifact resolution", expanded=False):
    summary_path = find_dataset_model_plot(figures_dir, dataset, model, "shap_summary.png")
    bar_path = find_dataset_model_plot(figures_dir, dataset, model, "shap_bar.png")
    waterfall_files = find_dataset_model_files(figures_dir, dataset, model, "*shap_waterfall_*.png")
    st.write("**Selected dataset (artifact key):**", dataset)
    st.write("**Selected model:**", model)
    st.write("**figures_dir:**", str(figures_dir))
    st.write("**Resolved SHAP summary path:**", str(summary_path) if summary_path else "—")
    st.write("**Resolved SHAP bar path:**", str(bar_path) if bar_path else "—")
    st.write("**Count of resolved waterfall files:**", len(waterfall_files))


render_header("Model Explainability", "")
render_agent_section("explainability_agent", "#6a51a3")
st.caption("Outputs from the **Explainability Agent**: SHAP, LIME, and natural language explanations.")

# ----- SHAP (dataset-specific only; no fallback to model-only files) -----
section_header("SHAP Explanations", caption="Global and per-feature importance for the selected model.")

summary_path = find_dataset_model_plot(figures_dir, dataset, model, "shap_summary.png")
bar_path = find_dataset_model_plot(figures_dir, dataset, model, "shap_bar.png")

if not figures_dir.exists():
    st.info("Figures directory not found. Run the pipeline to generate SHAP plots.")
elif summary_path is None and bar_path is None:
    st.warning(
        "Dataset-specific SHAP artifacts were not found for the selected dataset and model. "
        "Only files matching the selected dataset and model (e.g. "
        f"`{dataset}_{model}_shap_summary.png`) are shown to avoid cross-dataset leakage."
    )
else:
    col1, col2 = st.columns(2)
    with col1:
        if summary_path and summary_path.exists():
            parsed = _parse_dataset_prefix_from_shap_filename(summary_path.name)
            if parsed != dataset:
                st.error(f"Artifact mismatch: selected dataset is **{dataset}**, but SHAP file is **{parsed or 'unknown'}**. Not rendered.")
            else:
                st.image(str(summary_path), use_container_width=True)
                st.caption("SHAP summary")
        else:
            st.info("SHAP summary not available for this dataset/model.")
    with col2:
        if bar_path and bar_path.exists():
            parsed = _parse_dataset_prefix_from_shap_filename(bar_path.name)
            if parsed != dataset:
                st.error(f"Artifact mismatch: selected dataset is **{dataset}**, but SHAP bar file is **{parsed or 'unknown'}**. Not rendered.")
            else:
                st.image(str(bar_path), use_container_width=True)
                st.caption("Mean |SHAP|")
        else:
            st.info("SHAP bar not available for this dataset/model.")

st.markdown("---")
section_header("Individual Predictions (SHAP Waterfall)", caption=None)

waterfall_files = find_dataset_model_files(figures_dir, dataset, model, "*shap_waterfall_*.png")
if not figures_dir.exists():
    st.info("Figures directory not found.")
elif not waterfall_files:
    st.warning(
        "Dataset-specific SHAP waterfall artifacts were not found for the selected dataset and model. "
        f"Expected files like `{dataset}_{model}_shap_waterfall_*.png`."
    )
else:
    selected_idx = st.selectbox(
        "Select instance",
        range(len(waterfall_files)),
        format_func=lambda x: f"Instance {x}",
    )
    wf_path = waterfall_files[selected_idx]
    parsed_wf = _parse_dataset_prefix_from_shap_filename(wf_path.name)
    if parsed_wf != dataset:
        st.error(f"Artifact mismatch: selected dataset is **{dataset}**, but SHAP waterfall file is **{parsed_wf or 'unknown'}**. Not rendered.")
    else:
        st.image(str(wf_path), use_container_width=True)

# ----- LIME (dataset-specific only) -----
st.markdown("---")
section_header("LIME Explanations", caption=None)

lime_path = explainability_dir / f"lime_explanations_{dataset}_{model}.json"
if not lime_path.exists():
    lime_path = None

if lime_path is not None and lime_path.exists():
    with open(lime_path, "r") as f:
        lime_data = json.load(f)
    explanations = lime_data.get("explanations", {})
    feature_names = lime_data.get("feature_names", [])

    if explanations:
        instance_ids = sorted(explanations.keys(), key=lambda x: int(x) if x.isdigit() else 0)
        selected_id = st.selectbox("Select instance", instance_ids, format_func=lambda x: f"Instance {x}")
        exp = explanations.get(selected_id, {})
        if exp:
            pred = exp.get("prediction", [])
            if len(pred) >= 2:
                st.caption(f"Predicted probabilities: No Disease = {pred[0]:.3f}, Disease = {pred[1]:.3f}")
            weights = exp.get("feature_weights", [])
            if weights:
                st.markdown("**Feature contributions (LIME)**")
                for feat, w in weights:
                    st.markdown(f"- **{feat}**: {w:+.3f}")
            else:
                st.json(exp)
    else:
        st.info("No per-instance LIME explanations in this file.")
else:
    st.warning(
        "Dataset-specific LIME explanations were not found for the selected dataset and model. "
        f"Expected file: `lime_explanations_{dataset}_{model}.json`. "
        "Generic or other-dataset LIME files are not shown to avoid cross-dataset leakage."
    )

# ----- Natural Language (dataset-specific only) -----
st.markdown("---")
section_header("Natural Language Explanations", caption=None)

nl_path = explainability_dir / f"natural_language_{dataset}_{model}.txt"
if not nl_path.exists():
    nl_path = None

if nl_path is not None and nl_path.exists():
    with open(nl_path, "r") as f:
        full_text = f.read()

    # Parse by model sections: "==========...\nModel: xxx\n==========...\n\n"
    sections = re.split(r"\n=+\nModel:\s*(\w+)\n=+", full_text)
    model_to_text = {}
    if len(sections) >= 2:
        for i in range(1, len(sections), 2):
            model_name = sections[i].strip()
            model_to_text[model_name] = sections[i + 1].strip() if i + 1 < len(sections) else ""
    if not model_to_text and full_text.strip():
        model_to_text["_all_"] = full_text.strip()

    # Show section for current model (normalize for display: model might be "gradient_boosting" -> "Gradient Boosting")
    model_display = model.replace("_", " ").title()
    nl_section = model_to_text.get(model, model_to_text.get(model_display, model_to_text.get("_all_", full_text)))
    if nl_section:
        st.text_area("Explanation", nl_section, height=350, label_visibility="collapsed")
    else:
        st.text_area("Explanation", full_text, height=350, label_visibility="collapsed")
else:
    st.warning(
        "Dataset-specific natural language explanations were not found for the selected dataset and model. "
        f"Expected file: `natural_language_{dataset}_{model}.txt`. "
        "Generic or other-dataset files are not shown to avoid cross-dataset leakage."
    )
