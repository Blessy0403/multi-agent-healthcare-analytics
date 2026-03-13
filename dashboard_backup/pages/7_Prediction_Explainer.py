"""Interactive prediction explainer page."""

import json
import re
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from dashboard.components.styling import render_header
from utils.json_utils import json_safe
from dashboard.components.artifacts import load_run_artifacts, resolve_dataset_model_fallback, safe_read_csv, safe_read_json

if "run_id" not in st.session_state and "selected_run_id" not in st.session_state:
    st.error("Please select a run from the sidebar.")
    st.stop()

run_id = st.session_state.get("selected_run_id") or st.session_state.get("run_id")
from dashboard.components.layout import get_run_dir_path, resolve_paths
dataset = st.session_state.get("selected_dataset_key") or st.session_state.get("dataset", "heart_disease")
model = st.session_state.get("selected_model_key") or st.session_state.get("model", "logistic_regression")

run_dir = Path(st.session_state.get("run_dir") or get_run_dir_path(run_id))
artifacts = load_run_artifacts(run_dir, dataset)
resolved_dataset, resolved_model, fallback_warning = resolve_dataset_model_fallback(artifacts, dataset, model)
if fallback_warning:
    st.warning(fallback_warning)
dataset = resolved_dataset
model = resolved_model

paths = {k: Path(v) for k, v in (artifacts.get("files") or {}).get("paths", {}).items()}
if not paths:
    paths = resolve_paths(run_dir)

render_header("Interactive Prediction Explainer", "")
st.caption("Enter feature values to get a prediction and see feature contributions.")

model_path = paths.get("models_dir", run_dir / "models")
if not hasattr(model_path, "exists"):
    model_path = Path(model_path)
model_path = model_path / f"{model}.pkl"
if not model_path.exists():
    model_path = run_dir / "multi_agent" / "models" / f"{model}.pkl"
if not model_path.exists():
    model_path = run_dir / "models" / f"{model}.pkl"
if not model_path.exists():
    st.error(f"Model **{model}** not found for this run. Select another model in the sidebar or run the pipeline.")
    st.stop()

with open(model_path, "rb") as f:
    loaded_model = pickle.load(f)

# Feature names and ranges from profile (data_profile.json) or metadata (data_metadata)
data_section = artifacts.get("data") or {}
metadata = data_section.get("metadata") or {}
if not metadata and data_section.get("metadata_path"):
    metadata = safe_read_json(Path(data_section["metadata_path"]))
profile = data_section.get("profile") or {}
target_col = profile.get("target_column") or metadata.get("target_column", "target")
feature_cols = [c for c in (profile.get("columns") or []) if c != target_col]
if not feature_cols and metadata.get("raw_columns"):
    feature_cols = [c for c in metadata["raw_columns"] if c != target_col]
feature_stats = profile.get("feature_stats") or {}

# Test data (optional): if present, offer "Select from Test Set"
data_dir = Path(data_section.get("data_dir") or str(paths.get("data_dir", run_dir / "data")))
test_path = data_dir / f"{dataset}_test.csv"
test_df = safe_read_csv(test_path) if test_path.exists() else None
if test_df is not None and not test_df.empty and not feature_cols:
    feature_cols = [c for c in test_df.columns if c != target_col]
if not feature_cols and metadata.get("raw_columns"):
    feature_cols = [c for c in metadata["raw_columns"] if c != target_col]
# Known fallbacks for common datasets
if not feature_cols:
    if "diabetes" in dataset.lower():
        feature_cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
    elif "heart" in dataset.lower():
        feature_cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalch", "exang", "oldpeak", "slope", "ca", "thal"]
if not feature_cols:
    st.error("No feature list found for this run/dataset. Run the pipeline to generate data_profile or data_metadata.")
    st.stop()

# Input method: "Select from Test Set" only if test data exists and has all feature columns
test_has_features = test_df is not None and not test_df.empty and all(f in test_df.columns for f in feature_cols)
if test_has_features:
    input_method = st.radio("Input Method", ["Select from Test Set", "Manual Input"])
else:
    input_method = "Manual Input"

if input_method == "Select from Test Set" and test_df is not None and not test_df.empty:
    instance_idx = st.selectbox("Select Instance", range(len(test_df)), format_func=lambda i: f"Instance {i}")
    instance = test_df.iloc[instance_idx]
    st.subheader("Selected Instance")
    instance_display = instance.drop(labels=[target_col], errors="ignore") if target_col in instance.index else instance
    st.dataframe(instance_display.to_frame().T, use_container_width=True)
    X_input = np.array([[instance[f] for f in feature_cols]])
else:
    st.subheader("Enter Feature Values")
    feature_values = {}
    for feat in feature_cols:
        stats = feature_stats.get(feat, {})
        min_val = stats.get("min")
        max_val = stats.get("max")
        default_val = stats.get("median")
        if test_df is not None and feat in test_df.columns:
            min_val = float(test_df[feat].min()) if min_val is None else min_val
            max_val = float(test_df[feat].max()) if max_val is None else max_val
            default_val = float(test_df[feat].median()) if default_val is None else default_val
        if default_val is None:
            default_val = 0.0
        if min_val is None:
            min_val = -1e6
        if max_val is None:
            max_val = 1e6
        feature_values[feat] = st.number_input(
            feat, min_value=float(min_val), max_value=float(max_val), value=float(default_val)
        )
    X_input = np.array([[feature_values[f] for f in feature_cols]])

# Predict
try:
    prediction = loaded_model.predict(X_input)[0]
    prediction_proba = loaded_model.predict_proba(X_input)[0] if hasattr(loaded_model, "predict_proba") else None
except Exception as e:
    st.error(f"Prediction failed: {e}")
    prediction = 0
    prediction_proba = np.array([0.5, 0.5])

prob = float(prediction_proba[1]) if prediction_proba is not None and len(prediction_proba) > 1 else (float(prediction_proba[0]) if prediction_proba is not None else 0.5)
pred_class = "Disease" if prediction == 1 else "No Disease"
prob_pct = prob * 100 if prediction == 1 else (1 - prob) * 100
bg_color = "#dcfce7" if prediction == 0 else "#fef3c7"
border_color = "#16a34a" if prediction == 0 else "#d97706"
icon = "✓" if prediction == 0 else "⚠"

st.markdown("---")
st.subheader("Prediction")
st.markdown(f"""
<div style="
    background: {bg_color};
    border-left: 4px solid {border_color};
    border-radius: 0 12px 12px 0;
    padding: 1.25rem 1.5rem;
    margin: 0.5rem 0 1rem 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
">
    <div style="font-size: 0.85rem; color: #4b5563; font-weight: 600; text-transform: uppercase;">Predicted class</div>
    <div style="font-size: 1.75rem; font-weight: 700; color: #1a1a2e; margin-top: 0.25rem;">{icon} {pred_class}</div>
    <div style="font-size: 0.9rem; color: #6b7280; margin-top: 0.5rem;">Confidence: <strong>{prob_pct:.1f}%</strong> (probability of positive class: {prob:.4f})</div>
</div>
""", unsafe_allow_html=True)

# Feature contributions: SHAP if available, else LR coefficients, else permutation or message
st.markdown("---")
st.subheader("Feature Contributions")

explainability_dir = paths.get("explainability_dir", run_dir / "explainability")
if not hasattr(explainability_dir, "exists"):
    explainability_dir = Path(explainability_dir)
figures_dir = paths.get("figures_dir", run_dir / "figures")
if not hasattr(figures_dir, "exists"):
    figures_dir = Path(figures_dir)

contrib_shown = False
# 1) SHAP from explanations.json or shap_values_*.json
for name in [f"shap_values_{dataset}_{model}.json", "explanations.json"]:
    shap_path = explainability_dir / name
    if shap_path.exists():
        try:
            shap_data = safe_read_json(shap_path)
            if shap_data and "shap" in shap_data and model in shap_data["shap"]:
                mean_shap = shap_data["shap"][model].get("mean_abs_shap", [])
                names = shap_data["shap"][model].get("feature_names", [])
                if mean_shap and names:
                    import plotly.graph_objects as go
                    fig = go.Figure(go.Bar(x=mean_shap, y=list(names)[:20], orientation="h"))
                    fig.update_layout(title="Mean |SHAP| (feature importance)", xaxis_title="Mean |SHAP|", height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    contrib_shown = True
                    break
        except Exception:
            pass
# 2) Fallback: Logistic Regression coefficients
if not contrib_shown and hasattr(loaded_model, "coef_"):
    coef = np.ravel(loaded_model.coef_)
    if len(coef) == len(feature_cols):
        import plotly.graph_objects as go
        imp = np.abs(coef)
        fig = go.Figure(go.Bar(x=imp, y=feature_cols, orientation="h"))
        fig.update_layout(title="Feature importance (model coefficients)", xaxis_title="|Coefficient|", height=400)
        st.plotly_chart(fig, use_container_width=True)
        contrib_shown = True
# 3) Waterfall image if available
if figures_dir.exists():
    waterfall_files = sorted([f for f in figures_dir.iterdir() if f.is_file() and f.name.startswith(f"{model}_shap_waterfall_")])
    if waterfall_files:
        idx_sel = st.selectbox("SHAP waterfall instance", range(len(waterfall_files)), format_func=lambda x: f"Instance {x}")
        st.image(str(waterfall_files[idx_sel]), use_container_width=True)
        contrib_shown = True
if not contrib_shown:
    st.info("SHAP or coefficient-based explanations not available for this run/model. Run the full pipeline to generate SHAP, or view coefficients above for linear models.")

# Natural language explanation
st.markdown("---")
st.subheader("Natural Language Explanation")
nl_path = explainability_dir / f"natural_language_{dataset}_{model}.txt"
if not nl_path.exists():
    nl_path = explainability_dir / "natural_language_explanations.txt"
if nl_path.exists():
    try:
        full_text = nl_path.read_text(encoding="utf-8", errors="replace")
        sections = re.split(r"\n=+\nModel:\s*(\w+)\n=+", full_text)
        nl_section = None
        if len(sections) >= 2:
            for i in range(1, len(sections), 2):
                if i + 1 < len(sections) and sections[i].strip() == model:
                    nl_section = sections[i + 1].strip()
                    break
        st.text_area("Explanation", nl_section or full_text.strip() or "(No explanation text)", height=280, label_visibility="collapsed")
    except Exception:
        st.info(f"The model predicts **{pred_class}** with {prob*100:.1f}% confidence.")
else:
    st.info(f"The model predicts **{pred_class}** with {prob*100:.1f}% confidence. This prediction is based on the feature values provided.")

# Export
st.markdown("---")
st.subheader("Export Explanation")
explanation_data = {
    "run_id": run_id,
    "dataset": dataset,
    "model": model,
    "prediction": int(prediction),
    "probability": float(prob),
    "predicted_class": pred_class,
    "feature_values": {f: float(v) for f, v in zip(feature_cols, X_input[0])},
}
st.download_button(
    "Download as JSON",
    json.dumps(explanation_data, indent=2, default=json_safe),
    file_name=f"explanation_{run_id}_{dataset}_{model}.json",
    mime="application/json",
)
