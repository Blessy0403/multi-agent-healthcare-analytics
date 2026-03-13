"""Overview page for dashboard."""

import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
from dashboard.components.styling import render_header, render_pipeline_flow, render_agent_section
from dashboard.components.charts import plot_metrics_comparison
from dashboard.ui import section_header, kpi_card
from dashboard.components.artifacts import load_run_artifacts, resolve_dataset_model_fallback, get_run_health_summary, safe_read_csv

if "run_id" not in st.session_state and "selected_run_id" not in st.session_state:
    st.error("Please select a run from the sidebar.")
    st.stop()

run_id = st.session_state.get("run_id") or st.session_state.get("selected_run_id")
from dashboard.components.layout import get_run_dir_path
dataset = st.session_state.get("selected_dataset_key") or st.session_state.get("dataset", "heart_disease")
model = st.session_state.get("selected_model_key") or st.session_state.get("model", "logistic_regression")

_run_dir_raw = st.session_state.get("run_dir")
run_dir = Path(_run_dir_raw) if _run_dir_raw else get_run_dir_path(run_id)
run_dir = Path(run_dir)
artifacts = load_run_artifacts(run_dir, dataset)
resolved_dataset, resolved_model, fallback_warning = resolve_dataset_model_fallback(artifacts, dataset, model)
if fallback_warning:
    st.warning(fallback_warning)
dataset = resolved_dataset
model = resolved_model

metadata = artifacts.get("run_metadata") or {}
if not metadata.get("run_id"):
    metadata["run_id"] = run_id
if not metadata.get("status"):
    metadata["status"] = "unknown"
if not metadata.get("timestamp"):
    metadata["timestamp"] = datetime.now().isoformat()
if "datasets" not in metadata or not metadata["datasets"]:
    metadata["datasets"] = [dataset] if dataset else ["diabetes"]

paths = {k: Path(v) for k, v in (artifacts.get("files") or {}).get("paths", {}).items()}
reports_dir = paths.get("reports_dir", run_dir / "reports")

if metadata.get("status") == "unknown" and run_dir.exists():
    models_dir = paths.get("models_dir", run_dir / "models")
    has_models = models_dir.exists() and any(models_dir.iterdir())
    has_reports = reports_dir.exists() and any(reports_dir.iterdir())
    if has_models and has_reports:
        metadata["status"] = "success"
    elif reports_dir.exists():
        metadata["status"] = "running"

render_header("Pipeline Overview", "")
section_header("Run summary", caption="Best models, KPIs, and baseline comparison for the selected run.")
render_pipeline_flow()

# Run health panel: which key artifacts exist / missing
health = get_run_health_summary(artifacts)
with st.expander("Run health — key artifacts", expanded=True):
    labels = {
        "run_metadata": "Run metadata",
        "data": "Data",
        "models": "Models",
        "evaluation": "Evaluation",
        "explainability": "Explainability",
        "baseline": "Baseline",
        "cross_dataset": "Cross-dataset",
    }
    cols = st.columns(min(len(health), 4))
    for idx, (key, exists) in enumerate(health.items()):
        label = labels.get(key, key.replace("_", " ").title())
        with cols[idx % len(cols)]:
            if exists:
                st.markdown(f"<span style='color:#38a169'>● {label}</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span style='color:#e53e3e'>○ {label}</span>", unsafe_allow_html=True)

# Run summary
from dashboard.components.layout import format_run_name
formatted_run_name = format_run_name(run_id)
col1, col2, col3, col4 = st.columns(4)
with col1:
    kpi_card("Run ID", formatted_run_name, max_value_len=22)
with col2:
    timestamp = metadata.get('timestamp') or 'N/A'
    ts_str = str(timestamp)
    kpi_card("Created", ts_str[:10] if len(ts_str) > 10 else ts_str, max_value_len=14)
with col3:
    status = metadata.get('status', 'unknown')
    kpi_card("Status", "Success" if status == 'success' else ("Failed" if status == 'failed' else "Unknown"), max_value_len=10)
with col4:
    total_time = metadata.get('total_execution_time', 0)
    kpi_card("Total Time", f"{total_time:.3f}s", max_value_len=12)

# Treat as failed if status is 'failed' or run ended with 0s (crashed before saving)
status = metadata.get('status', 'unknown')
total_time = metadata.get('total_execution_time', 0)
is_failed = status == 'failed' or (total_time == 0 and status != 'success')
if is_failed:
    st.markdown("---")
    err_msg = metadata.get('error', 'Unknown error')
    st.error(f"**Pipeline failed:** {err_msg}")
    st.caption("Full traceback is in the **Run Log** section below. Select a run with ✅ Success in the sidebar to see full metrics.")
    run_log_path = paths.get("logs_dir", run_dir / "logs") / "run.log"
    if run_log_path.exists():
        try:
            with open(run_log_path, 'r', encoding='utf-8') as f:
                log_content = f.read()
            with st.expander("📋 Run log (failure details)", expanded=True):
                st.text_area("Pipeline log", log_content, height=250, disabled=True, label_visibility="collapsed")
        except Exception as e:
            st.warning(f"Could not read log file: {e}")

st.markdown("---")
render_agent_section("model_agent", "#238b45")
section_header("Best Models", caption=None)
if 'datasets' in metadata:
    cols = st.columns(len(metadata['datasets']) if metadata['datasets'] else 1)
    for idx, ds in enumerate(metadata.get('datasets', [])):
        with cols[idx] if idx < len(cols) else st.container():
            best_model_key = f'best_model_{ds}'
            if best_model_key in metadata:
                best_info = metadata[best_model_key]
                model_name = best_info.get('model', 'N/A').replace('_', ' ').title()
                roc_auc = best_info.get('roc_auc', 0)
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
                    padding: 1.5rem;
                    border-radius: 12px;
                    border-left: 4px solid #667eea;
                    margin-bottom: 1rem;
                ">
                    <h4 style="margin: 0 0 0.5rem 0; color: #2d3748;">{ds.replace('_', ' ').title()}</h4>
                    <p style="margin: 0; font-size: 1.1rem; font-weight: 600; color: #667eea;">{model_name}</p>
                    <p style="margin: 0.5rem 0 0 0; color: #718096;">ROC-AUC: <strong>{roc_auc:.4f}</strong></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Try alternative format
                best_model_ma = metadata.get('best_model_multi_agent', 'N/A')
                best_roc_auc = metadata.get('best_model_roc_auc_ma', 0)
                if best_model_ma != 'N/A':
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
                        padding: 1.5rem;
                        border-radius: 12px;
                        border-left: 4px solid #667eea;
                        margin-bottom: 1rem;
                    ">
                        <h4 style="margin: 0 0 0.5rem 0; color: #2d3748;">{ds.replace('_', ' ').title()}</h4>
                        <p style="margin: 0; font-size: 1.1rem; font-weight: 600; color: #667eea;">{best_model_ma.replace('_', ' ').title()}</p>
                        <p style="margin: 0.5rem 0 0 0; color: #718096;">ROC-AUC: <strong>{best_roc_auc:.4f}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)

st.markdown("---")
section_header("Key Performance Indicators", caption=None)
metrics_data = (artifacts.get("models") or {}).get("metrics")

if metrics_data:
    # Handle different metric file structures
    if isinstance(metrics_data, dict):
        # Try to find model metrics
        model_metrics = None
        if model in metrics_data:
            model_metrics = metrics_data[model]
        elif dataset in metrics_data and isinstance(metrics_data[dataset], dict):
            if model in metrics_data[dataset]:
                model_metrics = metrics_data[dataset][model]
            elif metrics_data[dataset]:
                # Use first available model
                model_metrics = list(metrics_data[dataset].values())[0]
        else:
            # Try to find any model metrics
            for key, value in metrics_data.items():
                if isinstance(value, dict) and 'accuracy' in value:
                    model_metrics = value
                    break
        
        if model_metrics and isinstance(model_metrics, dict) and 'accuracy' in model_metrics:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                kpi_card("Accuracy", model_metrics.get('accuracy', 0))
            with col2:
                kpi_card("ROC-AUC", model_metrics.get('roc_auc', 0))
            with col3:
                kpi_card("F1-Score", model_metrics.get('f1_score', 0))
            with col4:
                kpi_card("Precision", model_metrics.get('precision', 0))
        else:
            st.info("ℹ️ Model metrics structure not recognized. The pipeline may still be processing.")
    else:
        st.info("ℹ️ Model metrics file format not recognized.")
else:
    if is_failed:
        st.info("ℹ️ Model metrics were not generated because this run failed. Check the error and Run log above.")
    else:
        st.info("ℹ️ Model metrics file not found. The pipeline may still be processing.")

# Generalization Gap (same block as Cross_Dataset page)
st.markdown("---")
try:
    from dashboard.components.generalization import render_generalization_gap_block
    render_generalization_gap_block(run_dir)
except Exception as e:
    st.caption(f"Generalization gap unavailable: {e}")

# Feedback & iterations (multi-agent)
st.markdown("---")
st.markdown("### 🔄 Feedback & Iterations")
feedback_data = (artifacts.get("evaluation") or {}).get("feedback") or {}
if feedback_data:
    try:
        fd = feedback_data.get('feedback_decision') or {}
        action_taken = feedback_data.get('action_taken') or fd.get('action') or fd.get('decision') or 'accept'
        reason = fd.get('reason') or '—'
        iteration_count = feedback_data.get('iteration_count', 1)
        retraining_status = feedback_data.get('retraining_status', fd.get('retrained', False))
        iteration_2_ran = feedback_data.get('iteration_2_ran', False)
        trigger_name = fd.get('trigger_metric_name')
        trigger_val = fd.get('trigger_metric_value')
        threshold = fd.get('threshold')
        decision = fd.get('decision', action_taken)
        model_before = fd.get('selected_model_before')
        model_after = fd.get('selected_model_after') or fd.get('selected_model_after_feedback')
        try:
            trigger_str = f"{float(trigger_val):.4f}" if trigger_val is not None else ""
        except (TypeError, ValueError):
            trigger_str = str(trigger_val) if trigger_val is not None else ""
        lines = [
            f"**Decision:** {decision} — {reason}",
            f"**Trigger:** {trigger_name}={trigger_str} (threshold={threshold})" if trigger_name and trigger_str else None,
            f"**Model before feedback:** {model_before}" if model_before else None,
            f"**Model after feedback:** {model_after}" if model_after else None,
            f"**Retraining applied:** {'Yes' if retraining_status else 'No'}",
            f"**Iteration 2 ran:** {'Yes' if iteration_2_ran else 'No'}",
        ]
        st.info("\n".join(l for l in lines if l))
        if iteration_2_ran:
            m1 = feedback_data.get('iteration_1_selected_metrics') or {}
            m2 = feedback_data.get('iteration_2_selected_metrics') or {}
            sm1 = feedback_data.get('iteration_1_selected_model') or '—'
            sm2 = feedback_data.get('iteration_2_selected_model') or '—'
            metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
            rows = []
            for mn in metric_names:
                v1 = m1.get(mn)
                v2 = m2.get(mn)
                if v1 is not None or v2 is not None:
                    rows.append({'metric': mn, 'iteration_1': v1 if v1 is not None else '—', 'iteration_2': v2 if v2 is not None else '—'})
            if rows:
                df = pd.DataFrame(rows)
                st.caption(f"Selected model: Iteration 1 = {sm1}  |  Iteration 2 = {sm2}")
                st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.caption(f"Could not load feedback data: {e}")
else:
    st.caption("Feedback data is saved when the multi-agent pipeline runs (with FeedbackAgent).")

# Baseline vs Multi-Agent Comparison
st.markdown("---")
st.markdown("### ⚖️ Baseline vs Multi-Agent Comparison")
evaluation = artifacts.get("evaluation") or {}
baseline_comparison = evaluation.get("baseline_comparison") or {}
baseline_results = evaluation.get("baseline_results") or {}
multi_agent_results = evaluation.get("multi_agent_results") or {}

def _generate_baseline_comparison_from_artifacts():
    """Build baseline_comparison.json from baseline_results + multi_agent_results (no retraining)."""
    from evaluation.metrics import MetricsEvaluator
    from utils.json_utils import json_safe
    import json
    comp = MetricsEvaluator().generate_baseline_comparison_json(baseline_results, multi_agent_results, output_path=None)
    for target_dir in [Path(reports_dir), run_dir / "research_outputs", run_dir / "multi_agent" / "research_outputs"]:
        if target_dir.exists() or target_dir == Path(reports_dir):
            target_dir.mkdir(parents=True, exist_ok=True)
            out_path = target_dir / "baseline_comparison.json"
            try:
                with open(out_path, "w") as f:
                    json.dump(comp, f, indent=2, default=json_safe)
                return True
            except Exception:
                pass
    return False

reports_dir = paths.get("reports_dir", run_dir / "reports")
if not hasattr(reports_dir, "exists"):
    reports_dir = Path(reports_dir)

if baseline_comparison and (baseline_comparison.get("deltas") or baseline_comparison.get("improvements")):
    # Render table: metric | baseline | multi_agent | delta
    bl = baseline_comparison.get("baseline") or {}
    ma = baseline_comparison.get("multi_agent") or {}
    deltas = baseline_comparison.get("deltas") or {}
    metric_names = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    rows = []
    for m in metric_names:
        bl_v = (bl.get("metrics") or {}).get(m)
        ma_v = (ma.get("metrics") or {}).get(m)
        d = deltas.get(m)
        if bl_v is not None or ma_v is not None or d is not None:
            rows.append({"metric": m.replace("_", " ").title(), "baseline": bl_v, "multi_agent": ma_v, "delta": d})
    if bl.get("execution_time_seconds") is not None or ma.get("execution_time_seconds") is not None:
        rows.append({"metric": "Execution time (s)", "baseline": bl.get("execution_time_seconds"), "multi_agent": ma.get("execution_time_seconds"), "delta": deltas.get("execution_time_seconds")})
    if rows:
        comp_df = pd.DataFrame(rows)
        st.dataframe(comp_df, use_container_width=True, height=min(320, 80 + len(rows) * 40))
    improvements = baseline_comparison.get("improvements") or []
    if improvements:
        st.caption("**Improvements (multi-agent vs baseline):**")
        for imp in improvements[:10]:
            delta_val = None
            try:
                if "roc_auc" in imp and "+" in imp:
                    delta_val = float(imp.split("+")[-1].strip())
                elif "roc_auc" in imp and "-" in imp:
                    parts = imp.split("-", 1)
                    if len(parts) > 1:
                        delta_val = -float(parts[1].strip())
            except Exception:
                pass
            if delta_val is not None and delta_val > 0:
                st.markdown(f"- <span style='color:#38a169'>**{imp}**</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"- {imp}")
else:
    comparison_path_str = evaluation.get("comparison_path")
    comparison_df = safe_read_csv(Path(comparison_path_str)) if comparison_path_str else None
    if comparison_df is not None and not comparison_df.empty:
        st.dataframe(comparison_df, use_container_width=True, height=300)
    elif baseline_results and multi_agent_results:
        st.info("Comparison summary not found for this run. You can generate it from existing artifacts (no retraining).")
        if st.button("Generate comparison for this run"):
            try:
                if _generate_baseline_comparison_from_artifacts():
                    st.success("Comparison generated. Reload the page to see the table.")
                    st.rerun()
                else:
                    st.warning("Could not write baseline_comparison.json to run directory.")
            except Exception as e:
                st.error(f"Generation failed: {e}")
    elif not baseline_results:
        st.info("**Baseline results are not available** for this run. The comparison requires both the baseline pipeline (single-model run) and the multi-agent pipeline. Run the full pipeline with `python main.py` to generate baseline and multi-agent results, then the evaluation step will save the comparison automatically.")
    else:
        st.info("**Multi-agent results are not available** for this run. Run the full pipeline to generate both baseline and multi-agent outputs and the comparison.")

# Cross-Dataset Validation
st.markdown("---")
st.markdown("### 📂 Cross-Dataset Validation")
cross_data = (artifacts.get("cross_dataset") or {}).get("report") or {}
if cross_data:
    train_ds = cross_data.get("train_dataset") or cross_data.get("source_dataset") or "—"
    target_ds = cross_data.get("target_dataset") or "—"
    eval_split = cross_data.get("eval_split") or cross_data.get("split") or "test"
    test_datasets = cross_data.get("test_datasets") or []
    if isinstance(test_datasets, (list, tuple)) and len(test_datasets) > 0:
        target_ds = test_datasets[0] or target_ds
    st.caption(f"Model trained on {train_ds}, evaluated on {target_ds} ({eval_split}). No retraining.")
    results = cross_data.get("results") or []
    if results:
        cross_df = pd.DataFrame(results)
        st.dataframe(cross_df, use_container_width=True, height=min(200, 80 + len(results) * 35))
    else:
        st.info("No cross-dataset results in report.")
else:
    pred_path = (artifacts.get("cross_dataset") or {}).get("predictions_path")
    cross_df = safe_read_csv(Path(pred_path)) if pred_path else None
    if cross_df is not None and not cross_df.empty:
        st.caption("Cross-dataset validation results (trained on primary dataset, evaluated on target without retraining).")
        st.dataframe(cross_df, use_container_width=True, height=min(200, 80 + len(cross_df) * 35))
    else:
        st.caption("Run with `python main.py --dataset diabetes --cross_dataset heart_disease` to generate cross-dataset validation.")

# Run log
st.markdown("---")
st.markdown("### 📋 Run Log")
run_log_path = paths.get("logs_dir", run_dir / "logs") / "run.log"
if run_log_path.exists():
    try:
        with open(run_log_path, 'r', encoding='utf-8') as f:
            log_content = f.read()
        st.text_area("Pipeline log output", log_content, height=300, disabled=True)
    except Exception as e:
        st.warning(f"Could not read log file: {e}")
else:
    st.info("ℹ️ No run log yet. Logs are written when the pipeline runs (or fails). Run `python main.py` to generate this run's log.")
