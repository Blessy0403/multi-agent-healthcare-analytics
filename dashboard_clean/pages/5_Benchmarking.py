"""Benchmarking comparison page.

Compares baseline_results and multi_agent_results produced by:
  baseline_results = run_baseline_pipeline()
  multi_agent_results = run_multi_agent_pipeline()
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
from dashboard.components.styling import render_header, render_agent_section
from dashboard.ui import section_header, kpi_card
from dashboard.components.charts import plot_metrics_comparison

if "run_id" not in st.session_state and "selected_run_id" not in st.session_state:
    st.error("Please select a run from the sidebar.")
    st.stop()

run_id = st.session_state.get("selected_run_id") or st.session_state.get("run_id")
from dashboard.components.layout import get_run_dir_path, get_reports_dir, resolve_paths
dataset = st.session_state.get("selected_dataset_key") or st.session_state.get("dataset", "heart_disease")

render_header("Baseline vs Multi-Agent Comparison", "")
render_agent_section("evaluation_agent", "#b45309")
st.caption("Comparison of **baseline** vs **multi-agent** pipeline result dictionaries (run_baseline_pipeline vs run_multi_agent_pipeline).")

run_dir = get_run_dir_path(run_id)
paths = resolve_paths(run_dir)
reports_dir = paths["reports_dir"]

comparison_df = None
source_note = None

# Primary: build comparison table ONLY from selected_metrics (do NOT use all_models).
# One row per metric (accuracy, precision, recall, f1_score, roc_auc); no per-model rows.
baseline_path = reports_dir / 'baseline_results.json'
multi_agent_path = reports_dir / 'multi_agent_results.json'
if baseline_path.exists() and multi_agent_path.exists():
    try:
        with open(baseline_path, 'r') as f:
            baseline_results = json.load(f)
        with open(multi_agent_path, 'r') as f:
            multi_agent_results = json.load(f)
        bl_selected = baseline_results.get('selected_metrics') or {}
        ma_selected = (multi_agent_results.get('models') or {}).get('selected_metrics') or {}
        bl_sel = baseline_results.get('selected_model') or baseline_results.get('best_model_name')
        ma_sel = (multi_agent_results.get('models') or {}).get('selected_model') or (multi_agent_results.get('models') or {}).get('best_model_name')
        metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        rows = []
        for metric in metric_names:
            ma_v = ma_selected.get(metric)
            bl_v = bl_selected.get(metric)
            if ma_v is not None or bl_v is not None:
                try:
                    ma_f = float(ma_v) if ma_v is not None else 0.0
                    bl_f = float(bl_v) if bl_v is not None else 0.0
                    rows.append({
                        'metric': metric,
                        'baseline_selected_model': bl_sel,
                        'multi_agent_selected_model': ma_sel,
                        'baseline': bl_f,
                        'multi_agent': ma_f,
                        'difference': ma_f - bl_f,
                    })
                except (TypeError, ValueError):
                    pass
        if rows:
            comparison_df = pd.DataFrame(rows)
            ma_time = (multi_agent_results.get('collaboration_metrics') or {}).get('total_execution_time') or multi_agent_results.get('execution_time') or 0
            bl_time = baseline_results.get('execution_time', 0)
            try:
                ma_time, bl_time = float(ma_time), float(bl_time)
            except (TypeError, ValueError):
                ma_time, bl_time = 0, 0
            comparison_df = pd.concat([
                comparison_df,
                pd.DataFrame([{
                    'metric': 'execution_time_seconds',
                    'baseline_selected_model': bl_sel,
                    'multi_agent_selected_model': ma_sel,
                    'baseline': bl_time,
                    'multi_agent': ma_time,
                    'difference': ma_time - bl_time,
                }])
            ], ignore_index=True)
            source_note = "Comparison of selected model only: baseline_results['selected_metrics'] vs multi_agent_results['selected_metrics']."
    except Exception:
        pass

if comparison_df is None:
    # Fallback: load comparison_report.csv (research_outputs or reports)
    comparison_path = reports_dir / 'comparison_report.csv'
    if not comparison_path.exists():
        comparison_path = reports_dir / f'comparison_report_{dataset}.csv'
    if comparison_path.exists():
        comparison_df = pd.read_csv(comparison_path)
        # If CSV has "model" column, keep only selected rows (one row per metric) so we don't show per-model rows
        if comparison_df is not None and 'model' in comparison_df.columns and 'metric' in comparison_df.columns:
            if (comparison_df['model'] == 'selected').any():
                comparison_df = comparison_df[comparison_df['model'] == 'selected'].drop(columns=['model'], errors='ignore')
        if comparison_df is not None and 'baseline_selected_model' not in comparison_df.columns:
            meta_path = paths['reports_dir'] / 'run_metadata.json' if (paths['reports_dir'] / 'run_metadata.json').exists() else (run_dir / 'run_metadata.json' if (run_dir / 'run_metadata.json').exists() else None)
            if meta_path is not None:
                try:
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    comparison_df['baseline_selected_model'] = meta.get('best_model_baseline') or meta.get('selected_model_baseline')
                    comparison_df['multi_agent_selected_model'] = meta.get('best_model_multi_agent') or meta.get('selected_model_multi_agent')
                except Exception:
                    pass
        source_note = None

if comparison_df is None:
    # Fallback 1: build from run_metadata.json (one row per metric; roc_auc only)
    meta_path = run_dir / 'run_metadata.json'
    if not meta_path.exists():
        meta_path = paths['reports_dir'] / 'run_metadata.json'
    if meta_path.exists():
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            ma_auc = meta.get('best_model_roc_auc_ma')
            bl_auc = meta.get('best_model_roc_auc_bl')
            if ma_auc is not None and bl_auc is not None:
                comparison_df = pd.DataFrame([{
                    "metric": "roc_auc",
                    "baseline_selected_model": meta.get("best_model_baseline"),
                    "multi_agent_selected_model": meta.get("best_model_multi_agent"),
                    "multi_agent": ma_auc,
                    "baseline": bl_auc,
                    "difference": ma_auc - bl_auc,
                }])
                source_note = "Comparison from run metadata (roc_auc only)."
        except Exception:
            pass

    # Fallback 2: from model_metrics.json show only best model's metrics (one row per metric), not all_models
    if comparison_df is None:
        for metrics_path in [paths['models_dir'] / 'model_metrics.json', paths['reports_dir'] / 'model_metrics.json',
                            paths['models_dir'] / f'model_metrics_{dataset}.json', paths['reports_dir'] / f'model_metrics_{dataset}.json']:
            if metrics_path.exists():
                try:
                    with open(metrics_path, 'r') as f:
                        metrics_data = json.load(f)
                    if isinstance(metrics_data, dict) and metrics_data:
                        best_name = max(metrics_data.keys(), key=lambda m: metrics_data[m].get("roc_auc", 0) if isinstance(metrics_data.get(m), dict) else 0)
                        m = metrics_data.get(best_name)
                        if isinstance(m, dict):
                            metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
                            rows = []
                            for metric in metric_names:
                                v = m.get(metric)
                                if v is not None:
                                    try:
                                        rows.append({
                                            "metric": metric,
                                            "baseline_selected_model": None,
                                            "multi_agent_selected_model": best_name,
                                            "multi_agent": float(v),
                                            "baseline": None,
                                            "difference": None,
                                        })
                                    except (TypeError, ValueError):
                                        pass
                            if rows:
                                comparison_df = pd.DataFrame(rows)
                                source_note = "Multi-agent selected model metrics only (no baseline results). Run full pipeline for full comparison."
                            break
                except Exception:
                    continue

if comparison_df is None or comparison_df.empty:
    st.info("Comparison report not available for this run. This may be generated during evaluation.")
    st.markdown("""
    **To see the comparison:**
    1. Run the full pipeline: `python main.py`
    2. Ensure the run completes successfully (Overview shows ✅ Success).
    3. The Evaluation Agent will then save `reports/comparison_report.csv` and you can select that run here.
    """)
    placeholder_df = pd.DataFrame([
        {"metric": "accuracy", "baseline_selected_model": "—", "multi_agent_selected_model": "—", "baseline": "—", "multi_agent": "—", "difference": "—"},
        {"metric": "roc_auc", "baseline_selected_model": "—", "multi_agent_selected_model": "—", "baseline": "—", "multi_agent": "—", "difference": "—"},
    ])
    st.caption("Preview: one row per metric, selected model only (run the pipeline to fill with real metrics).")
    st.dataframe(placeholder_df, use_container_width=True)
    st.markdown("---")
    section_header("Summary Statistics", caption=None)
    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Multi-Agent Avg ROC-AUC", "—", max_value_len=8)
    with c2:
        kpi_card("Baseline Avg ROC-AUC", "—", max_value_len=8)
    with c3:
        kpi_card("Difference", "—", max_value_len=8)
    st.markdown("---")
    section_header("Detailed Comparison by Metric", caption=None)
    st.caption("A comparison chart will appear here after a successful pipeline run.")
    st.stop()

if source_note:
    st.caption(source_note)

st.dataframe(comparison_df, use_container_width=True)

# Summary statistics (safe column access)
st.markdown("---")
section_header("Summary Statistics", caption=None)

has_metric = 'metric' in comparison_df.columns
has_ma = 'multi_agent' in comparison_df.columns
has_bl = 'baseline' in comparison_df.columns and comparison_df['baseline'].notna().any()
if has_metric and has_ma:
    ma_metrics = comparison_df[comparison_df['metric'] == 'roc_auc'] if has_metric else comparison_df
    if len(ma_metrics) > 0:
        col1, col2, col3 = st.columns(3)
        with col1:
            kpi_card("Multi-Agent Avg ROC-AUC", ma_metrics['multi_agent'].mean())
        with col2:
            if has_bl:
                kpi_card("Baseline Avg ROC-AUC", ma_metrics['baseline'].mean())
            else:
                kpi_card("Baseline Avg ROC-AUC", "—", max_value_len=8)
        with col3:
            if has_bl:
                diff = ma_metrics['multi_agent'].mean() - ma_metrics['baseline'].mean()
                kpi_card("Difference", f"{diff:+.3f}")
            else:
                kpi_card("Difference", "—", max_value_len=8)

# Detailed comparison by metric
st.markdown("---")
section_header("Detailed Comparison by Metric", caption=None)

model_col = "model" if "model" in comparison_df.columns else comparison_df.columns[0]
numeric_cols = [c for c in comparison_df.columns if c != model_col and pd.api.types.is_numeric_dtype(comparison_df[c]) and comparison_df[c].notna().any()]
if model_col in comparison_df.columns and numeric_cols:
    metric_options = comparison_df['metric'].dropna().unique().tolist() if 'metric' in comparison_df.columns else ["roc_auc"]
    if metric_options:
        metric_type = st.selectbox("Select Metric", metric_options)
        metric_df = comparison_df[comparison_df['metric'] == metric_type] if 'metric' in comparison_df.columns else comparison_df
        if len(metric_df) > 0:
            try:
                fig = plot_metrics_comparison(metric_df)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not plot comparison: {e}")
else:
    st.caption("Comparison report does not have the expected columns (model, numeric metrics).")
