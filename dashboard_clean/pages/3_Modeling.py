"""Modeling results page."""

import streamlit as st
import json
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from dashboard.components.styling import render_header, render_agent_section
from dashboard.ui import section_header, kpi_card
from dashboard.components.charts import plot_confusion_matrix, plot_roc_curve, plot_pr_curve
from dashboard.components.artifacts import load_run_artifacts, resolve_dataset_model_fallback

if "run_id" not in st.session_state and "selected_run_id" not in st.session_state:
    st.error("Please select a run from the sidebar.")
    st.stop()

run_id = st.session_state.get("run_id") or st.session_state.get("selected_run_id")
from dashboard.components.layout import get_run_dir_path, resolve_paths
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

paths = {k: Path(v) for k, v in (artifacts.get("files") or {}).get("paths", {}).items()}
if not paths:
    paths = resolve_paths(run_dir)

render_header("Modeling Results", "")
render_agent_section("model_agent", "#238b45")
st.caption("Each model has a distinct color in the charts below (Logistic Regression = blue, Random Forest = green, XGBoost = orange).")
models_section = artifacts.get("models") or {}
model_metrics = models_section.get("metrics") or {}
selection_reasoning = model_metrics.get("selection_reasoning")
if selection_reasoning:
    st.info(f"**Selection reasoning:** {selection_reasoning}")


# Same artifact lookup as confusion matrix: use resolve_paths (reports_dir, figures_dir) first, then plots/results/evaluation
def _roc_candidate_dirs(run_dir: Path, paths: dict) -> list:
    """Same folder order as metrics/figures: figures_dir, reports_dir, then plots/results/evaluation and multi_agent."""
    run_dir = Path(run_dir)
    return [
        paths.get("figures_dir", run_dir / "figures"),
        paths.get("reports_dir", run_dir / "reports"),
        run_dir / "plots",
        run_dir / "results",
        run_dir / "evaluation",
        run_dir / "multi_agent" / "figures",
        run_dir / "multi_agent" / "reports",
        run_dir / "multi_agent" / "research_outputs",
        run_dir / "multi_agent" / "plots",
        run_dir / "multi_agent" / "results",
        run_dir / "multi_agent" / "evaluation",
    ]


def _normalize(s: str) -> str:
    """Normalize for filename matching: lowercase, replace hyphens with underscores."""
    return (s or "").lower().replace("-", "_").replace(" ", "_")


def _find_roc_artifact(run_dir: Path, model: str, dataset: str, paths: dict):
    """Return (path or None, list of dirs searched). Same dirs as CM; accept roc_curve.png, roc.png, roc_auc.png, roc_curve_<model>.png, <model>_roc.png, <model>_roc_curve.png, roc_curve_<dataset>_<model>.png (case-insensitive, hyphen/underscore)."""
    import re
    dirs = _roc_candidate_dirs(run_dir, paths)
    m = _normalize(model)
    d = _normalize(dataset)
    model_safe = re.escape(m)
    dataset_safe = re.escape(d)
    # Model-specific first, then generic. Patterns: exact names and model/dataset variants
    patterns = [
        (rf"roc_curve_{dataset_safe}_{model_safe}\.(png|html)$", "model_dataset"),
        (rf"roc_curve_{model_safe}\.(png|html)$", "model_specific"),
        (rf"{model_safe}_roc_curve\.(png|html)$", "model_specific"),
        (rf"{model_safe}_roc\.(png|html)$", "model_specific"),
        (r"roc_curve\.(png|html)$", "generic"),
        (r"roc\.(png|html)$", "generic"),
        (r"roc_auc\.(png|html)$", "generic"),
        (r"roc_curve.*\.(png|html)$", "generic"),
        (r"roc_auc.*\.(png|html)$", "generic"),
        (r"roc.*\.(png|html)$", "generic"),
    ]
    generic_match = None
    for cand_dir in dirs:
        if not cand_dir.exists() or not cand_dir.is_dir():
            continue
        for f in sorted(cand_dir.iterdir(), key=lambda p: p.name):
            if not f.is_file():
                continue
            name_lower = _normalize(f.stem) + f.suffix.lower()
            ext = f.suffix.lower()
            if ext not in (".png", ".html"):
                continue
            for pat, kind in patterns:
                if re.search(pat, name_lower):
                    if kind != "generic":
                        return f, dirs
                    if generic_match is None:
                        generic_match = f
    return (generic_match, dirs) if generic_match is not None else (None, dirs)


def _find_predictions_file(run_dir: Path, paths: dict, dataset: str = "") -> Path:
    """Search run for predictions CSV (same style as artifact lookup). Return first path found or None."""
    run_dir = Path(run_dir)
    reports = paths.get("reports_dir", run_dir / "reports")
    models = paths.get("models_dir", run_dir / "models")
    candidates = [
        models / "validation_predictions.csv",
        reports / "validation_predictions.csv",
        reports / "predictions.csv",
    ]
    if dataset:
        candidates.insert(1, reports / f"validation_predictions_{dataset}.csv")
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    # Glob patterns in research_outputs, results, evaluation, multi_agent
    for base in [run_dir / "research_outputs", run_dir / "results", run_dir / "evaluation",
                 run_dir / "multi_agent" / "research_outputs", run_dir / "multi_agent" / "reports",
                 run_dir / "multi_agent" / "results", run_dir / "multi_agent" / "evaluation",
                 paths.get("reports_dir"), paths.get("models_dir")]:
        if base is None or not base.exists():
            continue
        for f in base.glob("*predictions*.csv"):
            if f.is_file():
                return f
    for base in [run_dir / "multi_agent" / "models", run_dir / "models"]:
        if base.exists():
            for f in base.glob("*predictions*.csv"):
                if f.is_file():
                    return f
    return None


def _y_true_y_proba_from_df(pred_df: pd.DataFrame, model: str) -> tuple:
    """Return (y_true array, y_scores array or None). Uses y_true from true_label/y_true/label/target; scores from y_proba, proba, p1, logits, score, decision_score, or {model}_proba. If only hard labels (y_pred) exist, returns (y_true, None)."""
    def _norm(c: str) -> str:
        return (c or "").lower().strip().replace(" ", "_").replace("-", "_")
    cols_lower = {_norm(c): c for c in pred_df.columns}
    y_true_col = None
    for cand in ("y_true", "true", "label", "target", "true_label"):
        if cand in cols_lower:
            y_true_col = cols_lower[cand]
            break
    if y_true_col is None:
        return None, None
    y_true = pred_df[y_true_col].values

    # Score/proba column: prefer probability-like, then any continuous score (logits/decision scores work for ROC)
    score_col = None
    for cand in ("y_proba", "proba", "probability", "p1", "pred_proba", "positive_probability"):
        if cand in cols_lower:
            score_col = cols_lower[cand]
            break
    if score_col is None:
        for cand in (f"{model}_proba", f"{model}_score", "logits", "score", "decision_score", "scores"):
            if cand in pred_df.columns:
                score_col = cand
                break
            alt = cand.replace("_", "-")
            if alt in pred_df.columns:
                score_col = alt
                break
            if cand in cols_lower and cols_lower[cand] in pred_df.columns:
                score_col = cols_lower[cand]
                break
    if score_col is None or score_col not in pred_df.columns:
        return y_true, None
    y_scores = np.asarray(pred_df[score_col].values, dtype=float)
    return y_true, y_scores


# Load metrics from unified artifacts (never error; show message if missing)
metrics_data = (artifacts.get("models") or {}).get("metrics") or {}
if not metrics_data:
    st.info("Model metrics not found for this run. Run the pipeline to generate model metrics.")

# Model leaderboard
section_header("Model Leaderboard", caption=None)

all_models = list(metrics_data.keys())
leaderboard_data = []

for m in all_models:
    m_metrics = metrics_data[m]
    leaderboard_data.append({
        'Model': m.replace('_', ' ').title(),
        'Accuracy': m_metrics.get('accuracy', 0),
        'ROC-AUC': m_metrics.get('roc_auc', 0),
        'F1-Score': m_metrics.get('f1_score', 0),
        'Precision': m_metrics.get('precision', 0),
        'Recall': m_metrics.get('recall', 0)
    })

leaderboard_df = pd.DataFrame(leaderboard_data)
if not leaderboard_df.empty and "ROC-AUC" in leaderboard_df.columns:
    leaderboard_df = leaderboard_df.sort_values("ROC-AUC", ascending=False)
st.dataframe(leaderboard_df, use_container_width=True)

st.markdown("---")

# Selected model metrics (only show if this model is in metrics_data)
section_header(f"Metrics for {model.replace('_', ' ').title()}", caption=None)

model_metrics = metrics_data.get(model, {})
if not model_metrics:
    st.warning(f"No metrics for **{model}**. Choose a model from the leaderboard above (e.g. {', '.join(metrics_data.keys())}).")
else:
    if _minority_positive:
        st.caption("**Recall**, **F1**, and **PR-AUC** are especially important when the positive class is the minority.")
    cols_metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    if model_metrics.get("average_precision") is not None:
        cols_metrics.append("PR-AUC")
    ncol = len(cols_metrics)
    cols = st.columns(ncol)
    for i, name in enumerate(cols_metrics):
        key = name.lower().replace("-", "_").replace(" ", "_")
        if key == "pr_auc":
            key = "average_precision"
        with cols[i]:
            kpi_card(name, model_metrics.get(key, 0))

# Confusion matrix
st.markdown("---")
section_header("Confusion Matrix", caption=None)
if model_metrics and 'confusion_matrix' in model_metrics:
    cm = np.array(model_metrics['confusion_matrix'])
    fig = plot_confusion_matrix(cm, model_name=model)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.caption("Confusion matrix not available for this model.")

# ROC curve: same artifact lookup as confusion matrix (figures_dir, reports_dir); else compute from y_true/y_proba
st.markdown("---")
section_header("ROC Curve", caption=None)

roc_path, roc_dirs_searched = _find_roc_artifact(run_dir, model, dataset, paths)
# Collect all candidate prediction files (validation + cross_dataset) from artifacts and search
predictions_candidates = []
if (artifacts.get("models") or {}).get("validation_predictions_path"):
    predictions_candidates.append(Path((artifacts["models"]["validation_predictions_path"])))
if (artifacts.get("cross_dataset") or {}).get("predictions_path"):
    predictions_candidates.append(Path(artifacts["cross_dataset"]["predictions_path"]))
if not predictions_candidates:
    p = _find_predictions_file(run_dir, paths, dataset)
    if p:
        predictions_candidates.append(Path(p) if not isinstance(p, Path) else p)
predictions_path = predictions_candidates[0] if predictions_candidates else None
no_proba_msg_shown = False

# Debug: list files in plots_dir and figures_dir/reports_dir (first 40 each) to see naming
plots_dirs = [run_dir / "plots", run_dir / "multi_agent" / "plots"]
files_in_plots_dir = []
for pd in plots_dirs:
    if pd.exists() and pd.is_dir():
        files_in_plots_dir.extend([f.name for f in pd.iterdir() if f.is_file()])
files_in_plots_dir = sorted(set(files_in_plots_dir))[:40]
plots_like_dir = paths.get("figures_dir") or paths.get("reports_dir") or run_dir / "figures"
files_in_figures_reports = []
if plots_like_dir and Path(plots_like_dir).exists():
    files_in_figures_reports = sorted([f.name for f in Path(plots_like_dir).iterdir() if f.is_file()])[:40]

with st.expander("Debug: ROC resolution", expanded=False):
    st.text(f"run_id: {run_id}")
    st.text(f"model: {model}")
    st.text(f"run_dir: {run_dir}")
    st.text("Directories searched for ROC artifact:")
    for d in roc_dirs_searched:
        st.text(f"  - {d} (exists={d.exists()})")
    st.text(f"Files in plots_dir (run_dir/plots, multi_agent/plots; first 40): {files_in_plots_dir}")
    st.text(f"Files in figures_dir/reports_dir (first 40): {files_in_figures_reports}")
    st.text(f"roc file picked: {roc_path}")
    st.text(f"predictions candidates: {[str(p) for p in predictions_candidates]}")

roc_rendered = False
if roc_path is not None and roc_path.exists():
    try:
        if roc_path.suffix.lower() == ".html":
            html_content = roc_path.read_text(encoding="utf-8", errors="replace")
            st.components.v1.html(html_content, height=500, scrolling=False)
            roc_rendered = True
        else:
            st.image(str(roc_path), use_container_width=True)
            roc_rendered = True
    except Exception as e:
        st.caption(f"Could not load ROC file: {e}")

def _load_y_true_y_proba_from_model_val(run_dir: Path, paths: dict, dataset: str, model: str) -> tuple:
    """Load model + validation CSV, run predict_proba; return (y_true, y_proba) or (None, None). Tries val.csv and {dataset}_val.csv."""
    run_dir = Path(run_dir)
    models_dir = paths.get("models_dir") or run_dir / "models"
    data_dir = paths.get("data_dir") or run_dir / "data"
    if not hasattr(models_dir, "exists"):
        models_dir = Path(models_dir)
    if not hasattr(data_dir, "exists"):
        data_dir = Path(data_dir)
    model_path = models_dir / f"{model}.pkl"
    if not model_path.exists():
        model_path = run_dir / "multi_agent" / "models" / f"{model}.pkl"
    if not model_path.exists():
        model_path = run_dir / "models" / f"{model}.pkl"
    val_candidates = [
        data_dir / f"{dataset}_val.csv",
        data_dir / "val.csv",
        run_dir / "multi_agent" / "data" / f"{dataset}_val.csv",
        run_dir / "multi_agent" / "data" / "val.csv",
        run_dir / "data" / f"{dataset}_val.csv",
        run_dir / "data" / "val.csv",
    ]
    val_path = None
    for p in val_candidates:
        if p.exists() and p.is_file():
            val_path = p
            break
    if not val_path or not model_path.exists():
        return None, None
    reports_dir = paths.get("reports_dir") or run_dir / "reports"
    metadata_candidates = [
        reports_dir / f"data_metadata_{dataset}.json",
        run_dir / "multi_agent" / "research_outputs" / f"data_metadata_{dataset}.json",
        run_dir / "reports" / f"data_metadata_{dataset}.json",
    ]
    target_col = "target"
    for mp in metadata_candidates:
        if mp.exists():
            try:
                with open(mp, "r") as f:
                    meta = json.load(f)
                target_col = meta.get("target_column", "target")
            except Exception:
                pass
            break
    try:
        val_df = pd.read_csv(val_path)
        if target_col not in val_df.columns:
            return None, None
        X_val = val_df.drop(columns=[target_col])
        y_val = val_df[target_col]
        with open(model_path, "rb") as f:
            loaded_model = pickle.load(f)
        if not hasattr(loaded_model, "predict_proba"):
            return None, None
        y_true = np.asarray(y_val.values)
        y_proba = np.asarray(loaded_model.predict_proba(X_val)[:, 1], dtype=float)
        return y_true, y_proba
    except Exception:
        return None, None


if not roc_rendered:
    y_true, y_proba = None, None
    # 1) Try each predictions CSV (validation_predictions, cross_dataset_predictions)
    for pred_path in predictions_candidates:
        if pred_path is None or not Path(pred_path).exists():
            continue
        try:
            pred_df = pd.read_csv(pred_path)
            y_true, y_proba = _y_true_y_proba_from_df(pred_df, model)
            if y_true is not None and y_proba is not None and len(y_true) == len(y_proba):
                break
        except Exception:
            continue
    # 2) If no scores in any CSV, compute from model + validation data
    if y_true is not None and y_proba is None:
        y_true, y_proba = _load_y_true_y_proba_from_model_val(run_dir, paths, dataset, model)
    elif y_true is None or y_proba is None:
        y_true, y_proba = _load_y_true_y_proba_from_model_val(run_dir, paths, dataset, model)

    if y_true is not None and y_proba is not None and len(y_true) == len(y_proba):
        try:
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(np.asarray(y_true), np.asarray(y_proba))
            roc_auc_val = auc(fpr, tpr)
            fig = plot_roc_curve(np.asarray(y_true), np.asarray(y_proba), model)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"**AUC = {roc_auc_val:.4f}** (computed from validation predictions)")
            roc_rendered = True
        except Exception as e:
            st.caption(f"Could not plot ROC: {e}")
    elif y_true is not None and y_proba is None:
        st.warning("Only hard labels (y_pred) found; ROC requires probabilities or scores. Re-run pipeline with probability outputs or select a run that has ROC artifacts.")
        no_proba_msg_shown = True

if not roc_rendered and not no_proba_msg_shown:
    st.info("ROC artifact not found for this run/model. Re-run pipeline or select a successful run.")

# Precision-Recall curve (on the fly from same data as ROC; important for imbalanced data)
st.markdown("---")
section_header("Precision-Recall Curve", caption=None)
if _minority_positive:
    st.caption("PR curve is more informative than ROC when the positive class is the minority.")
pr_y_true, pr_y_proba = None, None
for pred_path in (predictions_candidates or []):
    if pred_path is None or not Path(pred_path).exists():
        continue
    try:
        pred_df = pd.read_csv(pred_path)
        pr_y_true, pr_y_proba = _y_true_y_proba_from_df(pred_df, model)
        if pr_y_true is not None and pr_y_proba is not None:
            break
    except Exception:
        continue
if pr_y_true is None or pr_y_proba is None:
    pr_y_true, pr_y_proba = _load_y_true_y_proba_from_model_val(run_dir, paths, dataset, model)
if pr_y_true is not None and pr_y_proba is not None and len(pr_y_true) == len(pr_y_proba):
    try:
        fig_pr = plot_pr_curve(np.asarray(pr_y_true), np.asarray(pr_y_proba), model)
        st.plotly_chart(fig_pr, use_container_width=True)
        from sklearn.metrics import average_precision_score
        ap = average_precision_score(np.asarray(pr_y_true), np.asarray(pr_y_proba))
        st.caption(f"**Average precision (PR-AUC) = {ap:.4f}**")
    except Exception as e:
        st.caption(f"Could not plot PR curve: {e}")
else:
    st.caption("Precision-Recall curve requires validation predictions or model + validation data.")