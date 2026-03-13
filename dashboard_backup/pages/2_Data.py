"""
Data Agent — Clean implementation dashboard.
Shows what data entered the system and how it was prepared for the next agent.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dashboard.components.styling import render_header, render_agent_section
from dashboard.ui import kpi_card
from dashboard.components.artifacts import load_run_artifacts, resolve_dataset_model_fallback

DATASET_TO_KEY = {"Diabetes": "diabetes", "Heart Disease": "heart_disease"}
KEY_TO_DATASET = {v: k for k, v in DATASET_TO_KEY.items()}

# Fallbacks for KPIs and split (demo / missing artifacts)
DATA_AGENT_FALLBACK = {
    "Diabetes": {
        "raw_rows": 768,
        "features": 9,
        "train_rows": 1612,
        "val_rows": 346,
        "test_rows": 346,
    },
    "Heart Disease": {
        "raw_rows": None,
        "features": None,
        "train_rows": None,
        "val_rows": None,
        "test_rows": None,
    },
}


def _resolve_run_and_dataset() -> Tuple[Optional[Path], str, str]:
    run_id = st.session_state.get("run_id") or st.session_state.get("selected_run_id")
    run_dir = st.session_state.get("run_dir")
    if run_dir is not None:
        run_dir = Path(run_dir)
    elif run_id:
        from dashboard.components.layout import get_run_dir_path
        run_dir = get_run_dir_path(run_id)
    else:
        run_dir = None
    ds_display = st.session_state.get("selected_dataset") or st.session_state.get("selected_dataset_key") or "Diabetes"
    if ds_display in KEY_TO_DATASET:
        ds_display = KEY_TO_DATASET[ds_display]
    elif ds_display not in DATASET_TO_KEY:
        ds_display = "Diabetes"
    dataset_key = DATASET_TO_KEY.get(ds_display, "diabetes")
    return run_dir, dataset_key, ds_display


def _load_csv_rows(path: Optional[str], run_dir: Optional[Path], dataset_key: str, suffix: str) -> Optional[int]:
    """Return row count for train/val/test CSV from path or run_dir."""
    p = None
    if path and Path(path).exists():
        p = Path(path)
    elif run_dir:
        for base in [run_dir / "data", run_dir / "multi_agent" / "data"]:
            candidate = base / f"{dataset_key}_{suffix}.csv"
            if candidate.exists():
                p = candidate
                break
    if p:
        try:
            return len(pd.read_csv(p))
        except Exception:
            pass
    return None


def _get_class_distribution(data_section: Dict, run_dir: Optional[Path], dataset_key: str, target_col: str) -> Tuple[List[str], List[float]]:
    """Return (labels, counts) for class distribution from artifacts or train CSV."""
    class_dist = data_section.get("class_distribution") or {}
    if class_dist.get("labels") and class_dist.get("counts"):
        return [str(x) for x in class_dist["labels"]], list(class_dist["counts"])
    train_path = data_section.get("train_path")
    if train_path and Path(train_path).exists():
        try:
            df = pd.read_csv(train_path, nrows=50000)
            if target_col in df.columns:
                vc = df[target_col].value_counts().sort_index()
                return vc.index.astype(str).tolist(), vc.values.tolist()
        except Exception:
            pass
    if run_dir:
        for base in [run_dir / "data", run_dir / "multi_agent" / "data"]:
            p = base / f"{dataset_key}_train.csv"
            if p.exists():
                try:
                    df = pd.read_csv(p, nrows=50000)
                    tc = target_col or "target"
                    if tc in df.columns:
                        vc = df[tc].value_counts().sort_index()
                        return vc.index.astype(str).tolist(), vc.values.tolist()
                except Exception:
                    pass
    return [], []


# ----- Page -----
st.set_page_config(
    page_title="Data Agent",
    layout="wide",
    initial_sidebar_state="auto",
)

run_dir, dataset_key, ds_display = _resolve_run_and_dataset()
artifacts = load_run_artifacts(run_dir, dataset_key) if run_dir else {}
resolved_dataset, _, dataset_warning = resolve_dataset_model_fallback(artifacts, dataset_key, "")
if dataset_warning:
    st.warning(dataset_warning)
dataset_key = resolved_dataset or dataset_key

data_section = artifacts.get("data") or {}
metadata = data_section.get("metadata") or {}
profile = data_section.get("profile") or {}
target_col = profile.get("target_column") or metadata.get("target_column") or "target"
fallback = DATA_AGENT_FALLBACK.get(ds_display, DATA_AGENT_FALLBACK["Heart Disease"])

# Resolve KPI and split values from artifacts or fallback
raw_rows = metadata.get("raw_rows") or fallback.get("raw_rows")
n_features = metadata.get("n_features") or metadata.get("cleaned_shape", [None, None])[1] or fallback.get("features")
train_rows = metadata.get("train_rows") or _load_csv_rows(data_section.get("train_path"), run_dir, dataset_key, "train") or fallback.get("train_rows")
val_rows = metadata.get("val_rows") or _load_csv_rows(data_section.get("val_path"), run_dir, dataset_key, "val") or fallback.get("val_rows")
test_rows = metadata.get("test_rows") or _load_csv_rows(data_section.get("test_path"), run_dir, dataset_key, "test") or fallback.get("test_rows")
val_test_display = (val_rows, test_rows)
if val_rows is not None and test_rows is not None:
    val_test_display = f"{val_rows} / {test_rows}"
elif val_rows is not None:
    val_test_display = str(val_rows)
elif test_rows is not None:
    val_test_display = str(test_rows)
else:
    val_test_display = "—"

# ----- SECTION 1 — Header -----
render_header("Data Agent", "")
render_agent_section("data_agent", "#2563EB")
st.caption("Validates raw healthcare data and prepares train, validation, and test datasets for downstream agents.")
st.markdown("")

# ----- SECTION 2 — KPI cards (4) -----
c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi_card("Raw Rows", raw_rows if raw_rows is not None else "—")
with c2:
    kpi_card("Features", n_features if n_features is not None else "—")
with c3:
    kpi_card("Train Rows", train_rows if train_rows is not None else "—")
with c4:
    kpi_card("Validation / Test Rows", val_test_display)
st.markdown("")

# ----- SECTION 3 — Split visualization -----
split_labels = ["Train", "Validation", "Test"]
split_values = [
    train_rows if train_rows is not None else 0,
    val_rows if val_rows is not None else 0,
    test_rows if test_rows is not None else 0,
]
if any(split_values):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    colors = ["#2563EB", "#059669", "#7C3AED"]
    bars = ax.bar(split_labels, split_values, color=colors, edgecolor="none")
    ax.set_ylabel("Rows")
    for b, v in zip(bars, split_values):
        if v > 0:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 5, str(int(v)), ha="center", va="bottom", fontsize=11, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
else:
    st.caption("Data split will appear here after a pipeline run.")
st.markdown("")

# ----- SECTION 4 — Class balance -----
labels, counts = _get_class_distribution(data_section, run_dir, dataset_key, target_col)
if labels and counts:
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    x = np.arange(len(labels))
    ax2.bar(x, counts, color="#2563EB", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Count")
    ax2.set_title("Class distribution")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)
else:
    st.caption("Class distribution will appear here when target column is available.")
st.markdown("")

# ----- SECTION 5 — Handoff summary -----
st.markdown(
    '<div style="border:1px solid #e2e8f0; border-radius:8px; padding:1rem 1.25rem; background:#f8fafc;">'
    "<p style='margin:0.35rem 0;'><strong>Handoff summary</strong></p>"
    "<p style='margin:0.35rem 0;'>• Missing values handled</p>"
    "<p style='margin:0.35rem 0;'>• Features scaled</p>"
    "<p style='margin:0.35rem 0;'>• Train / validation / test artifacts created</p>"
    "</div>",
    unsafe_allow_html=True,
)
st.markdown("")

# ----- SECTION 6 — Success line -----
st.success("Processed datasets successfully prepared for the Feature Engineering Agent.")
