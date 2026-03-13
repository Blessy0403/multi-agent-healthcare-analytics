import json
from pathlib import Path

import pandas as pd
import streamlit as st


# ---------- helpers ----------
def safe_load_json(path: Path | None) -> dict | None:
    """Load JSON from path; return None on missing or error."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def extract_in_dataset_roc_auc(run_dir: Path) -> float | None:
    """
    Extract in-dataset (primary/validation) ROC-AUC for the selected/best model.
    Tries: research_outputs/multi_agent_results.json, baseline_results.json, then reports/*metrics*.json.
    """
    run_dir = Path(run_dir)
    # (i) multi_agent_results.json -> selected_metrics.roc_auc or selected model's roc_auc
    for base in (run_dir / "research_outputs", run_dir / "multi_agent" / "research_outputs"):
        p = base / "multi_agent_results.json"
        data = safe_load_json(p)
        if data:
            models = data.get("models") or {}
            sel_metrics = models.get("selected_metrics")
            if isinstance(sel_metrics, dict):
                r = sel_metrics.get("roc_auc")
                if r is not None:
                    try:
                        return float(r)
                    except (TypeError, ValueError):
                        pass
            sel_name = models.get("selected_model") or models.get("best_model_name")
            if sel_name and isinstance(models.get("models"), dict):
                m = models["models"].get(sel_name)
                if isinstance(m, dict) and m.get("roc_auc") is not None:
                    try:
                        return float(m["roc_auc"])
                    except (TypeError, ValueError):
                        pass
    # (ii) baseline_results.json
    for base in (run_dir / "research_outputs", run_dir / "multi_agent" / "research_outputs", run_dir / "reports"):
        p = base / "baseline_results.json"
        data = safe_load_json(p)
        if data:
            sel_metrics = data.get("selected_metrics")
            if isinstance(sel_metrics, dict) and sel_metrics.get("roc_auc") is not None:
                try:
                    return float(sel_metrics["roc_auc"])
                except (TypeError, ValueError):
                    pass
    # (iii) reports/*metrics*.json or models/model_metrics.json
    for base in (
        run_dir / "reports",
        run_dir / "models",
        run_dir / "research_outputs",
        run_dir / "multi_agent" / "reports",
        run_dir / "multi_agent" / "research_outputs",
    ):
        if not base.exists():
            continue
        for f in base.glob("*metrics*.json"):
            data = safe_load_json(f)
            if not isinstance(data, dict):
                continue
            for val in data.values():
                if isinstance(val, dict) and val.get("roc_auc") is not None:
                    try:
                        return float(val["roc_auc"])
                    except (TypeError, ValueError):
                        pass
    return None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _runs_dir() -> Path:
    return _project_root() / "outputs" / "runs"


def _newest_run_id() -> str | None:
    rd = _runs_dir()
    if not rd.exists():
        return None
    runs = [p for p in rd.iterdir() if p.is_dir() and p.name.startswith("run_")]
    if not runs:
        return None
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0].name


def _run_dir(run_id: str) -> Path:
    return _runs_dir() / run_id


def _cross_dataset_base_dirs(run_dir: Path) -> list[Path]:
    """All locations where cross-dataset artifacts may live."""
    return [
        run_dir / "research_outputs",
        run_dir / "multi_agent" / "research_outputs",
        run_dir / "reports",
        run_dir / "multi_agent" / "reports",
    ]


def _first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists() and p.is_file():
            return p
    return None


def _find_cross_dataset_files(run_dir: Path) -> tuple[Path | None, Path | None, Path | None, Path | None]:
    """Return (report_json, metrics_json, preds_csv, summary_txt) from any of the four base dirs."""
    bases = _cross_dataset_base_dirs(run_dir)
    report_json = _first_existing([b / "cross_dataset_report.json" for b in bases])
    metrics_json = _first_existing([b / "cross_dataset_metrics.json" for b in bases])
    preds_csv = _first_existing([b / "cross_dataset_predictions.csv" for b in bases])
    summary_txt = _first_existing([b / "cross_dataset_summary.txt" for b in bases])
    return report_json, metrics_json, preds_csv, summary_txt


def _has_any_cross_dataset_output(report_json: Path | None, metrics_json: Path | None, preds_csv: Path | None, summary_txt: Path | None) -> bool:
    return bool(report_json or metrics_json or preds_csv or summary_txt)


def _newest_run_with_cross_dataset_report() -> str | None:
    """Return the newest run_id that contains cross_dataset_report.json in any of the four locations."""
    rd = _runs_dir()
    if not rd.exists():
        return None
    run_dirs = [p for p in rd.iterdir() if p.is_dir() and p.name.startswith("run_")]
    if not run_dirs:
        return None
    run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for d in run_dirs:
        report_json, _, _, _ = _find_cross_dataset_files(d)
        if report_json is not None:
            return d.name
    return None


def _read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _str_or_dash(val) -> str:
    if val is None or (isinstance(val, float) and (val != val)):  # NaN
        return "—"
    return str(val)


def _extract_cross_dataset_fields(report: dict, metrics_data: dict, preds: "pd.DataFrame") -> tuple:
    """Extract display fields from report (fallback) and metrics (primary). Handles naming variants."""
    # Scalars: try report first, then metrics; support train_dataset/source_dataset, selected_model/model_used
    def _get(*keys_and_sources):
        for k, d in keys_and_sources:
            if isinstance(d, dict):
                v = d.get(k)
                if v is not None and v != "":
                    return v
        return None

    source_dataset = _get(
        ("source_dataset", report), ("train_dataset", report), ("source_dataset", metrics_data), ("train_dataset", metrics_data),
    )
    target_dataset = _get(("target_dataset", report), ("target_dataset", metrics_data))
    eval_split = _get(("eval_split", report), ("eval_split", metrics_data))
    model_used = _get(
        ("model_used", report), ("selected_model", report), ("model_used", metrics_data), ("selected_model", metrics_data),
    )
    status = _get(("status", report), ("status", metrics_data))
    n_samples = _get(("n_samples", report), ("n_samples", metrics_data))
    if n_samples is None and preds is not None and not preds.empty:
        try:
            n_samples = len(preds)
        except Exception:
            pass

    # Numeric metrics: prefer metrics.canonical_metrics, then report.metrics, then first model in metrics.metrics
    m = None
    if isinstance(metrics_data, dict):
        m = metrics_data.get("canonical_metrics")
        if isinstance(m, dict):
            pass
        else:
            m = report.get("metrics") if isinstance(report.get("metrics"), dict) else None
        if not m and isinstance(metrics_data.get("metrics"), dict):
            per_model = metrics_data["metrics"]
            if per_model:
                first_val = next(iter(per_model.values()), None)
                if isinstance(first_val, dict):
                    m = first_val
    if not m and isinstance(report, dict):
        m = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
    if not m:
        m = {}

    roc_auc = m.get("roc_auc")
    acc = m.get("accuracy")
    prec = m.get("precision")
    rec = m.get("recall")
    f1 = m.get("f1") or m.get("f1_score")

    return (
        _str_or_dash(source_dataset),
        _str_or_dash(target_dataset),
        _str_or_dash(eval_split),
        _str_or_dash(model_used),
        _str_or_dash(status),
        _str_or_dash(n_samples),
        roc_auc,
        acc,
        prec,
        rec,
        f1,
    )


def _read_csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p)


def _fmt(x, nd=4):
    if x is None:
        return "—"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


# ---------- page ----------
st.set_page_config(page_title="Cross-Dataset Generalization", page_icon="🔁", layout="wide")

st.title("🔁 Cross-Dataset Generalization")
st.caption("Train on one dataset, evaluate on a different dataset to test transfer/generalization.")

# Use selected run from sidebar (single source of truth: selected_run_id)
run_id = st.session_state.get("selected_run_id") or st.session_state.get("run_id") or _newest_run_id()

if not run_id:
    st.error("No runs found in outputs/runs. Run the pipeline first (python3 main.py).")
    st.stop()

# Prefer run_dir from session state; else build from run_id
_run_dir_from_app = st.session_state.get("run_dir")
run_dir = Path(_run_dir_from_app) if _run_dir_from_app else _run_dir(run_id)
run_dir = Path(run_dir)

# Use unified artifact loader for cross-dataset section
from dashboard.components.artifacts import load_run_artifacts, safe_read_csv
artifacts = load_run_artifacts(run_dir)
report = (artifacts.get("cross_dataset") or {}).get("report") or {}
metrics = (artifacts.get("cross_dataset") or {}).get("metrics") or {}
preds_path = (artifacts.get("cross_dataset") or {}).get("predictions_path")
preds = safe_read_csv(Path(preds_path)) if preds_path else pd.DataFrame()
if preds is None:
    preds = pd.DataFrame()
found = artifacts.get("health", {}).get("cross_dataset", False) or bool(report or metrics or (preds is not None and not preds.empty))
report_json = Path((artifacts.get("cross_dataset") or {}).get("report_path") or "") if (artifacts.get("cross_dataset") or {}).get("report_path") else None
metrics_json = Path((artifacts.get("cross_dataset") or {}).get("metrics_path") or "") if (artifacts.get("cross_dataset") or {}).get("metrics_path") else None
preds_csv = Path(preds_path) if preds_path else None
summary_txt = None

# Selected run banner
colA, colB = st.columns([2, 3])
with colA:
    st.markdown("**Selected run**")
    st.code(run_id)
with colB:
    st.markdown("**Run path**")
    st.code(str(run_dir))

if not found:
    st.warning("No cross-dataset files in this selected run.")
    # Show Generalization Gap block with N/A so page still useful
    try:
        from dashboard.components.generalization import render_generalization_gap_block
        render_generalization_gap_block(run_dir)
    except Exception:
        pass
    # Prefer newest SUCCESS run with cross-dataset outputs (from layout discovery)
    try:
        from dashboard.components.layout import discover_runs, get_newest_success_run_with_cross_dataset
        discovered = discover_runs()
        newest_with = get_newest_success_run_with_cross_dataset(discovered)
    except Exception:
        newest_with = _newest_run_with_cross_dataset_report()
    if newest_with:
        st.info(f"Newest SUCCESS run with cross-dataset outputs: **{newest_with}**")
        if st.button("Switch to that run", type="primary"):
            st.session_state["selected_run_id"] = newest_with
            st.rerun()
    else:
        st.info("No run in outputs/runs contains cross-dataset outputs. Run the pipeline with cross-dataset evaluation, e.g.:")
        st.code("python3 main.py --dataset diabetes --cross_dataset heart_disease")
    st.stop()

# Debug expander (collapsed, safe)
with st.expander("Debug: paths and JSON keys", expanded=False):
    try:
        st.text(f"resolved run_dir: {run_dir}")
        st.text(f"cross_dataset_report.json exists: {report_json is not None and report_json.exists() if report_json else False}")
        st.text(f"cross_dataset_metrics.json exists: {metrics_json is not None and metrics_json.exists()}")
        st.text(f"cross_dataset_predictions.csv exists: {preds_csv is not None and preds_csv.exists()}")
        st.text("Keys in report JSON: " + (", ".join(sorted(report.keys())) if isinstance(report, dict) else str(type(report))))
        st.text("Keys in metrics JSON: " + (", ".join(sorted(metrics.keys())) if isinstance(metrics, dict) else str(type(metrics))))
    except Exception as e:
        st.caption(f"Debug info unavailable: {e}")

# Extract display fields: metrics primary, report fallback; handle train_dataset/source_dataset, selected_model/model_used, f1_score/f1
(
    source_dataset,
    target_dataset,
    eval_split,
    model_used,
    status,
    n_samples,
    roc_auc,
    acc,
    prec,
    rec,
    f1,
) = _extract_cross_dataset_fields(report, metrics, preds)

st.subheader("Result Summary")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Status", status)
c2.metric("Source", source_dataset)
c3.metric("Target", target_dataset)
c4.metric("Eval split", eval_split)
c5.metric("Model used", model_used)
c6.metric("N samples", n_samples)

st.markdown("")
st.subheader("Metrics")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("ROC-AUC", _fmt(roc_auc))
k2.metric("Accuracy", _fmt(acc))
k3.metric("Precision", _fmt(prec))
k4.metric("Recall", _fmt(rec))
k5.metric("F1", _fmt(f1))

# ---------- Generalization Gap (shared with Overview) ----------
st.markdown("")
try:
    from dashboard.components.generalization import render_generalization_gap_block
    render_generalization_gap_block(run_dir)
except Exception as e:
    st.caption(f"Generalization gap unavailable: {e}")

# Predictions preview (head 20)
st.subheader("Predictions Preview")
if preds.empty:
    st.info("No predictions CSV found (or it's empty).")
else:
    st.dataframe(preds.head(10), use_container_width=True)

# Full text of summary if present
if summary_txt:
    st.subheader("Cross-Dataset Summary")
    try:
        summary_content = summary_txt.read_text(encoding="utf-8")
        st.text(summary_content)
    except Exception:
        st.caption("(Could not read summary file.)")

# Interpretation expander
with st.expander("How to interpret this (quick)", expanded=True):
    st.write(
        "This test is intentionally hard: you train on one dataset and evaluate on a different one. "
        "A big performance drop is normal. What matters is that the pipeline produces a **reproducible** "
        "cross-dataset report and you can discuss why transfer is difficult (feature mismatch, label differences, population shift)."
    )
    if roc_auc is not None:
        try:
            if float(roc_auc) < 0.65:
                st.warning(
                    "Your current cross-dataset ROC-AUC is low. That's not a failure of implementation — "
                    "it's evidence that the datasets differ a lot. Document it as a generalization gap."
                )
        except Exception:
            pass

# Files and downloads
st.subheader("Files")
file_cols = st.columns(4)
with file_cols[0]:
    st.markdown("**Report (json)**")
    if report_json:
        try:
            st.caption(str(report_json.relative_to(run_dir)))
        except ValueError:
            st.caption(report_json.name)
        st.download_button(
            "Download report.json",
            data=report_json.read_bytes(),
            file_name="cross_dataset_report.json",
            mime="application/json",
            key="dl_report",
        )
    else:
        st.write("—")

with file_cols[1]:
    st.markdown("**Metrics (json)**")
    if metrics_json:
        try:
            st.caption(str(metrics_json.relative_to(run_dir)))
        except ValueError:
            st.caption(metrics_json.name)
        st.download_button(
            "Download metrics.json",
            data=metrics_json.read_bytes(),
            file_name="cross_dataset_metrics.json",
            mime="application/json",
            key="dl_metrics",
        )
    else:
        st.write("—")

with file_cols[2]:
    st.markdown("**Predictions (csv)**")
    if preds_csv:
        try:
            st.caption(str(preds_csv.relative_to(run_dir)))
        except ValueError:
            st.caption(preds_csv.name)
        st.download_button(
            "Download predictions.csv",
            data=preds_csv.read_bytes(),
            file_name="cross_dataset_predictions.csv",
            mime="text/csv",
            key="dl_preds",
        )
    else:
        st.write("—")

with file_cols[3]:
    st.markdown("**Summary (txt)**")
    if summary_txt:
        try:
            st.caption(str(summary_txt.relative_to(run_dir)))
        except ValueError:
            st.caption(summary_txt.name)
        st.download_button(
            "Download summary.txt",
            data=summary_txt.read_bytes(),
            file_name="cross_dataset_summary.txt",
            mime="text/plain",
            key="dl_summary",
        )
    else:
        st.write("—")

# Raw JSON preview
st.subheader("Raw JSON Preview")
with st.expander("cross_dataset_report.json (preview)", expanded=False):
    if report:
        st.json(report)
    else:
        st.write("—")
with st.expander("cross_dataset_metrics.json (preview)", expanded=False):
    if metrics:
        st.json(metrics)
    else:
        st.write("—")
