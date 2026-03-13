"""
Generalization Gap helpers: primary vs cross-dataset ROC-AUC and gap.
Used by Overview and Cross_Dataset pages. All paths derived from run_dir (from session/sidebar).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json


def load_json(path: Optional[Path]) -> Optional[dict]:
    """Safe JSON loader. Returns None on missing path or parse error."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def find_first_existing(paths: List[Path]) -> Optional[Path]:
    """Return the first path that exists and is a file."""
    for p in paths:
        p = Path(p)
        if p.exists() and p.is_file():
            return p
    return None


def get_run_paths(run_dir: Path) -> Dict[str, Path]:
    """Return resolved dirs for a run: research_outputs, multi_agent/research_outputs, reports_dir, models_dir."""
    run_dir = Path(run_dir)
    try:
        from dashboard.components.layout import resolve_paths
        resolved = resolve_paths(run_dir)
    except Exception:
        resolved = {}
    return {
        "reports_dir": resolved.get("reports_dir") or run_dir / "reports",
        "models_dir": resolved.get("models_dir") or run_dir / "models",
        "research_outputs_dir": run_dir / "research_outputs",
        "multi_agent_research_outputs_dir": run_dir / "multi_agent" / "research_outputs",
    }


def get_primary_best_roc_auc(run_dir: Path) -> Optional[float]:
    """
    In-dataset ROC-AUC = best model ROC-AUC on primary dataset.
    Prefer research_outputs (non-multi_agent) then multi_agent/research_outputs.
    Sources: multi_agent_results.json (selected_metrics.roc_auc), baseline_results.json, model_metrics.json.
    """
    run_dir = Path(run_dir)
    paths = get_run_paths(run_dir)
    ro = paths["research_outputs_dir"]
    ma_ro = paths["multi_agent_research_outputs_dir"]
    reports_dir = paths["reports_dir"]
    models_dir = paths["models_dir"]

    # (1) multi_agent_results.json -> selected_metrics.roc_auc
    for base in (ro, ma_ro, reports_dir):
        data = load_json(base / "multi_agent_results.json")
        if not data:
            continue
        models = data.get("models") or {}
        sel_metrics = models.get("selected_metrics")
        if isinstance(sel_metrics, dict) and sel_metrics.get("roc_auc") is not None:
            try:
                return float(sel_metrics["roc_auc"])
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

    # (2) baseline_results.json -> selected_metrics.roc_auc
    for base in (ro, ma_ro, reports_dir):
        data = load_json(base / "baseline_results.json")
        if not data:
            continue
        sel_metrics = data.get("selected_metrics")
        if isinstance(sel_metrics, dict) and sel_metrics.get("roc_auc") is not None:
            try:
                return float(sel_metrics["roc_auc"])
            except (TypeError, ValueError):
                pass

    # (3) model_metrics.json or reports/model_metrics.json -> max roc_auc across models
    for base in (models_dir, reports_dir, ro, ma_ro):
        if not base.exists():
            continue
        for f in list(base.glob("model_metrics*.json")) + list(base.glob("*model_metrics*.json")):
            data = load_json(f)
            if not isinstance(data, dict):
                continue
            best_auc = None
            for v in data.values():
                if isinstance(v, dict) and v.get("roc_auc") is not None:
                    try:
                        x = float(v["roc_auc"])
                        if best_auc is None or x > best_auc:
                            best_auc = x
                    except (TypeError, ValueError):
                        pass
            if best_auc is not None:
                return best_auc
    return None


def get_best_cross_roc_auc(run_dir: Path) -> Optional[float]:
    """
    Cross-dataset ROC-AUC from cross_dataset_report.json (or cross_dataset_metrics.json).
    Prefer: best_model_on_target.roc_auc, else max ROC-AUC across results/models in report.
    """
    run_dir = Path(run_dir)
    paths = get_run_paths(run_dir)
    ro = paths["research_outputs_dir"]
    ma_ro = paths["multi_agent_research_outputs_dir"]

    # Prefer non-multi_agent first
    candidates = [
        ro / "cross_dataset_report.json",
        ro / "cross_dataset_metrics.json",
        ma_ro / "cross_dataset_report.json",
        ma_ro / "cross_dataset_metrics.json",
    ]
    report_path = find_first_existing(candidates)
    if not report_path:
        return None

    data = load_json(report_path)
    if not isinstance(data, dict):
        return None

    # best_model_on_target.roc_auc
    best_on_target = data.get("best_model_on_target")
    if isinstance(best_on_target, dict) and best_on_target.get("roc_auc") is not None:
        try:
            return float(best_on_target["roc_auc"])
        except (TypeError, ValueError):
            pass

    # report.metrics.roc_auc or canonical_metrics.roc_auc
    for key in ("metrics", "canonical_metrics"):
        m = data.get(key)
        if isinstance(m, dict) and m.get("roc_auc") is not None:
            try:
                return float(m["roc_auc"])
            except (TypeError, ValueError):
                pass

    # max over results[].cross_dataset_metrics.roc_auc
    results = data.get("results") or data.get("models") or []
    if isinstance(results, dict):
        results = list(results.values())
    if not isinstance(results, list):
        results = []
    best = None
    for r in results:
        if not isinstance(r, dict):
            continue
        cross_m = r.get("cross_dataset_metrics") or r
        if isinstance(cross_m, dict) and cross_m.get("roc_auc") is not None:
            try:
                x = float(cross_m["roc_auc"])
                if best is None or x > best:
                    best = x
            except (TypeError, ValueError):
                pass
    if best is not None:
        return best

    # metrics dict keyed by model name (e.g. cross_dataset_metrics.json)
    metrics = data.get("metrics")
    if isinstance(metrics, dict):
        for v in metrics.values():
            if isinstance(v, dict) and v.get("roc_auc") is not None:
                try:
                    x = float(v["roc_auc"])
                    if best is None or x > best:
                        best = x
                except (TypeError, ValueError):
                    pass
    return best


def compute_generalization_gap(
    primary_auc: Optional[float],
    cross_auc: Optional[float],
) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (gap, pct_drop).
    gap = primary_auc - cross_auc; pct_drop = gap / primary_auc if primary_auc > 0.
    """
    if primary_auc is None or cross_auc is None:
        return None, None
    try:
        gap = float(primary_auc) - float(cross_auc)
        pct = (gap / float(primary_auc)) * 100.0 if primary_auc else None
        return gap, pct
    except (TypeError, ValueError, ZeroDivisionError):
        return None, None


def _fmt_auc(value: Optional[float], decimals: int = 4) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def render_generalization_gap_block(run_dir: Path) -> None:
    """
    Render the Generalization Gap KPI block (3 cards + caption).
    Uses run_dir derived from selected run. Shows "—" and friendly note when data missing.
    """
    import streamlit as st
    from dashboard.ui import kpi_card, section_header

    run_dir = Path(run_dir)
    primary_auc = get_primary_best_roc_auc(run_dir)
    cross_auc = get_best_cross_roc_auc(run_dir)
    gap, pct_drop = compute_generalization_gap(primary_auc, cross_auc)

    section_header("Generalization Gap", caption="In-dataset ROC-AUC − Cross-dataset ROC-AUC. % drop = gap / in-dataset ROC-AUC.")
    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("In-dataset ROC-AUC", _fmt_auc(primary_auc), max_value_len=10)
    with c2:
        kpi_card("Cross-dataset ROC-AUC", _fmt_auc(cross_auc), max_value_len=10)
    with c3:
        gap_str = _fmt_auc(gap) if gap is not None else "—"
        sub = f"{pct_drop:.1f}% drop" if pct_drop is not None else None
        kpi_card("Generalization Gap", gap_str, subtitle=sub, max_value_len=10)

    if primary_auc is None or cross_auc is None:
        st.caption("Run this pipeline with cross-dataset enabled to populate. Select a run that has cross-dataset outputs for full metrics.")
