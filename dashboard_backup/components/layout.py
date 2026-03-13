"""Layout utilities for dashboard."""

import streamlit as st
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
RUNS_BASE = PROJECT_ROOT / "outputs" / "runs"

# Status constants for run discovery
RUN_STATUS_SUCCESS = "SUCCESS"
RUN_STATUS_FAILED = "FAILED"
RUN_STATUS_INCOMPLETE = "INCOMPLETE"


def discover_runs() -> List[dict]:
    """
    Canonical run discovery: scan outputs/runs/run_*, compute has_artifacts, has_cross_dataset, status.
    Returns list of dicts: run_id, path, mtime, has_any_artifacts, has_cross_dataset, status, display_label.
    Sorted by mtime descending. Respects run_dir_override.
    """
    override = _get_run_dir_override()
    if override is not None and override.exists():
        run_dirs = [override]
    elif not RUNS_BASE.exists():
        return []
    else:
        run_dirs = [d for d in RUNS_BASE.iterdir() if d.is_dir() and d.name.startswith("run_")]

    result = []
    for run_path in run_dirs:
        run_id = run_path.name
        try:
            mtime = run_path.stat().st_mtime
        except OSError:
            mtime = 0.0

        # has_any_artifacts: known result files or at least one json in research_outputs
        ro = run_path / "research_outputs"
        ma_ro = run_path / "multi_agent" / "research_outputs"
        reports = run_path / "reports"
        has_any_artifacts = (
            (ro / "multi_agent_results.json").exists()
            or (ro / "baseline_results.json").exists()
            or (reports / "multi_agent_results.json").exists()
            or (ma_ro / "multi_agent_results.json").exists()
            or (ma_ro / "baseline_results.json").exists()
        )
        if not has_any_artifacts and ro.exists():
            has_any_artifacts = any(ro.iterdir()) and any(f.suffix == ".json" for f in ro.iterdir() if f.is_file())
        if not has_any_artifacts and ma_ro.exists():
            has_any_artifacts = any(f.suffix == ".json" for f in ma_ro.iterdir() if f.is_file())

        # has_cross_dataset: cross_dataset_report.json | metrics | predictions in ro or multi_agent/research_outputs
        cross_bases = [ro, ma_ro]
        has_cross_dataset = False
        for base in cross_bases:
            if not base.exists():
                continue
            for name in ("cross_dataset_report.json", "cross_dataset_metrics.json", "cross_dataset_predictions.csv"):
                if (base / name).exists():
                    has_cross_dataset = True
                    break
            if has_cross_dataset:
                break

        # status: SUCCESS / FAILED / INCOMPLETE
        logs_dir = run_path / "logs"
        ma_logs = run_path / "multi_agent" / "logs"
        has_logs = False
        log_content = ""
        for ld in (logs_dir, ma_logs):
            if ld.exists() and ld.is_dir():
                log_files = list(ld.glob("*.log"))
                if log_files:
                    has_logs = True
                    try:
                        with open(log_files[0], "r", encoding="utf-8", errors="ignore") as f:
                            log_content = f.read()
                    except Exception:
                        pass
                    break

        if has_any_artifacts and has_logs:
            status = RUN_STATUS_SUCCESS
            if log_content and (
                "Traceback" in log_content
                or "segmentation fault" in log_content.lower()
                or "ERROR" in log_content
            ):
                status = RUN_STATUS_FAILED
        elif has_logs:
            status = RUN_STATUS_FAILED
        else:
            status = RUN_STATUS_INCOMPLETE

        display_label = format_run_name(run_id)
        result.append({
            "run_id": run_id,
            "path": str(run_path.resolve()),
            "mtime": mtime,
            "has_any_artifacts": has_any_artifacts,
            "has_cross_dataset": has_cross_dataset,
            "status": status,
            "display_label": display_label,
        })

    result.sort(key=lambda x: x["mtime"], reverse=True)
    return result


def get_default_run_id(runs: List[dict]) -> Optional[str]:
    """Prefer newest run with status SUCCESS; else newest overall."""
    for r in runs:
        if r.get("status") == RUN_STATUS_SUCCESS:
            return r["run_id"]
    return runs[0]["run_id"] if runs else None


def get_newest_success_run_with_cross_dataset(runs: List[dict]) -> Optional[str]:
    """Return run_id of newest SUCCESS run that has cross-dataset outputs."""
    for r in runs:
        if r.get("status") == RUN_STATUS_SUCCESS and r.get("has_cross_dataset"):
            return r["run_id"]
    return None


def resolve_paths(run_dir: Path) -> Dict[str, Path]:
    """
    Resolve artifact directories for a run, supporting both new layout (multi_agent/, baseline/, research_outputs/)
    and legacy layout (reports/, models/, explainability/, figures/ at run root).

    Search order per key:
      a) run_dir/multi_agent/*
      b) run_dir/baseline/*
      c) run_dir/research_outputs/* (reports only)
      d) run_dir/* (legacy root)

    Returns dict: data_dir, models_dir, reports_dir, explainability_dir, figures_dir, logs_dir.
    For each key returns the first path that exists as a directory; otherwise the legacy path.
    """
    run_dir = Path(run_dir)
    ma = run_dir / "multi_agent"
    bl = run_dir / "baseline"
    ro = run_dir / "research_outputs"

    def first_existing(candidates: List[Path]) -> Path:
        for p in candidates:
            if p.exists() and p.is_dir():
                return p
        return candidates[-1] if candidates else run_dir

    # Reports: research_outputs preferred (comparison_report.csv, etc.), then reports
    reports_candidates = [
        ma / "research_outputs",
        ma / "reports",
        bl / "reports",
        ro,
        run_dir / "reports",
    ]
    reports_dir = first_existing(reports_candidates)

    return {
        "data_dir": first_existing([ma / "data", bl / "data", run_dir / "data"]),
        "models_dir": first_existing([ma / "models", bl / "models", run_dir / "models"]),
        "reports_dir": reports_dir,
        "explainability_dir": first_existing([ma / "explainability", bl / "explainability", run_dir / "explainability"]),
        "figures_dir": first_existing([ma / "figures", bl / "figures", run_dir / "figures"]),
        "logs_dir": first_existing([ma / "logs", bl / "logs", run_dir / "logs"]),
    }


def _get_run_dir_override() -> Optional[Path]:
    """Return override run directory from session state (set by app.py from --run_dir CLI)."""
    override = st.session_state.get("run_dir_override")
    if override is None:
        return None
    p = Path(override) if not isinstance(override, Path) else override
    return p.resolve() if p.exists() else None


def get_run_dir_path(run_id: str) -> Path:
    """Resolve run directory path. Uses --run_dir override when it matches run_id."""
    override = _get_run_dir_override()
    if override is not None and override.name == run_id:
        return override
    return (RUNS_BASE / run_id).resolve()


def get_reports_dir(run_dir: Path) -> Path:
    """Directory for reports/comparison (uses resolve_paths: research_outputs when present, else reports)."""
    return resolve_paths(Path(run_dir))["reports_dir"]


def get_newest_run_dir() -> Optional[Path]:
    """Return path to newest folder in outputs/runs whose name starts with 'run_', by modified time."""
    if not RUNS_BASE.exists():
        return None
    run_dirs = [d for d in RUNS_BASE.iterdir() if d.is_dir() and d.name.startswith("run_")]
    if not run_dirs:
        return None
    return max(run_dirs, key=lambda d: d.stat().st_mtime)


def get_available_runs(dataset_filter: Optional[str] = 'diabetes') -> List[str]:
    """Get list of available run IDs. Uses app.py's run_ids_newest_first when set (newest first). Override via --run_dir uses only that run."""
    override = _get_run_dir_override()
    if override is not None:
        return [override.name]

    # Use run list from app.py (newest first) when set; otherwise discover here
    all_runs = st.session_state.get("run_ids_newest_first")
    if all_runs is None:
        if not RUNS_BASE.exists():
            return []
        run_dirs = [d for d in RUNS_BASE.iterdir() if d.is_dir() and d.name.startswith("run_")]
        run_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        all_runs = [d.name for d in run_dirs]

    if not dataset_filter:
        return all_runs

    filtered = []
    for run_id in all_runs:
        run_dir = get_run_dir_path(run_id)
        paths = resolve_paths(run_dir)
        has_dataset = (
            (paths['reports_dir'] / f'data_metadata_{dataset_filter}.json').exists()
            or (paths['data_dir'] / f'{dataset_filter}_train.csv').exists()
        )
        if not has_dataset:
            meta = get_run_metadata(run_id)
            if meta and meta.get('datasets'):
                has_dataset = dataset_filter in meta['datasets']
        if has_dataset:
            filtered.append(run_id)
    return filtered


def format_run_name(run_id: str) -> str:
    """Format run ID into a readable display name."""
    from datetime import datetime
    
    # Handle short format: run_001, run_002, ...
    if run_id.startswith('run_'):
        suffix = run_id[4:]  # after 'run_'
        if suffix.isdigit():
            return f"Run {int(suffix)}"
    
    # Handle timestamp format: run_2026-01-20_19-21-43_dd426bb9
    if run_id.startswith('run_'):
        parts = run_id.replace('run_', '').split('_')
        if len(parts) >= 2:
            date_part = parts[0]  # 2026-01-20
            time_part = parts[1]  # 19-21-43
            
            # Format date and time nicely
            try:
                date_obj = datetime.strptime(date_part, "%Y-%m-%d")
                formatted_date = date_obj.strftime("%b %d, %Y")  # Jan 20, 2026
                formatted_time = time_part.replace('-', ':')  # 19:21:43
                return f"{formatted_date} at {formatted_time}"
            except:
                return f"{date_part} {time_part}"
    
    # Handle old format: 20260120_191140_ee21f377
    if '_' in run_id:
        parts = run_id.split('_')
        if len(parts) >= 2:
            date_str = parts[0]  # 20260120
            time_str = parts[1]  # 191140
            
            try:
                # Try to parse date
                if len(date_str) == 8:  # YYYYMMDD format
                    date_obj = datetime.strptime(date_str, "%Y%m%d")
                    formatted_date = date_obj.strftime("%b %d, %Y")  # Jan 20, 2026
                    
                    # Format time: 191140 -> 19:11:40
                    if len(time_str) == 6 and time_str.isdigit():
                        formatted_time = f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
                        return f"{formatted_date} at {formatted_time}"
                    else:
                        # If time format is different, just show date
                        return formatted_date
                else:
                    # If date format is different, try to parse as-is
                    return run_id
            except:
                # If parsing fails, return a shortened version
                if len(run_id) > 30:
                    return f"{run_id[:20]}..."
                return run_id
    
    # Fallback: return shortened version if too long
    if len(run_id) > 30:
        return f"{run_id[:20]}..."
    return run_id


def is_run_valid(run_dir: Path) -> bool:
    """
    A run is valid if it contains at least one of:
    - run_metadata.json (run root, reports, research_outputs, or multi_agent/reports, multi_agent/research_outputs)
    - reports/comparison_report.csv or research_outputs/comparison_report.csv (under run_dir or multi_agent)
    - multi_agent/models/model_metrics.json or models/model_metrics.json
    """
    run_dir = Path(run_dir)
    # Check run_metadata.json
    for p in (
        run_dir / "run_metadata.json",
        run_dir / "reports" / "run_metadata.json",
        run_dir / "research_outputs" / "run_metadata.json",
        run_dir / "multi_agent" / "reports" / "run_metadata.json",
        run_dir / "multi_agent" / "research_outputs" / "run_metadata.json",
    ):
        if p.exists():
            return True
    # Check comparison_report.csv (look under multi_agent first)
    for base in (run_dir / "multi_agent", run_dir):
        for sub in ("research_outputs", "reports"):
            if (base / sub / "comparison_report.csv").exists():
                return True
    # Check model_metrics.json (multi_agent first)
    for p in (
        run_dir / "multi_agent" / "models" / "model_metrics.json",
        run_dir / "models" / "model_metrics.json",
        run_dir / "multi_agent" / "reports" / "model_metrics.json",
        run_dir / "reports" / "model_metrics.json",
    ):
        if p.exists():
            return True
    return False


def get_run_log_reason(run_dir: Path, max_lines: int = 30) -> str:
    """Return last lines of logs/run.log if present, else empty string."""
    run_dir = Path(run_dir)
    for log_path in (run_dir / "logs" / "run.log", run_dir / "multi_agent" / "logs" / "run.log"):
        if log_path.exists():
            try:
                with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
                tail = lines[-max_lines:] if len(lines) > max_lines else lines
                text = "".join(tail).strip()
                return text if text else ""
            except Exception:
                pass
    return ""


def get_run_status(run_id: str) -> Tuple[str, str]:
    """
    Returns (status, reason). status is one of 'success', 'failed', 'unknown'.
    If run is invalid (see is_run_valid), status is forced to 'failed' and reason from logs/run.log or generic message.
    Where status is computed: here (layout) using is_run_valid + metadata + run.log.
    """
    run_dir = get_run_dir_path(run_id)
    valid = is_run_valid(run_dir)
    metadata = get_run_metadata(run_id) or {}
    if not valid:
        reason = get_run_log_reason(run_dir)
        return ("failed", reason or "Run failed or incomplete — check logs.")
    status = (metadata.get("status") or "unknown").lower()
    if status not in ("success", "failed", "unknown"):
        status = "unknown"
    reason = (metadata.get("error") or "") if status == "failed" else ""
    if status == "failed" and not reason:
        reason = get_run_log_reason(run_dir) or "Pipeline reported failure."
    return (status, reason)


def get_run_metadata(run_id: str) -> Optional[dict]:
    """Load metadata for a specific run - checks multiple locations (multi_agent first for reports)."""
    run_dir = get_run_dir_path(run_id)

    # Try multiple locations (multi_agent first, then research_outputs, reports, root)
    metadata_paths = [
        run_dir / 'multi_agent' / 'research_outputs' / 'run_metadata.json',
        run_dir / 'multi_agent' / 'reports' / 'run_metadata.json',
        run_dir / 'reports' / 'run_metadata.json',
        run_dir / 'run_metadata.json',
        run_dir / 'research_outputs' / 'run_metadata.json',
    ]
    
    for metadata_path in metadata_paths:
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    # Ensure required fields exist
                    if 'run_id' not in metadata:
                        metadata['run_id'] = run_id
                    if 'status' not in metadata:
                        metadata['status'] = 'unknown'
                    if 'timestamp' not in metadata:
                        from datetime import datetime
                        metadata['timestamp'] = datetime.now().isoformat()
                    return metadata
            except Exception as e:
                continue
    
    # Return minimal metadata if not found
    return {
        'run_id': run_id,
        'status': 'unknown',
        'timestamp': datetime.now().isoformat() if 'datetime' in dir() else '',
        'datasets': []
    }


def get_best_model_for_run(run_id: str, dataset: str) -> Optional[str]:
    """Get best model name for this run (from run_metadata or model_metrics). Looks under multi_agent first."""
    meta = get_run_metadata(run_id)
    if meta and meta.get("best_model_multi_agent"):
        return meta["best_model_multi_agent"]
    run_dir = get_run_dir_path(run_id)
    for path in [
        run_dir / "multi_agent" / "models" / "model_metrics.json",
        run_dir / "multi_agent" / "reports" / "model_metrics.json",
        run_dir / "models" / "model_metrics.json",
        run_dir / "reports" / "model_metrics.json",
    ]:
        if path.exists():
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict) and data:
                    best = max(data.keys(), key=lambda m: data[m].get("roc_auc", 0) if isinstance(data[m], dict) else 0)
                    return best
            except Exception:
                continue
    return None


def count_run_artifacts(run_dir: Path) -> Tuple[int, int, int]:
    """Count .csv, .json, .png files under run_dir (recursive). Returns (n_csv, n_json, n_png)."""
    run_dir = Path(run_dir)
    n_csv = n_json = n_png = 0
    for f in run_dir.rglob("*"):
        if f.is_file():
            if f.suffix.lower() == ".csv":
                n_csv += 1
            elif f.suffix.lower() == ".json":
                n_json += 1
            elif f.suffix.lower() == ".png":
                n_png += 1
    return n_csv, n_json, n_png


# Expected files by section: map section -> list of (resolved_path_key, glob_pattern)
# Used with resolve_paths() so both new and legacy layouts are checked.
EXPECTED_BY_SECTION = {
    "Data": [
        ("data_dir", "*_train.csv"),
        ("data_dir", "*_val.csv"),
        ("data_dir", "*_test.csv"),
        ("reports_dir", "data_metadata_*.json"),
    ],
    "Models": [
        ("models_dir", "*.pkl"),
        ("models_dir", "model_metrics.json"),
        ("reports_dir", "model_metrics.json"),
    ],
    "Reports": [
        ("reports_dir", "run_metadata.json"),
        ("reports_dir", "comparison_report.csv"),
    ],
    "Explainability": [
        ("explainability_dir", "explanations.json"),
        ("figures_dir", "*.png"),
    ],
}


def _dir_matches_pattern(directory: Path, pattern: str) -> bool:
    """Return True if directory exists and contains any file matching pattern."""
    if not directory.exists() or not directory.is_dir():
        return False
    return any(directory.glob(pattern))


def get_missing_expected_files(run_dir: Path) -> List[Tuple[str, List[str]]]:
    """Return list of (section_name, [missing patterns]) using resolved paths (multi_agent, baseline, research_outputs, legacy)."""
    run_dir = Path(run_dir)
    paths = resolve_paths(run_dir)
    missing_by_section = []
    for section, spec_list in EXPECTED_BY_SECTION.items():
        found_any = False
        for key, pattern in spec_list:
            dir_path = paths.get(key)
            if dir_path and _dir_matches_pattern(dir_path, pattern):
                found_any = True
                break
        if not found_any:
            # Display patterns in legacy form for UI
            display = {
                "Data": ["data/*_train.csv", "data/*_val.csv", "data/*_test.csv", "reports/data_metadata_*.json"],
                "Models": ["models/*.pkl", "models/model_metrics.json", "reports/model_metrics.json"],
                "Reports": ["reports/run_metadata.json", "reports/comparison_report.csv"],
                "Explainability": ["explainability/explanations.json", "figures/*.png"],
            }
            missing_by_section.append((section, display.get(section, [pattern for _, pattern in spec_list])))
    return missing_by_section


def render_sidebar():
    """Render compact sidebar: run, status chips (Dataset, Feedback, Retraining), view artifacts (model), defense mode, run pack."""
    st.sidebar.markdown(
        '<div style="font-size:1.1rem; font-weight:700; color:#2d3748; margin-bottom:0.5rem;">🔬 Pipeline Dashboard</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")

    # Use canonical discovered runs (from app.py); fallback to discover now if missing
    discovered = st.session_state.get("discovered_runs")
    if not discovered:
        discovered = discover_runs()
        st.session_state["discovered_runs"] = discovered
    if not discovered:
        st.sidebar.warning("No runs found. Run `python main.py` first.")
        return None, None, None, None, "—", "No"

    # Session state is source of truth: selected_run_id. Only fix when missing or invalid (do not overwrite user choice on rerun).
    run_ids = [r["run_id"] for r in discovered]
    if "selected_run_id" not in st.session_state or st.session_state["selected_run_id"] not in run_ids:
        st.session_state["selected_run_id"] = get_default_run_id(discovered) or (run_ids[0] if run_ids else None)

    # Toggle: show failed/incomplete runs (default OFF = only SUCCESS)
    show_all_runs = st.sidebar.checkbox("Show failed/incomplete runs", value=False, key="show_failed_incomplete_runs")
    if show_all_runs:
        run_options = [r["run_id"] for r in discovered]
        run_labels = [f"{r['display_label']} ({r['status']})" for r in discovered]
    else:
        success_runs = [r for r in discovered if r["status"] == RUN_STATUS_SUCCESS]
        run_options = [r["run_id"] for r in success_runs]
        run_labels = [r["display_label"] for r in success_runs]
        if not run_options:
            run_options = [r["run_id"] for r in discovered]
            run_labels = [f"{r['display_label']} ({r['status']})" for r in discovered]
        # Only reset when current selection is not in the filtered list (e.g. selected run is failed)
        elif st.session_state["selected_run_id"] not in run_options:
            st.session_state["selected_run_id"] = run_options[0]

    # Stable options: run_id strings. Sync widget state from session only when widget state is missing or invalid.
    run_id_to_label = dict(zip(run_options, run_labels))
    _current = st.session_state.get("selected_run_id")
    if _current not in run_options:
        _current = run_options[0] if run_options else None
        if _current:
            st.session_state["selected_run_id"] = _current
    if "run_selectbox" not in st.session_state or st.session_state.get("run_selectbox") not in run_options:
        st.session_state["run_selectbox"] = _current

    def _on_run_change():
        if "run_selectbox" in st.session_state:
            st.session_state["selected_run_id"] = st.session_state["run_selectbox"]

    selected_run_id = st.sidebar.selectbox(
        "Select Run",
        options=run_options,
        format_func=lambda x: run_id_to_label.get(x, format_run_name(x)),
        key="run_selectbox",
        on_change=_on_run_change,
    )
    # Persist widget value to session (single source of truth for pages)
    st.session_state["selected_run_id"] = selected_run_id
    run_id = st.session_state["selected_run_id"]
    run_dir = get_run_dir_path(run_id)

    # Current run status from discovery (for accurate warning)
    current_run_info = next((r for r in discovered if r["run_id"] == run_id), None)
    run_status = (current_run_info or {}).get("status", RUN_STATUS_INCOMPLETE)

    metadata = get_run_metadata(run_id) or {}

    # Status is computed from discovery; show warning only when not SUCCESS
    valid = is_run_valid(run_dir)
    status, status_reason = get_run_status(run_id)
    status_label = "Success" if run_status == RUN_STATUS_SUCCESS else ("Failed" if run_status == RUN_STATUS_FAILED else "Incomplete")
    sc_border, sc_bg = ("#38a169", "#f0fff4") if run_status == RUN_STATUS_SUCCESS else (("#d69e2e", "#fffff0") if run_status == RUN_STATUS_FAILED else ("#718096", "#f7fafc"))

    # Selected run info
    st.sidebar.markdown("**Run folder name**")
    st.sidebar.markdown(f"`{run_id}`")
    st.sidebar.markdown("**Full path**")
    st.sidebar.caption(f"📁 `{run_dir}`")
    n_csv, n_json, n_png = count_run_artifacts(run_dir)
    st.sidebar.caption(f"📊 **{n_csv}** CSV · **{n_json}** JSON · **{n_png}** PNG")

    # Only show "Run failed or incomplete" when status != SUCCESS
    if run_status != RUN_STATUS_SUCCESS:
        st.sidebar.warning("**Run failed or incomplete — check logs.**")
        if status_reason:
            with st.sidebar.expander("Log excerpt", expanded=False):
                st.text(status_reason[:2000] + ("..." if len(status_reason) > 2000 else ""))
    else:
        missing = get_missing_expected_files(run_dir)
        if missing:
            warn_lines = ["**Missing expected files:**"]
            for section, patterns in missing:
                warn_lines.append(f"- **{section}:** " + ", ".join(patterns))
            st.sidebar.warning("\n".join(warn_lines))

    # Status chip
    st.sidebar.markdown(
        f'<div style="display:inline-block; padding:0.2rem 0.5rem; border-radius:999px; background:{sc_bg}; border:1px solid {sc_border}; font-size:0.78rem; color:#2d3748; margin-bottom:0.35rem;">Run: <strong>{status_label}</strong></div>',
        unsafe_allow_html=True,
    )

    # Optional debug expander
    with st.sidebar.expander("Debug: run selection", expanded=False):
        st.text(f"selected_run_id: {st.session_state.get('selected_run_id')}")
        st.text(f"run_dir: {run_dir}")
        st.text(f"status: {run_status}")
        st.text(f"has_cross_dataset: {(current_run_info or {}).get('has_cross_dataset', False)}")

    # Datasets that exist in this run (from run_metadata or data_metadata files)
    datasets_in_run = list(metadata.get("datasets", [])) or []
    reports_dir = get_reports_dir(run_dir)
    if not datasets_in_run and reports_dir.exists():
        for f in reports_dir.iterdir():
            if f.is_file() and f.name.startswith("data_metadata_") and f.suffix == ".json":
                ds = f.name.replace("data_metadata_", "").replace(".json", "")
                if ds and ds not in datasets_in_run:
                    datasets_in_run.append(ds)
    if not datasets_in_run and (run_dir / "multi_agent" / "research_outputs").exists():
        for f in (run_dir / "multi_agent" / "research_outputs").iterdir():
            if f.is_file() and f.name.startswith("data_metadata_") and f.suffix == ".json":
                ds = f.name.replace("data_metadata_", "").replace(".json", "")
                if ds and ds not in datasets_in_run:
                    datasets_in_run.append(ds)
    if not datasets_in_run and (run_dir / "data").exists():
        for f in (run_dir / "data").glob("*_train.csv"):
            ds = f.stem.replace("_train", "")
            if ds and ds not in datasets_in_run:
                datasets_in_run.append(ds)
    if not datasets_in_run and (run_dir / "multi_agent" / "data").exists():
        for f in (run_dir / "multi_agent" / "data").glob("*_train.csv"):
            ds = f.stem.replace("_train", "")
            if ds and ds not in datasets_in_run:
                datasets_in_run.append(ds)
    try:
        from utils.config import get_config
        all_config_datasets = list(get_config().data.dataset_urls.keys()) if hasattr(get_config().data, "dataset_urls") else ["diabetes", "heart_disease"]
    except Exception:
        all_config_datasets = ["diabetes", "heart_disease"]
    if not all_config_datasets:
        all_config_datasets = ["diabetes", "heart_disease"]
    # Use run's datasets for dropdown; if none, show config list so user can switch run
    datasets = list(datasets_in_run) if datasets_in_run else list(all_config_datasets)
    run_primary = metadata.get("dataset_key") or metadata.get("dataset") or (datasets_in_run[0] if datasets_in_run else None) or (datasets[0] if datasets else None)
    # Default selected_dataset_key to this run's primary when run matches or selection invalid
    if st.session_state.get("selected_run_id") == run_id:
        if "selected_dataset_key" not in st.session_state or st.session_state["selected_dataset_key"] not in datasets:
            st.session_state["selected_dataset_key"] = run_primary if run_primary in datasets else (datasets[0] if datasets else "diabetes")
    _ds_index = datasets.index(st.session_state["selected_dataset_key"]) if st.session_state["selected_dataset_key"] in datasets else 0

    def _on_dataset_change():
        if "dataset_selectbox" not in st.session_state:
            return
        new_ds = st.session_state["dataset_selectbox"]
        st.session_state["selected_dataset_key"] = new_ds
        # When dataset changes, use a run that contains this dataset (avoid cross-dataset artifacts)
        if new_ds not in datasets_in_run:
            runs_with_ds = get_available_runs(dataset_filter=new_ds)
            if runs_with_ds:
                st.session_state["selected_run_id"] = runs_with_ds[0]

    selected_dataset = st.sidebar.selectbox(
        "Select Dataset",
        datasets,
        index=_ds_index,
        key="dataset_selectbox",
        on_change=_on_dataset_change,
    )
    st.session_state["selected_dataset_key"] = selected_dataset
    ds_label = (selected_dataset or "").replace("_", " ").title()
    in_this_run = selected_dataset in datasets_in_run
    if not in_this_run and datasets_in_run:
        st.sidebar.warning(f"This run does not contain **{ds_label}**. Switch to another run or run the pipeline for this dataset.")
    st.sidebar.caption("Dataset in this run" if run_primary else "Dataset")
    run_primary_label = (run_primary or selected_dataset or "").replace("_", " ").title()
    st.sidebar.markdown(
        f'<div style="display:inline-block; padding:0.2rem 0.5rem; border-radius:999px; background:#ebf8ff; border:1px solid #3182ce; font-size:0.78rem; color:#2d3748; margin-bottom:0.35rem;">Dataset in this run: <strong>{run_primary_label}</strong></div>',
        unsafe_allow_html=True,
    )
    # Load feedback: research_outputs (preferred) then multi_agent/reports then reports
    feedback_data = None
    for feedback_path in [
        run_dir / "multi_agent" / "research_outputs" / "feedback.json",
        run_dir / "multi_agent" / "reports" / "feedback.json",
        run_dir / "research_outputs" / "feedback.json",
        run_dir / "reports" / "feedback.json",
    ]:
        if feedback_path.exists():
            try:
                with open(feedback_path, "r") as f:
                    feedback_data = json.load(f)
                if feedback_data:
                    break
            except Exception:
                pass
    feedback_data = feedback_data or {}
    feedback_decision = (feedback_data.get("decision") or "").strip() or None
    feedback_mode = feedback_decision or "—"
    retraining_performed = "Yes" if (feedback_data.get("retrained") or feedback_data.get("retraining") or feedback_data.get("retraining_performed")) else "No"
    best_model_after_feedback = (feedback_data.get("selected_model_after_feedback") or "").strip() or None
    trigger_metric = feedback_data.get("trigger_metric_name")
    trigger_value = feedback_data.get("trigger_metric_value")
    threshold = feedback_data.get("threshold")
    st.sidebar.caption("Feedback (trigger_metric, decision, retrained)")
    st.sidebar.markdown(
        f'<div style="display:inline-block; padding:0.2rem 0.5rem; border-radius:999px; background:#f7fafc; border:1px solid #e2e8f0; font-size:0.78rem; color:#2d3748; margin-bottom:0.35rem;">Feedback: <strong>{feedback_mode}</strong></div>',
        unsafe_allow_html=True,
    )
    if trigger_metric is not None and trigger_value is not None:
        try:
            tv = f"{float(trigger_value):.4f}"
        except (TypeError, ValueError):
            tv = str(trigger_value)
        th = f" (threshold={threshold})" if threshold is not None else ""
        st.sidebar.caption(f"Trigger: {trigger_metric}={tv}{th}")
    st.sidebar.markdown(
        f'<div style="display:inline-block; padding:0.2rem 0.5rem; border-radius:999px; background:#f7fafc; border:1px solid #e2e8f0; font-size:0.78rem; color:#2d3748; margin-bottom:0.5rem;">Retraining: <strong>{retraining_performed}</strong></div>',
        unsafe_allow_html=True,
    )

    # View artifacts for model – persist with selected_model_key; stable option values (model names)
    full_model_list = ["logistic_regression", "random_forest", "xgboost", "svm", "gradient_boosting", "knn"]
    models_dir = resolve_paths(run_dir)["models_dir"]
    available_models = []
    if models_dir.exists():
        run_models = [f.stem for f in models_dir.iterdir() if f.suffix == ".pkl"]
        for m in full_model_list:
            if m in run_models:
                available_models.append(m)
        for m in run_models:
            if m not in available_models:
                available_models.append(m)
    if not available_models:
        available_models = full_model_list.copy()
    for m in full_model_list:
        if m not in available_models:
            available_models.append(m)
    if "selected_model_key" not in st.session_state or st.session_state["selected_model_key"] not in available_models:
        st.session_state["selected_model_key"] = available_models[0]
    _model_index = available_models.index(st.session_state["selected_model_key"]) if st.session_state["selected_model_key"] in available_models else 0

    def _on_model_change():
        if "model_selectbox" in st.session_state:
            st.session_state["selected_model_key"] = st.session_state["model_selectbox"]

    _model_help = (
        "Best model is selected by ROC-AUC on the validation set. You can view artifacts for any model; "
        "if it is not the best, the header shows 'Viewing: X (not best)'."
    )
    selected_model = st.sidebar.selectbox(
        "View artifacts for model",
        available_models,
        index=_model_index,
        key="model_selectbox",
        on_change=_on_model_change,
        help=_model_help,
    )
    st.session_state["selected_model_key"] = selected_model
    st.sidebar.markdown("---")

    st.sidebar.checkbox("Defense Mode", value=False, key="defense_mode", help="Placeholder for defense mode.")
    st.sidebar.markdown("")

    if st.sidebar.button("📦 Run Pack", use_container_width=True):
        st.sidebar.info("Run pack export can be wired here (e.g., zip of run directory).")
    st.sidebar.markdown("---")

    # Best model: prefer "best model after feedback" when FeedbackAgent ran, else from model agent
    best_model = best_model_after_feedback if best_model_after_feedback else get_best_model_for_run(selected_run_id, selected_dataset)
    return selected_run_id, selected_dataset, selected_model, best_model, feedback_mode, retraining_performed
