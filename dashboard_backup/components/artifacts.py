"""
Single source of truth for run artifact loading.
All pages should use load_run_artifacts(run_dir) and never error on missing files.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import json

import pandas as pd


def safe_read_json(path: Optional[Path]) -> dict:
    """Read JSON from path. Return {} if path is None, missing, or on error. Never raise."""
    if path is None:
        return {}
    p = Path(path)
    if not p.exists() or not p.is_file():
        return {}
    try:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def safe_read_csv(path: Optional[Path]) -> Optional[pd.DataFrame]:
    """Read CSV from path. Return None if path is None, missing, or on error. Never raise."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def _resolve_paths(run_dir: Path) -> Dict[str, Path]:
    """Resolve artifact directories (multi_agent, baseline, research_outputs, legacy)."""
    run_dir = Path(run_dir)
    ma = run_dir / "multi_agent"
    bl = run_dir / "baseline"
    ro = run_dir / "research_outputs"

    def first_existing(candidates: List[Path]) -> Path:
        for p in candidates:
            if p.exists() and p.is_dir():
                return p
        return candidates[-1] if candidates else run_dir

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


# Canonical locations for run_metadata (first existing wins)
def _run_metadata_candidates(run_dir: Path) -> List[Path]:
    run_dir = Path(run_dir)
    return [
        run_dir / "multi_agent" / "research_outputs" / "run_metadata.json",
        run_dir / "multi_agent" / "reports" / "run_metadata.json",
        run_dir / "reports" / "run_metadata.json",
        run_dir / "run_metadata.json",
        run_dir / "research_outputs" / "run_metadata.json",
    ]


def _data_metadata_candidate_dirs(run_dir: Path, paths: Dict[str, Path]) -> List[Path]:
    run_dir = Path(run_dir)
    return [
        paths["reports_dir"],
        run_dir / "research_outputs",
        paths["data_dir"],
        run_dir / "multi_agent" / "reports",
        run_dir / "multi_agent" / "research_outputs",
        run_dir / "multi_agent" / "data",
    ]


_DATA_METADATA_KEYS = {
    "n_rows", "n_cols", "schema", "missing", "class_distribution", "feature_summary",
    "cleaned_shape", "target_column", "raw_columns", "missing_values_after",
    "shape", "columns", "target", "features", "missing_values",
    "raw_rows", "train_rows", "val_rows", "test_rows", "data_quality_assessment",
    "agent_decisions", "artifacts_sent", "dataset_name",
}


def _find_data_metadata(paths: Dict[str, Path], run_dir: Path, dataset: Optional[str]) -> tuple:
    """Return (metadata dict, path or None, available_datasets list)."""
    run_dir = Path(run_dir)
    dirs = _data_metadata_candidate_dirs(run_dir, paths)
    available = []
    for d in dirs:
        if not d.exists():
            continue
        for f in d.glob("data_metadata_*.json"):
            if f.is_file():
                name = f.stem.replace("data_metadata_", "")
                if name and name not in available:
                    available.append(name)
    if not available and paths["reports_dir"].exists():
        for f in paths["reports_dir"].glob("*.json"):
            if f.is_file() and "data_metadata" in f.name.lower():
                name = f.stem.replace("data_metadata_", "").replace("data_metadata", "").strip("_")
                if name and name not in available:
                    available.append(name)
                elif not name and "data_metadata" in f.name.lower():
                    available.append("default")

    # Prefer exact data_metadata_{dataset}.json
    if dataset:
        exact = f"data_metadata_{dataset}.json"
        for d in dirs:
            if not d.exists():
                continue
            p = d / exact
            if p.exists() and p.is_file():
                return safe_read_json(p), p, available
    # Plain data_metadata.json
    for d in dirs:
        if not d.exists():
            continue
        p = d / "data_metadata.json"
        if p.exists() and p.is_file():
            return safe_read_json(p), p, available
    # Any JSON with data-metadata-like keys
    for d in dirs:
        if not d.exists():
            continue
        for f in sorted(d.glob("*.json")):
            if not f.is_file():
                continue
            data = safe_read_json(f)
            if data and _DATA_METADATA_KEYS & set(data.keys()):
                return data, f, available
    return {}, None, available


def _first_existing(paths_list: List[Path]) -> Optional[Path]:
    for p in paths_list:
        if p and Path(p).exists() and Path(p).is_file():
            return Path(p)
    return None


def load_run_artifacts(run_dir: Path, dataset: Optional[str] = None) -> Dict[str, Any]:
    """
    Load all run artifacts from canonical locations. Never raises; missing sections return empty dicts.

    Returns dict with keys:
      run_metadata: dict
      data: { metadata, metadata_path, data_dir, available_datasets, train_path, val_path, test_path }
      models: { metrics, metrics_path, models_dir, available_models, validation_predictions_path }
      evaluation: { feedback, comparison_report, multi_agent_results, baseline_results, collaboration }
      explainability: { explanations, figures_dir }
      baseline: { baseline_results }
      cross_dataset: { report, metrics, predictions_path, report_path, metrics_path }
      files: { paths: dict of dir name -> Path }
      health: { run_metadata, data, models, evaluation, explainability, baseline, cross_dataset } (bool per key)
    """
    run_dir = Path(run_dir)
    paths = _resolve_paths(run_dir)
    out = {
        "run_metadata": {},
        "data": {},
        "models": {},
        "evaluation": {},
        "explainability": {},
        "baseline": {},
        "cross_dataset": {},
        "files": {"paths": {k: str(v) for k, v in paths.items()}},
        "health": {},
    }

    # Run metadata
    for p in _run_metadata_candidates(run_dir):
        if p.exists() and p.is_file():
            out["run_metadata"] = safe_read_json(p)
            if "run_id" not in out["run_metadata"]:
                out["run_metadata"]["run_id"] = run_dir.name
            out["health"]["run_metadata"] = True
            break
    else:
        out["run_metadata"] = {"run_id": run_dir.name, "status": "unknown", "datasets": []}
        out["health"]["run_metadata"] = False

    # Data: metadata + data_dir + available_datasets
    data_meta, data_meta_path, available_datasets = _find_data_metadata(paths, run_dir, dataset)
    data_dir = paths["data_dir"]
    train_path = val_path = test_path = None
    if dataset:
        train_path = data_dir / f"{dataset}_train.csv" if (data_dir / f"{dataset}_train.csv").exists() else None
        val_path = data_dir / f"{dataset}_val.csv" if (data_dir / f"{dataset}_val.csv").exists() else None
        test_path = data_dir / f"{dataset}_test.csv" if (data_dir / f"{dataset}_test.csv").exists() else None
    if not available_datasets and out["run_metadata"].get("datasets"):
        available_datasets = list(out["run_metadata"]["datasets"])
    if not available_datasets:
        for d in [data_dir, run_dir / "multi_agent" / "data", run_dir / "data"]:
            if d.exists():
                for f in d.glob("*_train.csv"):
                    ds = f.stem.replace("_train", "")
                    if ds and ds not in available_datasets:
                        available_datasets.append(ds)
    # data_profile.json: columns, dtypes, feature_stats (min/max/median) for Prediction Explainer
    profile_candidates = []
    for base in [paths["reports_dir"], run_dir / "research_outputs", run_dir / "multi_agent" / "reports", run_dir / "multi_agent" / "research_outputs"]:
        if base.exists():
            if dataset:
                profile_candidates.append(base / f"data_profile_{dataset}.json")
            profile_candidates.append(base / "data_profile.json")
    data_profile = {}
    for p in profile_candidates:
        if p.exists() and p.is_file():
            data_profile = safe_read_json(p)
            if data_profile:
                break
    # Derive profile from data_meta if no data_profile file (e.g. older runs)
    if not data_profile and data_meta:
        cols = data_meta.get("raw_columns") or data_meta.get("columns") or []
        target_col = data_meta.get("target_column", "target")
        cols = [c for c in cols if c != target_col] if isinstance(cols, list) else []
        data_profile = {
            "columns": cols,
            "target_column": target_col,
            "feature_stats": data_meta.get("feature_stats") or {},
            "dtypes": data_meta.get("dtypes") or {},
        }

    # class_distribution_{dataset}.json: labels + counts for Data page
    class_distribution = {}
    for base in [paths["reports_dir"], run_dir / "research_outputs", run_dir / "multi_agent" / "reports", run_dir / "multi_agent" / "research_outputs"]:
        if base.exists():
            for name in ([f"class_distribution_{dataset}.json"] if dataset else []) + ["class_distribution.json"]:
                p = base / name
                if p.exists() and p.is_file():
                    class_distribution = safe_read_json(p)
                    if class_distribution and (class_distribution.get("labels") or class_distribution.get("counts")):
                        break
        if class_distribution:
            break
    if not class_distribution and data_meta and data_meta.get("class_distribution"):
        cd = data_meta["class_distribution"]
        if isinstance(cd, dict) and (cd.get("labels") or cd.get("counts")):
            class_distribution = cd
        elif isinstance(cd, list):
            class_distribution = {"labels": [str(x) for x in cd], "counts": []}

    out["data"] = {
        "metadata": data_meta,
        "metadata_path": str(data_meta_path) if data_meta_path else None,
        "data_dir": str(data_dir),
        "available_datasets": available_datasets,
        "train_path": str(train_path) if train_path else None,
        "val_path": str(val_path) if val_path else None,
        "test_path": str(test_path) if test_path else None,
        "profile": data_profile,
        "class_distribution": class_distribution,
    }
    out["health"]["data"] = bool(data_meta or available_datasets or data_profile)

    # Models: model_metrics.json, .pkl list, validation_predictions
    metrics_candidates = [
        paths["models_dir"] / "model_metrics.json",
        paths["reports_dir"] / "model_metrics.json",
        paths["models_dir"] / f"model_metrics_{dataset}.json" if dataset else None,
        paths["reports_dir"] / f"model_metrics_{dataset}.json" if dataset else None,
    ]
    metrics_path = _first_existing([p for p in metrics_candidates if p])
    model_metrics = safe_read_json(metrics_path)
    available_models = []
    if isinstance(model_metrics, dict):
        available_models = [k for k in model_metrics.keys() if isinstance(model_metrics.get(k), dict)]
    if not available_models and paths["models_dir"].exists():
        available_models = [f.stem for f in paths["models_dir"].iterdir() if f.suffix == ".pkl"]
    pred_candidates = [
        paths["models_dir"] / "validation_predictions.csv",
        paths["reports_dir"] / "validation_predictions.csv",
        paths["reports_dir"] / f"validation_predictions_{dataset}.csv" if dataset else None,
    ]
    pred_path = _first_existing([p for p in pred_candidates if p])
    out["models"] = {
        "metrics": model_metrics,
        "metrics_path": str(metrics_path) if metrics_path else None,
        "models_dir": str(paths["models_dir"]),
        "available_models": available_models,
        "validation_predictions_path": str(pred_path) if pred_path else None,
    }
    out["health"]["models"] = bool(model_metrics or available_models)

    # Evaluation: feedback, comparison, multi_agent_results, baseline_results, collaboration
    feedback_candidates = [
        run_dir / "multi_agent" / "research_outputs" / "feedback.json",
        run_dir / "multi_agent" / "reports" / "feedback.json",
        run_dir / "research_outputs" / "feedback.json",
        run_dir / "reports" / "feedback.json",
    ]
    feedback_path = _first_existing(feedback_candidates)
    comparison_path = _first_existing([paths["reports_dir"] / "comparison_report.csv", paths["reports_dir"] / f"comparison_report_{dataset}.csv" if dataset else None])
    ma_path = _first_existing([paths["reports_dir"] / "multi_agent_results.json", run_dir / "research_outputs" / "multi_agent_results.json", run_dir / "multi_agent" / "research_outputs" / "multi_agent_results.json"])
    bl_path = _first_existing([paths["reports_dir"] / "baseline_results.json", run_dir / "research_outputs" / "baseline_results.json", run_dir / "multi_agent" / "research_outputs" / "baseline_results.json"])
    collab_path = _first_existing([paths["reports_dir"] / "collaboration_evaluation.json"])
    baseline_comparison_path = _first_existing([paths["reports_dir"] / "baseline_comparison.json", run_dir / "research_outputs" / "baseline_comparison.json", run_dir / "multi_agent" / "research_outputs" / "baseline_comparison.json"])
    out["evaluation"] = {
        "feedback": safe_read_json(feedback_path),
        "comparison_path": str(comparison_path) if comparison_path else None,
        "multi_agent_results": safe_read_json(ma_path),
        "baseline_results": safe_read_json(bl_path),
        "collaboration": safe_read_json(collab_path),
        "baseline_comparison": safe_read_json(baseline_comparison_path) if baseline_comparison_path else {},
    }
    out["health"]["evaluation"] = bool(out["evaluation"]["feedback"] or out["evaluation"]["multi_agent_results"] or out["evaluation"]["baseline_results"])

    # Baseline (alias for evaluation.baseline_results)
    out["baseline"] = {"baseline_results": out["evaluation"].get("baseline_results", {})}
    out["health"]["baseline"] = bool(out["baseline"]["baseline_results"])

    # Explainability
    expl_path = _first_existing([paths["explainability_dir"] / "explanations.json", paths["reports_dir"] / "explanations.json"])
    out["explainability"] = {
        "explanations": safe_read_json(expl_path),
        "figures_dir": str(paths["figures_dir"]),
    }
    out["health"]["explainability"] = bool(out["explainability"]["explanations"] or (paths["figures_dir"].exists() and any(paths["figures_dir"].iterdir())))

    # Cross-dataset
    cross_bases = [paths["reports_dir"], run_dir / "research_outputs", run_dir / "multi_agent" / "research_outputs", run_dir / "multi_agent" / "reports"]
    report_path = _first_existing([b / "cross_dataset_report.json" for b in cross_bases if b.exists()])
    metrics_cross_path = _first_existing([b / "cross_dataset_metrics.json" for b in cross_bases if b.exists()])
    preds_cross_path = _first_existing([b / "cross_dataset_predictions.csv" for b in cross_bases if b.exists()])
    out["cross_dataset"] = {
        "report": safe_read_json(report_path),
        "metrics": safe_read_json(metrics_cross_path),
        "report_path": str(report_path) if report_path else None,
        "metrics_path": str(metrics_cross_path) if metrics_cross_path else None,
        "predictions_path": str(preds_cross_path) if preds_cross_path else None,
    }
    out["health"]["cross_dataset"] = bool(out["cross_dataset"]["report"] or out["cross_dataset"]["metrics"] or preds_cross_path)

    return out


def get_data_metadata_for_dataset(artifacts: Dict[str, Any], dataset: str) -> tuple:
    """
    Return (metadata dict, data_dir Path) for the given dataset from loaded artifacts.
    If the requested dataset is not in available_datasets, returns (first available metadata, data_dir) and caller should warn.
    """
    data = artifacts.get("data") or {}
    available = data.get("available_datasets") or []
    metadata = data.get("metadata") or {}
    data_dir_str = data.get("data_dir")
    data_dir = Path(data_dir_str) if data_dir_str else None
    if dataset in available or metadata:
        return metadata, data_dir
    return metadata, data_dir


def get_run_health_summary(artifacts: Dict[str, Any]) -> Dict[str, bool]:
    """Return the health dict (key -> exists) for Run health panel."""
    return artifacts.get("health") or {}


def resolve_dataset_model_fallback(artifacts: Dict[str, Any], dataset: str, model: str) -> tuple:
    """
    Return (resolved_dataset, resolved_model, warning_message or None).
    If dataset/model don't exist in run, fallback to first available and return a warning string.
    """
    data = artifacts.get("data") or {}
    models = artifacts.get("models") or {}
    available_ds = data.get("available_datasets") or []
    available_models = models.get("available_models") or []
    warning = None
    resolved_ds = dataset
    resolved_model = model
    if available_ds and dataset not in available_ds:
        resolved_ds = available_ds[0]
        warning = f"Dataset **{dataset}** not found for this run; showing **{resolved_ds}**."
    if available_models and model not in available_models:
        resolved_model = available_models[0]
        if not warning:
            warning = f"Model **{model}** not found for this run; showing **{resolved_model}**."
        else:
            warning += f" Model **{model}** not found; showing **{resolved_model}**."
    return resolved_ds, resolved_model, warning
