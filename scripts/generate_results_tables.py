#!/usr/bin/env python3
"""
Generate thesis-style results tables (Baseline vs Multi-Agent) for Diabetes and Heart Disease
from the latest pipeline runs. Output: RESULTS_TABLES.md in project root.
Run from project root: python scripts/generate_results_tables.py
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "outputs" / "runs"
METRIC_ORDER = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1_score": "F1 Score",
    "roc_auc": "ROC-AUC",
}


def _get_dataset_from_run(run_dir: Path):
    for p in [run_dir / "run_metadata.json", run_dir / "reports" / "run_metadata.json"]:
        if p.exists():
            try:
                with open(p) as f:
                    meta = json.load(f)
                key = meta.get("dataset_key") or meta.get("dataset") or (meta.get("datasets") or [None])[0]
                return str(key).strip().lower() if key else None
            except Exception:
                pass
    return None


def _get_comparison_csv(run_dir: Path):
    for sub in ["research_outputs", "multi_agent/research_outputs", "reports"]:
        p = run_dir / sub / "comparison_report.csv"
        if p.exists():
            return p
    return None


def _load_metrics(csv_path: Path) -> dict[str, tuple[float, float]]:
    """Return dict metric_name -> (baseline_value, multi_agent_value)."""
    import csv
    out = {}
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("model") != "selected":
                continue
            m = row.get("metric", "").strip().lower()
            if m not in METRIC_ORDER:
                continue
            try:
                bl = float(row.get("baseline", 0))
                ma = float(row.get("multi_agent", 0))
                out[m] = (bl, ma)
            except (TypeError, ValueError):
                pass
    return out


def _format_val(v: float, metric: str) -> str:
    if metric == "roc_auc":
        return f"{v:.4f}"
    return f"{v:.4f}"


def _build_table(dataset_label: str, metrics: dict[str, tuple[float, float]]) -> str:
    lines = [
        f"## {dataset_label} Dataset",
        "",
        "| Metric | Baseline Pipeline | Multi-Agent Pipeline |",
        "|--------|-------------------|----------------------|",
    ]
    for m in METRIC_ORDER:
        label = METRIC_LABELS.get(m, m.replace("_", " ").title())
        if m in metrics:
            bl, ma = metrics[m]
            lines.append(f"| {label} | {_format_val(bl, m)} | {_format_val(ma, m)} |")
        else:
            lines.append(f"| {label} | — | — |")
    lines.append("")
    return "\n".join(lines)


def main():
    if not RUNS_DIR.exists():
        print(f"Run directory not found: {RUNS_DIR}")
        return

    # Collect latest run per dataset
    runs_by_dataset: dict[str, Path] = {}
    for run_dir in sorted(RUNS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
            continue
        ds = _get_dataset_from_run(run_dir)
        if ds and ds not in runs_by_dataset:
            csv_path = _get_comparison_csv(run_dir)
            if csv_path:
                runs_by_dataset[ds] = run_dir

    # Only Diabetes and Heart Disease for thesis tables
    dataset_order = ["diabetes", "heart_disease"]
    runs_by_dataset = {k: v for k, v in runs_by_dataset.items() if k in dataset_order}

    sections = [
        "# Pipeline comparison: Baseline vs Multi-Agent",
        "",
        "Tables below are generated from the latest pipeline runs. Re-run `python scripts/generate_results_tables.py` after new runs to refresh.",
        "",
    ]

    for ds in dataset_order:
        if ds not in runs_by_dataset:
            label = ds.replace("_", " ").title()
            sections.append(_build_table(label, {}))
            continue
        run_dir = runs_by_dataset[ds]
        csv_path = _get_comparison_csv(run_dir)
        metrics = _load_metrics(csv_path) if csv_path else {}
        label = ds.replace("_", " ").title()
        sections.append(_build_table(label, metrics))

    out_path = PROJECT_ROOT / "RESULTS_TABLES.md"
    out_path.write_text("\n".join(sections), encoding="utf-8")
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
