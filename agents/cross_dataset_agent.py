"""
Cross-Dataset Agent: Evaluate primary trained models on a second (cross) dataset without retraining.

Runs after FeedbackAgent when config.cross_dataset.enabled is True.
Guarantees writing <run_dir>/research_outputs/cross_dataset_report.json and cross_dataset_summary.txt.
run_dir is taken from artifacts["run_dir"] (set by orchestrator).
"""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

from utils.config import DataConfig, get_config
from utils.json_utils import json_safe
from utils.logging import AgentLogger
from utils.feature_schema import canonicalize_features, CANONICAL_FEATURES
from agents.data_agent import DataAgent


MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost", "svm", "gradient_boosting", "knn"]
MIN_COMMON = 3  # Minimum canonical feature overlap; below this report is marked low_confidence


def _align_features(
    X: pd.DataFrame,
    train_feature_names: List[str],
    target_col: str,
) -> Optional[pd.DataFrame]:
    """Intersect common columns, drop target, align to train order. Returns None if no common features."""
    want = [c for c in train_feature_names if c != target_col]
    available = [c for c in want if c in X.columns]
    if not available:
        return None
    return X[available].copy()


def _build_aligned_target_matrix(
    target_X: pd.DataFrame,
    train_feature_names: List[str],
    model: Any,
    target_col: str,
) -> pd.DataFrame:
    """Build target feature matrix with same columns as training; fill missing with 0. Use model.feature_names_in_ if available."""
    if hasattr(model, "feature_names_in_") and model.feature_names_in_ is not None:
        train_features = list(model.feature_names_in_)
    else:
        train_features = [c for c in train_feature_names if c != target_col]
    if not train_features:
        return pd.DataFrame(index=target_X.index)
    X_aligned = pd.DataFrame(0.0, index=target_X.index, columns=train_features)
    for col in train_features:
        if col in target_X.columns:
            X_aligned[col] = target_X[col].values
    return X_aligned


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
) -> Dict[str, Optional[float]]:
    """Accuracy, F1, ROC-AUC (if binary), precision, recall. ROC-AUC is None when not computable."""
    out: Dict[str, Optional[float]] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="binary", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="binary", zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
    }
    if y_proba is not None and len(np.unique(y_true)) >= 2:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            out["roc_auc"] = None
    else:
        out["roc_auc"] = None
    return out


def _load_models_from_artifacts(
    artifacts: Dict[str, Any],
    models_dir: Optional[Path],
    logger: Any,
) -> Dict[str, Any]:
    """Load trained models from artifacts['models']['models'] or from models_dir (pickle)."""
    models = {}
    models_art = artifacts.get("models") or {}
    in_memory = (models_art.get("models") or {}).copy()

    for name in MODEL_NAMES:
        if name in in_memory:
            models[name] = in_memory[name]
            continue
        if models_dir:
            pkl = Path(models_dir) / f"{name}.pkl"
            if pkl.exists():
                try:
                    with open(pkl, "rb") as f:
                        models[name] = pickle.load(f)
                except Exception as e:
                    logger.warning(f"CrossDatasetAgent: could not load {name} from {pkl}: {e}")
        if name not in models and models_art.get("file_paths", {}).get("models"):
            for path in (models_art["file_paths"]["models"] or {}).values():
                path = Path(path) if path else None
                if path and path.name == f"{name}.pkl" and path.exists():
                    try:
                        with open(path, "rb") as f:
                            models[name] = pickle.load(f)
                        break
                    except Exception as e:
                        logger.warning(f"CrossDatasetAgent: could not load from {path}: {e}")
    return models


class CrossDatasetAgent:
    """
    Cross-dataset validation: run all primary trained models on a second dataset (target_dataset).
    Produces per-model metrics and generalization_drop (primary validation vs cross-dataset).
    """

    def __init__(self, config: Optional[Any] = None, pipeline_config: Optional[Any] = None, logger=None):
        self.config = config
        self.pipeline_config = pipeline_config or get_config()
        self.logger = logger or AgentLogger("cross_dataset_agent")

    def run(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run cross-dataset evaluation. Guarantees writing report and summary to
        <run_dir>/research_outputs/ using artifacts["run_dir"]. On any exception
        writes status "failed" and error message to the same files.
        """
        cross_cfg = self.config or getattr(self.pipeline_config, "cross_dataset", None)
        if not cross_cfg or not getattr(cross_cfg, "enabled", False):
            self.logger.info("CrossDatasetAgent: disabled (config.cross_dataset.enabled=False).")
            return {"enabled": False, "report": None, "summary": "Cross-dataset validation disabled."}

        # Use active pipeline run_dir only; do not create new run_* folders
        run_dir = artifacts.get("run_dir")
        if run_dir is None:
            run_dir = getattr(self.pipeline_config, "run_dir", None)
        run_dir = Path(run_dir) if run_dir else Path(".")
        run_dir.mkdir(parents=True, exist_ok=True)
        research_outputs = run_dir / "research_outputs"
        research_outputs.mkdir(parents=True, exist_ok=True)
        path1 = research_outputs / "cross_dataset_report.json"
        base_run_dir = artifacts.get("base_run_dir")
        if base_run_dir is None:
            base_run_dir = run_dir.parent if run_dir.name == "multi_agent" else run_dir
        base_run_dir = Path(base_run_dir)
        base_ro = base_run_dir / "research_outputs"
        base_ro.mkdir(parents=True, exist_ok=True)
        path2 = base_ro / "cross_dataset_report.json"
        path3_dir = base_run_dir / "multi_agent" / "research_outputs"
        path3_dir.mkdir(parents=True, exist_ok=True)
        path3 = path3_dir / "cross_dataset_report.json"

        _datasets = getattr(self.pipeline_config.data, "datasets", None)
        _first_ds = _datasets[0] if (_datasets is not None and len(_datasets) > 0) else None
        source_dataset = (
            getattr(self.pipeline_config.data, "dataset_name", None)
            or _first_ds
            or getattr(self.pipeline_config, "dataset", None)
        )
        target_dataset = getattr(cross_cfg, "target_dataset", None) or "heart_disease"
        eval_split = getattr(cross_cfg, "eval_split", "test") or "test"
        if not isinstance(target_dataset, str) or not target_dataset.strip():
            target_dataset = "heart_disease"
        target_dataset = target_dataset.strip()

        def _write_guaranteed(report: Dict[str, Any], summary_lines: List[str]) -> None:
            for p in (path1, path2, path3):
                try:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    with open(p, "w") as f:
                        json.dump(report, f, indent=2, default=json_safe)
                    self.logger.info(f"Cross-dataset report saved to: {str(p)}")
                except Exception as e:
                    self.logger.warning(f"CrossDatasetAgent: could not write report to {p}: {e}")
            for d in (research_outputs, base_ro, path3_dir):
                try:
                    with open(d / "cross_dataset_summary.txt", "w") as f:
                        f.write("\n".join(summary_lines))
                except Exception as e:
                    self.logger.warning(f"CrossDatasetAgent: could not write summary to {d}: {e}")

        try:
            data_art = artifacts.get("data") or {}
            train_df = data_art.get("train_df")
            target_col = data_art.get("target_column")
            if train_df is None or not target_col:
                self.logger.warning("CrossDatasetAgent: missing artifacts data/train_df or target_column.")
                report = {
                    "source_dataset": source_dataset,
                    "target_dataset": target_dataset,
                    "eval_split": eval_split,
                    "model_used": None,
                    "metrics": {},
                    "n_samples": None,
                    "timestamp": datetime.now().isoformat(),
                    "status": "failed",
                    "error": "Missing train data or target_column",
                }
                _write_guaranteed(report, [f"Cross-dataset: {report['status']}. {report['error']}"])
                return {"enabled": True, "report": report, "results": [], "summary": report["error"]}

            primary_dataset = (
                (data_art.get("metadata") or {}).get("dataset_name")
                or (_first_ds if _first_ds is not None else getattr(self.pipeline_config.data, "dataset_name", None))
            )
            train_feature_names = list(train_df.columns)
            if target_col in train_feature_names:
                train_feature_names = [c for c in train_feature_names if c != target_col]

            models_dir = getattr(self.pipeline_config.model, "models_dir", None)
            if not models_dir and (artifacts.get("models") or {}).get("file_paths", {}).get("models"):
                first_path = next(iter((artifacts["models"]["file_paths"]["models"] or {}).values()), None)
                if first_path:
                    models_dir = Path(first_path).parent
            models = _load_models_from_artifacts(artifacts, models_dir, self.logger)
            if not models:
                self.logger.warning("CrossDatasetAgent: no trained models found in artifacts or models_dir.")
                report = {
                    "source_dataset": source_dataset or primary_dataset,
                    "target_dataset": target_dataset,
                    "eval_split": eval_split,
                    "model_used": None,
                    "metrics": {},
                    "n_samples": None,
                    "timestamp": datetime.now().isoformat(),
                    "status": "failed",
                    "error": "No trained models found",
                }
                _write_guaranteed(report, [f"Cross-dataset: {report['status']}. {report['error']}"])
                return {"enabled": True, "report": report, "results": [], "summary": report["error"]}

            primary_metrics = (artifacts.get("models") or {}).get("all_models") or {}
            if not primary_metrics and (artifacts.get("models") or {}).get("selected_metrics"):
                best_name = (artifacts["models"].get("selected_model") or artifacts["models"].get("best_model_name")) or ""
                if best_name:
                    primary_metrics = {best_name: artifacts["models"]["selected_metrics"]}

            if target_dataset == primary_dataset:
                report = {
                    "source_dataset": primary_dataset,
                    "target_dataset": target_dataset,
                    "eval_split": eval_split,
                    "model_used": None,
                    "metrics": {},
                    "n_samples": None,
                    "timestamp": datetime.now().isoformat(),
                    "status": "failed",
                    "error": "Target same as source",
                }
                _write_guaranteed(report, [f"Cross-dataset: skipped (target=sources). {report['error']}"])
                return {"enabled": True, "report": report, "results": [], "summary": report["error"]}

            self.logger.info(f"CrossDatasetAgent: loading target dataset '{target_dataset}' (eval_split={eval_split}).")
            try:
                data_config = DataConfig()
                data_config.datasets = [target_dataset]
                data_config.dataset_name = target_dataset
                data_agent = DataAgent(data_config)
                target_data = data_agent.process()
            except Exception as e:
                self.logger.warning(f"CrossDatasetAgent: DataAgent failed for {target_dataset}: {e}")
                report = {
                    "source_dataset": source_dataset or primary_dataset,
                    "target_dataset": target_dataset,
                    "eval_split": eval_split,
                    "model_used": None,
                    "metrics": {},
                    "n_samples": None,
                    "timestamp": datetime.now().isoformat(),
                    "status": "failed",
                    "error": f"Preprocessing failed: {e}",
                }
                _write_guaranteed(report, [f"Cross-dataset: {report['status']}. {report['error']}"])
                return {"enabled": True, "report": report, "results": [], "summary": report["error"]}

            split_df = target_data.get(f"{eval_split}_df")
            if split_df is None:
                split_df = target_data.get("test_df")
            if split_df is None:
                split_df = target_data.get("val_df")
            if split_df is None:
                split_df = target_data.get("train_df")

            if split_df is None:
                available = list(target_data.keys()) if hasattr(target_data, "keys") else []
                msg = (
                    f"No split data for target dataset '{target_dataset}' (eval_split={eval_split}). Available keys: {available}"
                )
                self.logger.error(msg)
                report = {
                    "source_dataset": source_dataset or primary_dataset,
                    "target_dataset": target_dataset,
                    "eval_split": eval_split,
                    "model_used": None,
                    "metrics": {},
                    "n_samples": None,
                    "timestamp": datetime.now().isoformat(),
                    "status": "failed",
                    "error": msg,
                }
                _write_guaranteed(report, [f"Cross-dataset: {report['status']}. {report['error']}"])
                return {"enabled": True, "report": report, "results": [], "summary": msg}

            tcol = target_data.get("target_column", None) if target_data.get("target_column") is not None else target_col
            if isinstance(split_df, (tuple, list)) and len(split_df) == 2:
                X_cross = split_df[0]
                y_cross = np.asarray(split_df[1])
                if not isinstance(X_cross, pd.DataFrame):
                    X_cross = pd.DataFrame(X_cross)
            else:
                if tcol not in split_df.columns:
                    self.logger.warning("CrossDatasetAgent: target column not in split.")
                    report = {
                        "source_dataset": source_dataset or primary_dataset,
                        "target_dataset": target_dataset,
                        "eval_split": eval_split,
                        "model_used": None,
                        "metrics": {},
                        "n_samples": None,
                        "timestamp": datetime.now().isoformat(),
                        "status": "failed",
                        "error": "Target column missing in split",
                    }
                    _write_guaranteed(report, [f"Cross-dataset: {report['status']}. {report['error']}"])
                    return {"enabled": True, "report": report, "results": [], "summary": report["error"]}
                X_cross = split_df.drop(columns=[tcol])
                y_cross = np.asarray(split_df[tcol])

            n_samples = len(y_cross)
            results_list = []
            metrics_flat: Dict[str, Dict[str, float]] = {}

            # Canonical feature schema: align train and target to common names for meaningful cross-dataset overlap
            train_dataset_name = (primary_dataset or "").strip().lower()
            target_dataset_name = (target_dataset or "").strip().lower()
            df_train_canon = pd.DataFrame()
            df_target_canon = pd.DataFrame()
            train_canon_cols: List[str] = []
            target_canon_cols: List[str] = []
            try:
                df_train_canon, train_canon_cols, _ = canonicalize_features(train_df, train_dataset_name, target_column=target_col)
                target_combined = X_cross.copy()
                target_combined["_target"] = y_cross
                df_target_canon, target_canon_cols, _ = canonicalize_features(target_combined, target_dataset_name, target_column="_target")
            except Exception as e:
                self.logger.warning(f"CrossDatasetAgent: canonicalization failed: {e}")
            common_canon = sorted(set(train_canon_cols) & set(target_canon_cols)) if train_canon_cols and target_canon_cols else []
            low_confidence = len(common_canon) < MIN_COMMON
            canonical_metrics: Optional[Dict[str, Any]] = None
            canonical_y_pred = None
            canonical_y_proba = None
            y_target_canon: Optional[np.ndarray] = None
            if common_canon and len(df_train_canon) > 0 and len(df_target_canon) > 0 and target_col in df_train_canon.columns:
                try:
                    X_train_canon = df_train_canon[common_canon]
                    y_train_canon = np.asarray(df_train_canon[target_col])
                    X_target_canon = df_target_canon[common_canon]
                    y_target_canon = np.asarray(df_target_canon["_target"])
                    ref_model = LogisticRegression(max_iter=500, random_state=42)
                    ref_model.fit(X_train_canon, y_train_canon)
                    canonical_y_pred = ref_model.predict(X_target_canon)
                    canonical_y_proba = ref_model.predict_proba(X_target_canon)[:, 1] if hasattr(ref_model, "predict_proba") else None
                    canonical_metrics = _compute_metrics(y_target_canon, canonical_y_pred, canonical_y_proba)
                except Exception as e:
                    self.logger.warning(f"CrossDatasetAgent: canonical reference model failed: {e}")

            raw_overlap_count = len([c for c in train_feature_names if c != tcol and c in X_cross.columns])

            for model_name, model in models.items():
                X_aligned = _build_aligned_target_matrix(X_cross, train_feature_names, model, tcol)
                if X_aligned is None or len(X_aligned.columns) == 0:
                    self.logger.warning(
                        f"CrossDatasetAgent: no features for model={model_name}; skipping."
                    )
                    continue
                try:
                    y_pred = model.predict(X_aligned)
                    y_proba = model.predict_proba(X_aligned)[:, 1] if hasattr(model, "predict_proba") else None
                except Exception as e:
                    self.logger.warning(f"CrossDatasetAgent: prediction failed for {model_name}: {e}")
                    continue

                cross_metrics = _compute_metrics(y_cross, y_pred, y_proba)
                primary_m = primary_metrics.get(model_name) or {}
                if isinstance(primary_m, dict):
                    gen_drop = {
                        "accuracy": (primary_m.get("accuracy") or 0) - cross_metrics["accuracy"],
                        "f1_score": (primary_m.get("f1_score") or 0) - cross_metrics["f1_score"],
                        "roc_auc": (primary_m.get("roc_auc") or 0) - (cross_metrics.get("roc_auc") or 0),
                        "precision": (primary_m.get("precision") or 0) - cross_metrics["precision"],
                        "recall": (primary_m.get("recall") or 0) - cross_metrics["recall"],
                    }
                else:
                    gen_drop = {}
                try:
                    cm = confusion_matrix(y_cross, y_pred)
                    confusion_matrix_list = cm.tolist()
                except Exception:
                    confusion_matrix_list = None

                results_list.append({
                    "model": model_name,
                    "cross_dataset_metrics": cross_metrics,
                    "primary_validation_metrics": primary_m if isinstance(primary_m, dict) else {},
                    "generalization_drop": gen_drop,
                    "confusion_matrix": confusion_matrix_list,
                })
                metrics_flat[model_name] = {**cross_metrics, "generalization_drop": gen_drop, "confusion_matrix": confusion_matrix_list}

            # When all models were skipped, still write report with raw_transfer and canonical_transfer
            if not results_list:
                raw_transfer_empty = {
                    "models_attempted": list(models.keys()) if models else [],
                    "models_succeeded": [],
                    "overlap_count": raw_overlap_count,
                    "results": [],
                }
                canonical_transfer_empty = {
                    "used_features": common_canon,
                    "overlap_count": len(common_canon),
                    "metrics": canonical_metrics,
                    "low_confidence": low_confidence,
                    "reason": "Limited canonical feature overlap" if low_confidence else "All raw models skipped",
                }
                canonical_report = {
                    "status": "no_compatible_models",
                    "reason": "All models skipped (feature alignment or prediction failed).",
                    "source_dataset": primary_dataset,
                    "target_dataset": target_dataset,
                    "eval_split": eval_split,
                    "raw_transfer": raw_transfer_empty,
                    "canonical_transfer": canonical_transfer_empty,
                    "evaluated_models": [],
                    "skipped_models": list(models.keys()) if models else [],
                    "n_samples": n_samples,
                    "timestamp": datetime.now().isoformat(),
                }
                summary_lines = [
                    f"Cross-dataset: source={primary_dataset}, target={target_dataset}, eval_split={eval_split}.",
                    f"Status: {canonical_report['status']}. {canonical_report['reason']}",
                    f"Raw overlap: {raw_overlap_count}. Canonical overlap: {len(common_canon)}.",
                    "Low-confidence: true (limited overlap)." if low_confidence else "",
                ]
                _write_guaranteed(canonical_report, summary_lines)
                self.logger.info(f"Cross-dataset report saved to: {str(path1)}")
                return {
                    "enabled": True,
                    "report": canonical_report,
                    "train_dataset": primary_dataset,
                    "target_dataset": target_dataset,
                    "results": [],
                    "report_paths": [str(path1), str(path2), str(path3)],
                    "summary": "No compatible models for cross-dataset evaluation.",
                }

            selected_model = (
                (artifacts.get("feedback") or {}).get("selected_model_after_feedback")
                or (artifacts.get("feedback") or {}).get("selected_model_after")
                or (artifacts.get("models") or {}).get("selected_model")
                or (artifacts.get("models") or {}).get("best_model_name")
            )
            threshold = None
            try:
                fb = getattr(self.pipeline_config, "feedback", None)
                if fb is not None:
                    threshold = getattr(fb, "threshold", None)
            except Exception:
                pass

            feedback_art = artifacts.get("feedback") or {}
            feedback_triggered = feedback_art.get("decision") not in (None, "none", "")
            action = (feedback_art.get("decision") or feedback_art.get("action") or "none").strip() or "none"
            primary_roc = None
            try:
                sel_metrics = (artifacts.get("models") or {}).get("selected_metrics") or {}
                primary_roc = sel_metrics.get("roc_auc")
            except Exception:
                pass
            cross_gen_score = None
            if results_list:
                rocs = [r.get("cross_dataset_metrics", {}).get("roc_auc") for r in results_list if isinstance(r.get("cross_dataset_metrics"), dict)]
                rocs = [x for x in rocs if x is not None]
                if rocs:
                    cross_gen_score = float(sum(rocs) / len(rocs))

            research_scorecard = {
                "roc_auc": primary_roc,
                "explainability_readability": None,
                "explainability_stability": None,
                "explanation_fidelity": None,
                "ERI": None,
                "feedback_triggered": feedback_triggered,
                "action": action,
                "cross_dataset_generalization_score": cross_gen_score,
            }

            raw_transfer = {
                "models_attempted": list(models.keys()),
                "models_succeeded": list(metrics_flat.keys()),
                "overlap_count": raw_overlap_count,
                "results": results_list,
            }
            canonical_transfer = {
                "used_features": common_canon,
                "overlap_count": len(common_canon),
                "metrics": canonical_metrics,
                "low_confidence": low_confidence,
                "reason": "Limited canonical feature overlap" if low_confidence else None,
            }
            best_canon_roc = (canonical_metrics.get("roc_auc") if canonical_metrics else None)
            best_canon_acc = (canonical_metrics.get("accuracy") if canonical_metrics else None)
            summary_str = (
                f"Cross-dataset: {primary_dataset} -> {target_dataset}, eval_split={eval_split}. "
                f"Raw overlap={raw_overlap_count}, canonical overlap={len(common_canon)}. "
                f"Best canonical: roc_auc={best_canon_roc}, accuracy={best_canon_acc}. "
                + ("Low-confidence (limited overlap)." if low_confidence else "")
            )

            report = {
                "train_dataset": primary_dataset,
                "target_dataset": target_dataset,
                "eval_split": eval_split,
                "selected_model": selected_model,
                "threshold": threshold,
                "models_evaluated": list(metrics_flat.keys()),
                "results": results_list,
                "raw_transfer": raw_transfer,
                "canonical_transfer": canonical_transfer,
                "summary": summary_str,
                "research_scorecard": research_scorecard,
                "enabled": True,
            }
            metrics_report = {
                "train_dataset": primary_dataset,
                "target_dataset": target_dataset,
                "eval_split": eval_split,
                "selected_model": selected_model,
                "threshold": threshold,
                "metrics": metrics_flat,
                "canonical_metrics": canonical_metrics,
                "raw_overlap_count": raw_overlap_count,
                "canonical_overlap_count": len(common_canon),
            }

            # Canonical report for guaranteed artifact (and orchestrator)
            sel_metrics = metrics_flat.get(selected_model) if selected_model in metrics_flat else (list(metrics_flat.values())[0] if metrics_flat else {})
            canonical_report = {
                "source_dataset": primary_dataset,
                "target_dataset": target_dataset,
                "eval_split": eval_split,
                "model_used": selected_model,
                "metrics": {
                    "roc_auc": sel_metrics.get("roc_auc"),
                    "accuracy": sel_metrics.get("accuracy"),
                    "precision": sel_metrics.get("precision"),
                    "recall": sel_metrics.get("recall"),
                    "f1": sel_metrics.get("f1_score"),
                },
                "n_samples": n_samples,
                "timestamp": datetime.now().isoformat(),
                "status": "success",
                "raw_transfer": raw_transfer,
                "canonical_transfer": canonical_transfer,
                "summary": summary_str,
            }
            summary_lines = [
                f"Cross-dataset: source={primary_dataset}, target={target_dataset}, eval_split={eval_split}.",
                f"Raw overlap count: {raw_overlap_count}. Canonical overlap count: {len(common_canon)}.",
                f"Best canonical roc_auc={best_canon_roc}, accuracy={best_canon_acc}.",
                "Low-confidence: true (limited overlap)." if low_confidence else "Low-confidence: false.",
                f"Model used (raw): {selected_model}. n_samples={n_samples}. status=success.",
            ]
            _write_guaranteed(canonical_report, summary_lines)

            # Save predictions CSV: prefer canonical (y_true, y_pred, y_proba) when available
            pred_df = None
            if canonical_y_pred is not None and y_target_canon is not None:
                try:
                    pred_df = pd.DataFrame({"y_true": y_target_canon, "y_pred": canonical_y_pred})
                    if canonical_y_proba is not None:
                        pred_df["y_proba"] = canonical_y_proba
                except Exception:
                    pred_df = None
            if pred_df is None and results_list and models:
                pred_model_name = selected_model if (selected_model and selected_model in models) else (list(models.keys())[0])
                try:
                    model = models[pred_model_name]
                    X_aligned = _build_aligned_target_matrix(X_cross, train_feature_names, model, tcol)
                    if X_aligned is not None and len(X_aligned.columns) > 0:
                        y_pred_out = model.predict(X_aligned)
                        y_proba_out = model.predict_proba(X_aligned)[:, 1] if hasattr(model, "predict_proba") else None
                        pred_df = pd.DataFrame({"y_true": y_cross, "y_pred": y_pred_out})
                        if y_proba_out is not None:
                            pred_df["y_proba"] = y_proba_out
                except Exception:
                    pred_df = None
            if pred_df is not None:
                try:
                    for d in [research_outputs, base_ro, path3_dir]:
                        d = Path(d)
                        d.mkdir(parents=True, exist_ok=True)
                        out_path = d / "cross_dataset_predictions.csv"
                        pred_df.to_csv(out_path, index=False)
                        self.logger.info(f"CrossDatasetAgent: wrote {out_path}")
                except Exception as e:
                    self.logger.warning(f"CrossDatasetAgent: could not write predictions CSV: {e}")

            report_paths = []
            metrics_paths = []
            summary_paths = []

            def _write_to(d: Path) -> None:
                d = Path(d)
                d.mkdir(parents=True, exist_ok=True)
                rp, mp, sp = d / "cross_dataset_report.json", d / "cross_dataset_metrics.json", d / "cross_dataset_summary.txt"
                try:
                    with open(rp, "w") as f:
                        json.dump(report, f, indent=2, default=json_safe)
                    report_paths.append(str(rp))
                except Exception as e:
                    self.logger.warning(f"CrossDatasetAgent: could not write report to {rp}: {e}")
                try:
                    with open(mp, "w") as f:
                        json.dump(metrics_report, f, indent=2, default=json_safe)
                    metrics_paths.append(str(mp))
                except Exception as e:
                    self.logger.warning(f"CrossDatasetAgent: could not write metrics to {mp}: {e}")
                try:
                    if metrics_flat:
                        best_cross = max(metrics_flat.items(), key=lambda x: (x[1].get("accuracy") or 0, (x[1].get("roc_auc") if x[1].get("roc_auc") is not None else 0)))
                        lines = [
                            f"Cross-dataset validation: train={primary_dataset}, target={target_dataset}.",
                            f"Models evaluated: {len(metrics_flat)}. Eval split: {eval_split}.",
                            f"Selected model (primary): {selected_model or '—'}. Threshold: {threshold}.",
                            f"Best on cross-dataset: {best_cross[0]} (accuracy={best_cross[1].get('accuracy', 0):.4f}, roc_auc={(best_cross[1].get('roc_auc') or 0):.4f}).",
                        ]
                    else:
                        lines = [
                            f"Cross-dataset validation: train={primary_dataset}, target={target_dataset}.",
                            f"Models evaluated: 0. Eval split: {eval_split}.",
                        ]
                    with open(sp, "w") as f:
                        f.write("\n".join(lines))
                    summary_paths.append(str(sp))
                except Exception as e:
                    self.logger.warning(f"CrossDatasetAgent: could not write summary to {sp}: {e}")

            _write_to(research_outputs)
            _write_to(base_ro)
            _write_to(path3_dir)
            _write_to(base_run_dir / "reports")

            for p in report_paths:
                self.logger.info(f"CrossDatasetAgent: report saved to {p}")
            for p in metrics_paths:
                self.logger.info(f"CrossDatasetAgent: metrics saved to {p}")
            for p in summary_paths:
                self.logger.info(f"CrossDatasetAgent: summary saved to {p}")
            if report_paths:
                self.logger.info(f"Cross-dataset report saved to: {', '.join(report_paths)}")
            self.logger.info(f"Cross-dataset report saved to: {str(path1)}")

            return {
                "enabled": True,
                "report": canonical_report,
                "train_dataset": primary_dataset,
                "target_dataset": target_dataset,
                "results": results_list,
                "report_paths": report_paths,
                "metrics_paths": metrics_paths,
                "summary_paths": summary_paths,
                "summary": f"Evaluated {len(results_list)} model(s) on target dataset '{target_dataset}'.",
            }

        except Exception as e:
            self.logger.warning(f"CrossDatasetAgent: run failed: {e}")
            report = {
                "source_dataset": source_dataset,
                "target_dataset": target_dataset,
                "eval_split": eval_split,
                "model_used": None,
                "metrics": {},
                "n_samples": None,
                "timestamp": datetime.now().isoformat(),
                "status": "failed",
                "error": str(e),
            }
            _write_guaranteed(report, [f"Cross-dataset: {report['status']}. {report['error']}"])
            return {"enabled": True, "report": report, "results": [], "summary": str(e)}