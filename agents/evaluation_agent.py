"""
Evaluation Agent: Computes in-pipeline evaluation metrics from data and model artifacts.

Runs after ExplainabilityAgent. Produces metrics (roc_auc, ece, brier_score) for FeedbackAgent.
"""

import numpy as np
from typing import Dict, Any, Optional
from sklearn.metrics import brier_score_loss

from utils.logging import AgentLogger


def _compute_ece(y_true, y_proba, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    try:
        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)
        if len(y_true) == 0 or len(y_proba) == 0:
            return 0.0
        ece = 0.0
        for i in range(n_bins):
            lo, hi = i / n_bins, (i + 1) / n_bins
            in_bin = (y_proba >= lo) & (y_proba <= hi if i == n_bins - 1 else y_proba < hi)
            if not np.any(in_bin):
                continue
            acc = np.mean(y_true[in_bin])
            conf = np.mean(y_proba[in_bin])
            ece += np.sum(in_bin) * np.abs(acc - conf)
        return float(ece / len(y_true))
    except Exception:
        return 0.0


class EvaluationAgent:
    """
    Agent that produces evaluation metrics from data and model artifacts.
    Returns structured dict: roc_auc, ece, brier_score, selected_metrics (for downstream FeedbackAgent).
    """

    def __init__(self, config: Optional[Any] = None, logger=None):
        self.config = config
        self.logger = logger or AgentLogger("evaluation_agent")

    def run(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute evaluation metrics from artifacts["features"] or artifacts["data"] and artifacts["models"].

        Args:
            artifacts: Must contain "data" or "features" (train_df, val_df, target_column) and "models" (models, selected_model/best_model_name, selected_metrics).

        Returns:
            Dict with roc_auc, ece, brier_score, and optionally selected_metrics (copy).
        """
        result = {}
        data = artifacts.get("features") or artifacts.get("data") or {}
        model_results = artifacts.get("models") or {}
        models = model_results.get("models") or {}
        sel_name = model_results.get("selected_model") or model_results.get("best_model_name")
        selected_metrics = model_results.get("selected_metrics") or {}

        if not sel_name or sel_name not in models:
            self.logger.info("EvaluationAgent: no selected model in artifacts; returning empty metrics.")
            return result

        target_col = data.get("target_column")
        val_df = data.get("val_df")
        if val_df is None or target_col is None:
            return result

        try:
            X_val = val_df.drop(columns=[target_col])
            y_val = val_df[target_col]
            model = models[sel_name]
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_val)[:, 1]
                if len(np.unique(y_val)) >= 2:
                    result["brier_score"] = float(brier_score_loss(y_val, proba))
                result["ece"] = _compute_ece(y_val, proba)
            if isinstance(selected_metrics, dict):
                result["roc_auc"] = selected_metrics.get("roc_auc")
                result["selected_metrics"] = dict(selected_metrics)
        except Exception as e:
            self.logger.warning(f"EvaluationAgent: computation failed: {e}")
        return result
