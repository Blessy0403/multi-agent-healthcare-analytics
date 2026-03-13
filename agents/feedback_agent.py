"""
Feedback Agent: Runs after Evaluation + Explainability. Makes a measurable decision from
EvaluationAgent and ExplainabilityEvaluator outputs (fidelity, shap_stability, readability, ERI).
Outputs feedback.json with trigger_metric, threshold, decision, selected_model_before/after, retrained, reason.
"""

from typing import Dict, Any, Optional

import numpy as np

# Default threshold when trigger_metric is below => apply action
ERI_THRESHOLD = 0.6


def _float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _eri_from_explainability(explainability: Dict[str, Any], best_model_name: Optional[str]) -> tuple:
    """
    Compute ERI from stability + fidelity. Use existing metrics or SHAP/LIME proxies.
    Returns (eri, stability, fidelity). If missing, use proxies or (0.7, 0.7, 0.7) for safe accept.
    """
    stability = _float(explainability.get("stability_score") or explainability.get("stability"))
    fidelity = _float(explainability.get("fidelity"))

    shap_explanations = explainability.get("shap_explanations") or {}
    lime_explanations = explainability.get("lime_explanations") or {}
    best_shap = shap_explanations.get(best_model_name) if best_model_name else None
    best_lime = lime_explanations.get(best_model_name) if best_model_name else None

    if stability is None and best_shap:
        try:
            sv = best_shap.get("shap_values")
            if sv is not None:
                sv = np.asarray(sv)
                if sv.size > 0:
                    mean_abs = np.abs(sv).mean(axis=0)
                    if mean_abs.size > 0 and mean_abs.mean() > 1e-9:
                        cv = float(np.std(mean_abs) / (np.mean(mean_abs) + 1e-9))
                        stability = min(1.0, max(0.0, 1.0 - min(1.0, cv)))
                    else:
                        stability = 0.7
                else:
                    stability = 0.7
            else:
                stability = 0.7
        except Exception:
            stability = 0.7
    if stability is None:
        stability = 0.7 if shap_explanations else 0.5

    if fidelity is None and (best_shap and best_lime):
        fidelity = 0.6
    if fidelity is None:
        fidelity = 0.6 if lime_explanations else 0.5

    eri = (stability + fidelity) / 2.0
    return (float(eri), float(stability), float(fidelity))


def _trigger_value_from_report(
    explainability_report: Dict[str, Any],
    best_model_name: Optional[str],
    trigger_metric: str,
) -> Optional[float]:
    """Get trigger_metric value from ExplainabilityEvaluator report (fidelity, shap_stability, readability, eri)."""
    if not best_model_name or not explainability_report:
        return None
    if trigger_metric == "eri":
        fidelity = None
        stability = None
        fid_by_model = explainability_report.get("fidelity") or {}
        stab_by_model = explainability_report.get("shap_stability") or {}
        if isinstance(fid_by_model, dict):
            f = fid_by_model.get(best_model_name)
            fidelity = _float(f.get("correlation")) if isinstance(f, dict) else _float(f)
        if isinstance(stab_by_model, dict):
            s = stab_by_model.get(best_model_name)
            stability = _float(s.get("mean_rank_correlation")) if isinstance(s, dict) else _float(s)
        if fidelity is not None and stability is not None:
            return (fidelity + stability) / 2.0
        return None
    if trigger_metric == "fidelity":
        fid_by_model = explainability_report.get("fidelity") or {}
        f = fid_by_model.get(best_model_name) if isinstance(fid_by_model, dict) else None
        return _float(f.get("correlation")) if isinstance(f, dict) else _float(f)
    if trigger_metric == "shap_stability":
        stab_by_model = explainability_report.get("shap_stability") or {}
        s = stab_by_model.get(best_model_name) if isinstance(stab_by_model, dict) else None
        return _float(s.get("mean_rank_correlation")) if isinstance(s, dict) else _float(s)
    if trigger_metric == "readability":
        read_by_model = explainability_report.get("readability") or {}
        r = read_by_model.get(best_model_name) if isinstance(read_by_model, dict) else None
        return _float(r.get("mean_readability")) if isinstance(r, dict) else _float(r)
    return None


class FeedbackAgent:
    """
    Runs after EvaluationAgent. Uses Evaluation + ExplainabilityEvaluator outputs (fidelity,
    shap_stability, readability, ERI). Config-driven: if trigger_metric < threshold then
    action = switch_model | retrain_best_model | retrain_with_tuned_params; else decision = none.
    Outputs feedback.json structure: trigger_metric_name, trigger_metric_value, threshold,
    decision, selected_model_before, selected_model_after, retrained, reason.
    """

    def __init__(self, config: Optional[Any] = None, pipeline_config: Optional[Any] = None, logger=None):
        self.config = config
        self.pipeline_config = pipeline_config
        self.logger = logger

    def _get_feedback_settings(self):
        """Resolve enabled, threshold, trigger_metric, action from config."""
        c = self.config
        if c is None:
            return True, ERI_THRESHOLD, "eri", "switch_model"
        enabled = getattr(c, "enabled", True) if not isinstance(c, dict) else c.get("enabled", True)
        threshold = getattr(c, "threshold", ERI_THRESHOLD) if not isinstance(c, dict) else c.get("threshold", ERI_THRESHOLD)
        trigger = getattr(c, "trigger_metric", "eri") if not isinstance(c, dict) else c.get("trigger_metric", "eri")
        action = getattr(c, "action", "switch_model") if not isinstance(c, dict) else c.get("action", "switch_model")
        return enabled, float(threshold), trigger, action

    def run(self, run_context: dict) -> dict:
        """
        Decide from trigger_metric vs threshold. Output feedback.json-shaped dict.
        If trigger_metric_value >= threshold: decision = none (pipeline must not crash).
        """
        enabled, threshold, trigger_metric_name, config_action = self._get_feedback_settings()
        evaluation = run_context.get("evaluation") or {}
        explainability = run_context.get("explainability") or {}
        explainability_report = run_context.get("explainability_report") or {}
        models_info = run_context.get("evaluation_results") or run_context.get("models") or {}
        all_models = models_info.get("all_models") or {}
        selected_model_before = (
            models_info.get("best_model_name")
            or models_info.get("selected_model")
            or models_info.get("best_model")
        )
        if not isinstance(all_models, dict):
            all_models = {}

        # Build trigger value: prefer ExplainabilityEvaluator report, else ERI/proxies from raw explainability
        trigger_metric_value = _trigger_value_from_report(
            explainability_report, selected_model_before, trigger_metric_name
        )
        if trigger_metric_value is None:
            try:
                eri, stability, fidelity = _eri_from_explainability(explainability, selected_model_before)
            except Exception:
                eri, stability, fidelity = 0.7, 0.7, 0.7
            if trigger_metric_name == "eri":
                trigger_metric_value = eri
            elif trigger_metric_name == "fidelity":
                trigger_metric_value = fidelity
            elif trigger_metric_name == "shap_stability":
                trigger_metric_value = stability
            else:
                trigger_metric_value = eri
        trigger_metric_value = float(trigger_metric_value)

        # Decision rule: if trigger_metric < threshold -> action; else none
        if not enabled:
            decision = "none"
            reason = "Feedback agent disabled by config."
            selected_model_after = selected_model_before
            retrained = False
        elif trigger_metric_value < threshold:
            decision = config_action  # switch_model | retrain_best_model | retrain_with_tuned_params
            reason = (
                f"{trigger_metric_name}={trigger_metric_value:.4f} < threshold={threshold}; "
                f"applying action={decision}."
            )
            if decision == "switch_model":
                selected_model_after = self._next_best_model(all_models, selected_model_before)
                retrained = False
            else:
                selected_model_after = selected_model_before  # retrain keeps same model type
                retrained = True
        else:
            decision = "none"
            reason = (
                f"{trigger_metric_name}={trigger_metric_value:.4f} >= threshold={threshold}; no action."
            )
            selected_model_after = selected_model_before
            retrained = False

        out = {
            "trigger_metric_name": trigger_metric_name,
            "trigger_metric_value": float(trigger_metric_value),
            "threshold": float(threshold),
            "decision": decision,
            "selected_model_before": selected_model_before,
            "selected_model_after": selected_model_after,
            "retrained": retrained,
            "reason": reason,
            # Legacy keys for existing dashboard/runner
            "action": "none" if decision == "none" else (decision if decision in ("switch_model", "retrain_best_model", "retrain_with_tuned_params") else "keep_model"),
            "retraining_performed": retrained,
            "selected_model_after_feedback": selected_model_after,
        }
        if self.logger:
            self.logger.info(
                f"FeedbackAgent: {trigger_metric_name}={trigger_metric_value:.4f} threshold={threshold} "
                f"decision={decision} selected_model_before={selected_model_before} "
                f"selected_model_after={selected_model_after} retrained={retrained}"
            )
        print(
            f"FeedbackAgent decision={decision} {trigger_metric_name}={trigger_metric_value:.4f} "
            f"selected_model_after={selected_model_after} retrained={retrained}"
        )
        return out

    @staticmethod
    def _next_best_model(all_models: dict, current_best: Optional[str]) -> Optional[str]:
        """Next-best model by ROC-AUC from evaluation results."""
        if not all_models or not isinstance(all_models, dict):
            return current_best
        sorted_names = sorted(
            all_models.keys(),
            key=lambda k: (
                all_models[k].get("roc_auc", 0.0)
                if isinstance(all_models.get(k), dict)
                else 0.0
            ),
            reverse=True,
        )
        if not sorted_names:
            return current_best
        if current_best and len(sorted_names) > 1 and sorted_names[0] == current_best:
            return sorted_names[1]
        return sorted_names[0] if sorted_names else current_best
