"""
Explicit pipeline runners: baseline and multi-agent.

Ensures:
- Separate result objects (no shared references).
- No reading cached artifacts from the same run; each pipeline writes to its own subdir.
Multi-agent pipeline: Data → Model → Explainability → Evaluation → FeedbackAgent.
If feedback.action != "accept", trigger Iteration 2 and store iteration_1 / iteration_2; set retraining_status=True.
"""

import copy
import json
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
from sklearn.metrics import brier_score_loss
from sklearn.calibration import CalibratedClassifierCV

from utils.config import PipelineConfig, get_config
from utils.json_utils import json_safe
from utils.logging import setup_root_logger
from pipeline.orchestrator import execute_pipeline
from agents.model_agent import ModelAgent
from agents.explainability_agent import ExplainabilityAgent
from baseline.single_model_pipeline import SingleModelPipeline


def _apply_multi_agent_paths(config: PipelineConfig, base_run_dir: Path) -> Dict[str, Any]:
    """Point config to base_run_dir/multi_agent; create subdirs (research_outputs + dashboard_outputs). Returns saved originals for restore."""
    base_run_dir = Path(base_run_dir)
    ma_dir = base_run_dir / 'multi_agent'
    ma_dir.mkdir(parents=True, exist_ok=True)
    for sub in ('logs', 'models', 'explainability', 'figures', 'reports', 'data', 'research_outputs', 'dashboard_outputs'):
        (ma_dir / sub).mkdir(parents=True, exist_ok=True)

    saved = {
        'run_dir': config.run_dir,
        'log_dir': config.log_dir,
        'models_dir': config.model.models_dir,
        'explanations_dir': config.explainability.explanations_dir,
        'plots_dir': config.explainability.plots_dir,
        'results_dir': config.evaluation.results_dir,
        'dashboard_outputs_dir': getattr(config, 'dashboard_outputs_dir', None),
        'cross_dataset_outputs_dir': getattr(config, 'cross_dataset_outputs_dir', None),
    }
    config.run_dir = ma_dir
    config.log_dir = ma_dir / 'logs'
    config.model.models_dir = ma_dir / 'models'
    config.explainability.explanations_dir = ma_dir / 'explainability'
    config.explainability.plots_dir = ma_dir / 'figures'
    config.evaluation.results_dir = ma_dir / 'research_outputs'
    if hasattr(config, 'dashboard_outputs_dir'):
        config.dashboard_outputs_dir = ma_dir / 'dashboard_outputs'
    return saved


def _restore_paths(config: PipelineConfig, saved: Dict[str, Any]) -> None:
    """Restore config paths from saved originals."""
    config.run_dir = saved['run_dir']
    config.log_dir = saved['log_dir']
    config.model.models_dir = saved['models_dir']
    config.explainability.explanations_dir = saved['explanations_dir']
    config.explainability.plots_dir = saved['plots_dir']
    config.evaluation.results_dir = saved['results_dir']
    if saved.get('dashboard_outputs_dir') is not None and hasattr(config, 'dashboard_outputs_dir'):
        config.dashboard_outputs_dir = saved['dashboard_outputs_dir']
    if saved.get('cross_dataset_outputs_dir') is not None and hasattr(config, 'cross_dataset_outputs_dir'):
        config.cross_dataset_outputs_dir = saved['cross_dataset_outputs_dir']


def _compute_ece(y_true, y_proba, n_bins: int = 10) -> float:
    """Expected Calibration Error: average |accuracy - confidence| over bins."""
    try:
        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)
        if len(y_true) == 0 or len(y_proba) == 0:
            return 0.0
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            in_bin = (y_proba >= bin_edges[i]) & (y_proba < bin_edges[i + 1])
            if i == n_bins - 1:
                in_bin = (y_proba >= bin_edges[i]) & (y_proba <= bin_edges[i + 1])
            if not np.any(in_bin):
                continue
            acc = np.mean(y_true[in_bin])
            conf = np.mean(y_proba[in_bin])
            ece += np.sum(in_bin) * np.abs(acc - conf)
        return float(ece / len(y_true))
    except Exception:
        return 0.0


def _run_in_pipeline_evaluation(data_results: Dict, model_results: Dict) -> Dict[str, Any]:
    """Compute eval_results (roc_auc, ece, brier_score) for FeedbackAgent from current model_results and data."""
    eval_results = {}
    try:
        data = data_results
        models = model_results.get('models') or {}
        sel_name = model_results.get('selected_model') or model_results.get('best_model_name')
        selected_metrics = model_results.get('selected_metrics') or {}
        if sel_name and sel_name in models:
            target_col = data.get('target_column')
            val_df = data.get('val_df')
            if val_df is not None and target_col is not None:
                X_val = val_df.drop(columns=[target_col])
                y_val = val_df[target_col]
                model = models[sel_name]
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_val)[:, 1]
                    if len(np.unique(y_val)) >= 2:
                        eval_results['brier_score'] = float(brier_score_loss(y_val, proba))
                    eval_results['ece'] = _compute_ece(y_val, proba)
        if 'roc_auc' not in eval_results and isinstance(selected_metrics, dict):
            r = selected_metrics.get('roc_auc')
            if r is not None:
                eval_results['roc_auc'] = float(r)
    except Exception:
        pass
    return eval_results


def _apply_calibration_to_selected(model_results: Dict, data_results: Dict, method: str = "isotonic") -> Dict:
    """Apply probability calibration to the selected model; return updated model_results (copy)."""
    out = copy.deepcopy(model_results)
    models = out.get('models') or {}
    sel_name = out.get('selected_model') or out.get('best_model_name')
    if not sel_name or sel_name not in models:
        return out
    try:
        target_col = data_results.get('target_column')
        val_df = data_results.get('val_df')
        if val_df is None or target_col is None:
            return out
        X_val = val_df.drop(columns=[target_col])
        y_val = val_df[target_col]
        base = models[sel_name]
        calibrated = CalibratedClassifierCV(base, method=method, cv='prefit')
        calibrated.fit(X_val, y_val)
        out['models'] = dict(models)
        out['models'][sel_name] = calibrated
        # Recompute selected_metrics for calibrated model on val
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        y_pred = calibrated.predict(X_val)
        y_proba = calibrated.predict_proba(X_val)[:, 1]
        out['selected_metrics'] = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_val, y_pred, average='binary', zero_division=0),
            'f1_score': f1_score(y_val, y_pred, average='binary', zero_division=0),
            'roc_auc': roc_auc_score(y_val, y_proba) if len(np.unique(y_val)) > 1 else 0.0,
        }
    except Exception:
        pass
    return out


def run_baseline_pipeline(config: PipelineConfig, base_run_dir: Path) -> Dict[str, Any]:
    """
    Run the baseline pipeline: Data → Model → Explainability → Evaluation.
    Does NOT use FeedbackAgent and does NOT run CrossDatasetAgent.
    Uses config.data so --dataset from main() is respected.

    Writes only to base_run_dir/baseline/. Does not read from multi_agent or shared run.
    Returns a separate result dictionary (own models and metrics).
    """
    base_run_dir = Path(base_run_dir)
    setup_root_logger(config.log_dir)
    pipeline = SingleModelPipeline(config=config)
    results = pipeline.execute_standalone(base_run_dir)
    return results


def run_multi_agent_pipeline(
    config: PipelineConfig,
    base_run_dir: Path,
    baseline_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run the multi-agent pipeline: Data → Model → Explainability → Evaluation → Feedback Agent.
    If feedback.action != "accept", trigger Iteration 2 and store iteration_1, iteration_2, feedback_decision.

    Writes only to base_run_dir/multi_agent/. Optionally passes baseline_results to Model Agent for selection reasoning.
    Returns result with iteration_1, iteration_2 (if action != accept), feedback_decision, retraining_status, iteration_count.
    """
    base_run_dir = Path(base_run_dir)
    saved = _apply_multi_agent_paths(config, base_run_dir)
    # So CrossDatasetAgent can write to <base_run_dir>/research_outputs and <base_run_dir>/multi_agent/research_outputs
    setattr(config, "base_run_dir", base_run_dir)
    if baseline_results is not None:
        setattr(
            config,
            "baseline_for_model_agent",
            {
                "selected_metrics": baseline_results.get("selected_metrics"),
                "best_model_name": baseline_results.get("best_model_name"),
            },
        )
    try:
        # Central orchestrator: DataAgent → ModelAgent → ExplainabilityAgent → EvaluationAgent → FeedbackAgent → CrossDatasetAgent (if enabled)
        multi_agent_results = execute_pipeline(config)
        data_results = multi_agent_results.get("data") or {}
        feedback_dict = multi_agent_results.get("feedback") or {}
        print("=== FEEDBACK AGENT EXECUTED ===", feedback_dict)
        feedback = feedback_dict

        # Store feedback in run artifacts and persist for dashboard
        artifacts = multi_agent_results.get('artifacts') or {}
        artifacts['feedback'] = feedback_dict
        try:
            feedback_path = config.evaluation.results_dir / 'feedback.json'
            with open(feedback_path, 'w') as f:
                json.dump(feedback_dict, f, indent=2, default=json_safe)
        except Exception:
            pass

        def _is_simple(x):
            return isinstance(x, (str, int, float, bool, type(None)))

        def _to_serializable(obj):
            if _is_simple(obj):
                return obj
            if isinstance(obj, dict):
                return {k: _to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_to_serializable(v) for v in obj]
            return str(type(obj))

        def snapshot_results(res: dict) -> dict:
            out = {
                "run_id": res.get("run_id"),
                "dataset": res.get("dataset"),
                "best_model": res.get("best_model") or res.get("selected_model"),
                "evaluation": _to_serializable(res.get("evaluation") or res.get("metrics") or {}),
                "artifacts": _to_serializable(res.get("artifacts", {})),
                "feedback": _to_serializable(res.get("feedback", {})),
            }
            # Include only serializable model metadata (no model objects) for iteration comparison
            models = res.get("models") or {}
            if models:
                out["models"] = _to_serializable({
                    "best_model_name": models.get("best_model_name"),
                    "selected_model": models.get("selected_model"),
                    "selected_metrics": models.get("selected_metrics"),
                    "feature_names": models.get("feature_names"),
                })
            return out

        iteration_1 = snapshot_results(multi_agent_results)
        retraining_status = feedback.get('retraining_performed', feedback.get('retraining', False))
        iteration_count = 2 if retraining_status else 1
        results = {
            **multi_agent_results,
            'artifacts': artifacts,
            'iteration_1': iteration_1,
            'feedback_decision': feedback_dict,
            'retraining_status': retraining_status,
            'iteration_count': iteration_count,
        }

        action = feedback.get('action') or feedback.get('decision') or 'keep_model'
        # Pipeline must not crash when feedback decides "none"
        if action in ('none', 'keep_model'):
            return results

        # Re-run path: produce iteration_2; set retraining flag and iteration_count
        results['retraining_status'] = feedback.get('retrained', feedback.get('retraining_performed', True))
        results['iteration_count'] = 2
        collab = multi_agent_results.get('collaboration_metrics', {})
        force_model = feedback.get('selected_model_after_feedback') if action == 'switch_model' else None

        if action == 'switch_model':
            model_agent = ModelAgent(config.model)
            iter_artifacts = {"data": data_results}
            model_results = model_agent.run(iter_artifacts, force_model=force_model)
            iter_artifacts["models"] = model_results
            explainability_agent = ExplainabilityAgent(config.explainability)
            explainability_results = explainability_agent.run(iter_artifacts)
            iteration_2 = {
                'data': data_results,
                'models': model_results,
                'explainability': explainability_results,
                'collaboration_metrics': collab,
                'artifacts': {'data': data_results, 'models': model_results, 'explainability': explainability_results},
            }
            results['iteration_2'] = iteration_2
            results['data'] = data_results
            results['models'] = model_results
            results['explainability'] = explainability_results
            results['artifacts'] = iteration_2['artifacts']
            return results

        if action in ('retrain_best_model', 'retrain_with_tuned_params', 'retrain_with_adjustment'):
            iter_artifacts = {"data": data_results}
            model_agent = ModelAgent(config.model)
            model_results = model_agent.run(iter_artifacts)
            iter_artifacts["models"] = model_results
            explainability_agent = ExplainabilityAgent(config.explainability)
            explainability_results = explainability_agent.run(iter_artifacts)
            iteration_2 = {
                'data': data_results,
                'models': model_results,
                'explainability': explainability_results,
                'collaboration_metrics': collab,
                'artifacts': {'data': data_results, 'models': model_results, 'explainability': explainability_results},
            }
            results['iteration_2'] = iteration_2
            results['data'] = data_results
            results['models'] = model_results
            results['explainability'] = explainability_results
            results['artifacts'] = iteration_2['artifacts']
            return results

        return results
    finally:
        _restore_paths(config, saved)
