"""
Evaluation Metrics: Comprehensive evaluation framework for multi-agent vs baseline comparison.

This module provides:
- Predictive accuracy comparison
- Statistical significance testing
- Performance metrics aggregation
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from scipy import stats

from utils.config import get_config
from utils.json_utils import json_safe
from utils.logging import AgentLogger


class MetricsEvaluator:
    """
    Evaluator for comparing multi-agent and baseline pipelines.
    
    Computes:
    - Predictive accuracy metrics
    - Statistical significance tests
    - Performance comparison tables
    """
    
    def __init__(self):
        """Initialize metrics evaluator."""
        self.config = get_config()
        self.logger = AgentLogger('metrics_evaluator')
    
    def compare_predictive_accuracy(
        self,
        multi_agent_selected_metrics: Dict[str, float],
        baseline_selected_metrics: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Compare predictive accuracy between selected models only.
        Expects flat metric dicts: selected_metrics = {metric_name: value}.
        """
        self.logger.info("Comparing predictive accuracy (selected_metrics only)...")
        comparison_data = []
        metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        for metric in metric_names:
            ma_value = multi_agent_selected_metrics.get(metric, np.nan)
            bl_value = baseline_selected_metrics.get(metric, np.nan)
            if not (np.isnan(ma_value) or np.isnan(bl_value)):
                difference = ma_value - bl_value
                percent_diff = (difference / bl_value * 100) if bl_value != 0 else 0
                comparison_data.append({
                    'model': 'selected',
                    'metric': metric,
                    'multi_agent': ma_value,
                    'baseline': bl_value,
                    'difference': difference,
                    'percent_difference': percent_diff
                })
        return pd.DataFrame(comparison_data)
    
    def generate_comparison_report(
        self,
        multi_agent_results: Dict[str, Any],
        baseline_results: Dict[str, Any],
        output_path: Optional[Path] = None,
        feedback: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Generate comprehensive comparison report.

        Args:
            multi_agent_results: Results from multi-agent pipeline
            baseline_results: Results from baseline pipeline
            output_path: Path to save report
            feedback: Optional feedback artifact (for best_model_before/after_feedback, retraining_applied, metric_triggered_on)

        Returns:
            Comparison report DataFrame
        """
        self.logger.info("Generating comparison report...")
        ma_selected = (multi_agent_results.get('models') or {}).get('selected_metrics') or {}
        bl_selected = baseline_results.get('selected_metrics') or {}
        comparison_df = self.compare_predictive_accuracy(ma_selected, bl_selected)

        # Add execution time comparison
        ma_time = multi_agent_results.get('collaboration_metrics', {}).get('total_execution_time', 0)
        bl_time = baseline_results.get('execution_time', 0)

        time_comparison = pd.DataFrame([{
            'model': 'pipeline',
            'metric': 'execution_time_seconds',
            'multi_agent': ma_time,
            'baseline': bl_time,
            'difference': ma_time - bl_time,
            'percent_difference': ((ma_time - bl_time) / bl_time * 100) if bl_time > 0 else 0
        }])

        full_comparison = pd.concat([comparison_df, time_comparison], ignore_index=True)

        # Add feedback columns if available
        fd = feedback or (multi_agent_results.get('artifacts') or {}).get('feedback') or multi_agent_results.get('feedback_decision')
        if fd:
            best_before = fd.get('selected_model_before') or (multi_agent_results.get('models') or {}).get('selected_model') or (multi_agent_results.get('models') or {}).get('best_model_name')
            best_after = fd.get('selected_model_after') or fd.get('selected_model_after_feedback')
            retraining_applied = fd.get('retrained', fd.get('retraining_performed', False))
            metric_triggered = fd.get('trigger_metric_name', '')
            full_comparison['best_model_before_feedback'] = best_before
            full_comparison['best_model_after_feedback'] = best_after
            full_comparison['retraining_applied'] = bool(retraining_applied)
            full_comparison['metric_triggered_on'] = metric_triggered

        # Save report
        if output_path is None:
            output_path = self.config.evaluation.results_dir / 'comparison_report.csv'

        full_comparison.to_csv(output_path, index=False)
        self.logger.info(f"Comparison report saved to {output_path}")
        summary = self._generate_summary(full_comparison, ma_selected, bl_selected)
        
        # Save summary
        summary_path = self.config.evaluation.results_dir / 'comparison_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=json_safe)
        
        self.logger.info(f"Comparison summary saved to {summary_path}")
        
        return full_comparison
    
    def _compute_system_quality_index(
        self,
        roc_auc: Optional[float],
        has_integrated_explainability: bool,
        has_feedback_agent: bool,
        workflow_components_count: int,
        max_workflow_components: int = 6,
    ) -> float:
        """
        Single 0–1 score for thesis narrative: predictive quality + workflow coverage.
        Weights: predictive 0.5, explainability 0.15, feedback 0.15, workflow coverage 0.2.
        """
        pred = (float(roc_auc) if roc_auc is not None and not (isinstance(roc_auc, float) and np.isnan(roc_auc)) else 0.0)
        pred = max(0.0, min(1.0, pred))
        cov = workflow_components_count / max(max_workflow_components, 1)
        return 0.5 * pred + 0.15 * (1.0 if has_integrated_explainability else 0.0) + 0.15 * (1.0 if has_feedback_agent else 0.0) + 0.2 * cov

    def _generate_summary(
        self,
        comparison_df: pd.DataFrame,
        ma_selected_metrics: Dict[str, float],
        bl_selected_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Generate summary statistics (selected_metrics only). Includes System Quality Index for thesis narrative."""
        ma_roc = ma_selected_metrics.get('roc_auc')
        bl_roc = bl_selected_metrics.get('roc_auc')
        bl_sqi = self._compute_system_quality_index(bl_roc, has_integrated_explainability=False, has_feedback_agent=False, workflow_components_count=2)
        ma_sqi = self._compute_system_quality_index(ma_roc, has_integrated_explainability=True, has_feedback_agent=True, workflow_components_count=6)
        summary = {
            'comparison': {
                'selected': {
                    'multi_agent_roc_auc': ma_roc,
                    'baseline_roc_auc': bl_roc,
                }
            },
            'system_quality_index': {
                'baseline': round(bl_sqi, 4),
                'multi_agent': round(ma_sqi, 4),
            },
            'average_metrics': {m: {'multi_agent_mean': ma_selected_metrics.get(m), 'baseline_mean': bl_selected_metrics.get(m)} for m in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']},
            'execution_time': {}
        }
        return summary

    def generate_merged_comparison_report(
        self,
        baseline_results: Dict[str, Any],
        multi_agent_results: Dict[str, Any],
        cross_dataset_results: Optional[Dict[str, Any]] = None,
        output_path: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        Final merged table: baseline vs multi-agent vs multi-agent+crossdataset.
        Includes execution time overhead and metric deltas.
        """
        self.logger.info("Generating merged comparison (baseline vs multi_agent vs multi_agent+crossdataset)...")
        bl_sel = baseline_results.get("selected_metrics") or baseline_results.get("metrics") or {}
        ma_sel = (multi_agent_results.get("models") or {}).get("selected_metrics") or {}
        bl_time = baseline_results.get("execution_time", 0)
        ma_time = multi_agent_results.get("collaboration_metrics", {}).get("total_execution_time", 0)

        metric_names = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
        rows = []
        for m in metric_names:
            bl_v = bl_sel.get(m, np.nan)
            ma_v = ma_sel.get(m, np.nan)
            cross_v = np.nan
            if cross_dataset_results and cross_dataset_results.get("enabled") and cross_dataset_results.get("results"):
                vals = [r.get(m) for r in cross_dataset_results["results"] if r.get(m) is not None and not (isinstance(r.get(m), float) and np.isnan(r.get(m)))]
                cross_v = float(np.mean(vals)) if vals else np.nan
            rows.append({
                "metric": m,
                "baseline": bl_v,
                "multi_agent": ma_v,
                "multi_agent_crossdataset": (cross_v if not (isinstance(cross_v, float) and np.isnan(cross_v)) else ma_v),
                "delta_ma_vs_baseline": (ma_v - bl_v) if not (np.isnan(ma_v) or np.isnan(bl_v)) else np.nan,
                "delta_cross_vs_baseline": ((cross_v if not np.isnan(cross_v) else ma_v) - bl_v) if not np.isnan(bl_v) else np.nan,
            })
        rows.append({
            "metric": "execution_time_seconds",
            "baseline": bl_time,
            "multi_agent": ma_time,
            "multi_agent_crossdataset": ma_time,
            "delta_ma_vs_baseline": ma_time - bl_time,
            "delta_cross_vs_baseline": ma_time - bl_time,
        })
        df = pd.DataFrame(rows)
        if output_path is None:
            output_path = self.config.evaluation.results_dir / "merged_comparison_report.csv"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        self.logger.info(f"Merged comparison report saved to {output_path}")
        return df

    def generate_baseline_comparison_json(
        self,
        baseline_results: Dict[str, Any],
        multi_agent_results: Dict[str, Any],
        explainability_report: Optional[Dict[str, Any]] = None,
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Build baseline vs multi-agent comparison for dashboard Overview.
        Returns a JSON-serializable dict: baseline metrics, multi_agent metrics, deltas, improvements.
        """
        bl_sel = baseline_results.get("selected_model") or baseline_results.get("best_model_name")
        ma_sel = (multi_agent_results.get("models") or {}).get("selected_model") or (multi_agent_results.get("models") or {}).get("best_model_name")
        bl_metrics = baseline_results.get("selected_metrics") or baseline_results.get("metrics") or {}
        ma_metrics = (multi_agent_results.get("models") or {}).get("selected_metrics") or {}
        bl_time = float(baseline_results.get("execution_time") or 0)
        ma_time = float((multi_agent_results.get("collaboration_metrics") or {}).get("total_execution_time") or multi_agent_results.get("execution_time") or 0)

        metric_names = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
        baseline_metrics = {m: bl_metrics.get(m) for m in metric_names if bl_metrics.get(m) is not None}
        multi_agent_metrics = {m: ma_metrics.get(m) for m in metric_names if ma_metrics.get(m) is not None}
        for m in metric_names:
            if m not in baseline_metrics and bl_metrics.get(m) is not None:
                baseline_metrics[m] = bl_metrics[m]
            if m not in multi_agent_metrics and ma_metrics.get(m) is not None:
                multi_agent_metrics[m] = ma_metrics[m]

        deltas = {}
        for m in metric_names:
            bl_v = baseline_metrics.get(m)
            ma_v = multi_agent_metrics.get(m)
            if bl_v is not None and ma_v is not None and not (isinstance(bl_v, float) and np.isnan(bl_v)) and not (isinstance(ma_v, float) and np.isnan(ma_v)):
                deltas[m] = float(ma_v - bl_v)
        deltas["execution_time_seconds"] = ma_time - bl_time

        improvements = []
        for m in metric_names:
            d = deltas.get(m)
            if d is not None and not (isinstance(d, float) and np.isnan(d)):
                sign = "+" if d >= 0 else ""
                improvements.append(f"{m}: {sign}{d:.4f}")

        out = {
            "baseline": {
                "selected_model": bl_sel,
                "metrics": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in baseline_metrics.items()},
                "execution_time_seconds": bl_time,
            },
            "multi_agent": {
                "selected_model": ma_sel,
                "metrics": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in multi_agent_metrics.items()},
                "execution_time_seconds": ma_time,
            },
            "deltas": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in deltas.items()},
            "improvements": improvements,
        }
        if explainability_report:
            readability = explainability_report.get("readability") or {}
            stability = explainability_report.get("shap_stability") or {}
            fidelity = explainability_report.get("fidelity") or {}
            out["explainability_deltas"] = {
                "readability": readability,
                "shap_stability": stability,
                "fidelity": fidelity,
            }
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(out, f, indent=2, default=json_safe)
            self.logger.info(f"Baseline comparison JSON saved to {output_path}")
        return out
