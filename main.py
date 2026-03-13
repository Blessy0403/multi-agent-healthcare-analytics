"""
Main Entry Point: Multi-Agent AI Pipeline for Explainable Healthcare Analytics

This script executes the complete pipeline via explicit runners:
1. run_baseline_pipeline(): Data → Model → Explainability → Evaluation (no Feedback Agent)
2. run_multi_agent_pipeline(): Data → Model → Explainability → Evaluation → Feedback Agent (retrain if triggered)
3. Compare the two result dictionaries and save reports

Usage:
    python main.py                    # default dataset (e.g. diabetes)
    python main.py framingham         # Framingham Heart Study
    python main.py diabetes           # Pima diabetes
    from main import run_pipeline; run_pipeline(dataset="framingham")
"""

import argparse
import sys
import traceback
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.config import get_config
from utils.logging import setup_root_logger
from utils.artifacts import ArtifactManager
from utils.seed import set_seed
from pipeline.runner import run_baseline_pipeline, run_multi_agent_pipeline
from evaluation.metrics import MetricsEvaluator
from evaluation.explainability_eval import ExplainabilityEvaluator
from evaluation.collaboration_eval import CollaborationEvaluator


def json_safe(obj):
    """Convert numpy/pandas and other non-JSON types for json.dump."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, (pd.Series, pd.Index)):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return str(obj)


def _write_run_log(run_dir: Path, message: str):
    """Append a message to this run's run.log (so logs are never empty on failure)."""
    log_file = Path(run_dir) / 'logs' / 'run.log'
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(message)
            f.flush()
    except Exception:
        pass


def _serialize_results_for_dashboard(baseline_results: dict, multi_agent_results: dict) -> tuple:
    """Return JSON-serializable dicts for baseline and multi-agent (no model objects)."""
    def to_serializable_metrics(m):
        if not m or not isinstance(m, dict):
            return m
        out = {}
        for k, v in m.items():
            if isinstance(v, dict):
                out[k] = {kk: (float(vv) if isinstance(vv, (int, float)) else vv) for kk, vv in v.items()}
            elif isinstance(v, (int, float)):
                out[k] = float(v)
            else:
                out[k] = v  # e.g. list for confusion_matrix
        return out

    def flat_metrics(d):
        if not d or not isinstance(d, dict):
            return d
        return {k: float(v) if isinstance(v, (int, float)) else v for k, v in d.items()}
    bl = {
        'selected_model': baseline_results.get('selected_model') or baseline_results.get('best_model_name'),
        'selected_metrics': flat_metrics(baseline_results.get('selected_metrics')),
        'all_models': to_serializable_metrics(baseline_results.get('all_models')),
        'execution_time': float(baseline_results.get('execution_time', 0)),
    }
    ma_models = multi_agent_results.get('models') or {}
    ma = {
        'models': {
            'selected_model': ma_models.get('selected_model') or ma_models.get('best_model_name'),
            'selected_metrics': flat_metrics(ma_models.get('selected_metrics')),
            'all_models': to_serializable_metrics(ma_models.get('all_models')),
        },
        'collaboration_metrics': {
            k: (float(v) if isinstance(v, (int, float)) else v)
            for k, v in (multi_agent_results.get('collaboration_metrics') or {}).items()
        },
    }
    ma['execution_time'] = ma['collaboration_metrics'].get('total_execution_time', 0)
    return bl, ma


def run_pipeline(dataset=None, cross_dataset_target=None):
    """Run the full pipeline. dataset: primary dataset; cross_dataset_target: enable cross-dataset validation on that target (e.g. heart_disease)."""
    return main(dataset=dataset, cross_dataset_target=cross_dataset_target)


def main(dataset=None, cross_dataset_target=None, no_feature_engineering=False):
    """Execute the complete multi-agent healthcare analytics pipeline.
    dataset: primary dataset (e.g. diabetes, heart_disease).
    no_feature_engineering: if True, disable FeatureEngineeringAgent.
    cross_dataset_target: if set, enables cross-dataset validation and uses this as target dataset (e.g. heart_disease)."""
    
    # Set random seed for reproducibility
    set_seed(42)
    
    config = get_config()
    config.enable_feature_engineering = not no_feature_engineering
    if dataset is not None:
        config.data.datasets = [dataset]
        config.data.dataset_name = dataset
        if hasattr(config, "dataset"):
            config.dataset = dataset

    # Cross-dataset: enable and set target dataset when --cross_dataset TARGET is provided
    if cross_dataset_target is not None and str(cross_dataset_target).strip():
        config.cross_dataset.enabled = True
        config.cross_dataset.target_dataset = str(cross_dataset_target).strip()
        config.cross_dataset_enabled = True
        config.enable_cross_dataset = True
    
    # Initialize artifact manager and run directory first
    artifact_manager = ArtifactManager(config.run_id)
    
    # Save run metadata (so run exists and has status=running from the start)
    primary_dataset = getattr(config, 'dataset', None) or (config.data.datasets[0] if config.data.datasets else None)
    run_metadata = {
        'run_id': config.run_id,
        'timestamp': datetime.now().isoformat(),
        'status': 'running',
        'datasets': list(config.data.datasets) if config.data.datasets else [],
        'dataset': primary_dataset,
        'dataset_key': primary_dataset,
        'enable_baseline': getattr(config, 'enable_baseline', True),
        'enable_feedback': getattr(config.feedback, 'enabled', True) if getattr(config, 'feedback', None) else True,
        'enable_cross_dataset': getattr(config, 'enable_cross_dataset', False) or getattr(config, 'cross_dataset_enabled', False),
        'explain_n': getattr(config.explainability, 'explain_n', 200),
    }
    artifact_manager.save_metadata(run_metadata)
    
    # Setup logging so all messages go to run directory (logs never empty)
    setup_root_logger(config.log_dir)
    logger = setup_root_logger(config.log_dir)
    
    # Write first line to run.log immediately so run always has log content
    _write_run_log(config.run_dir, f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Pipeline started. Run ID: {config.run_id}\n")
    
    logger.info("="*80)
    logger.info("Multi-Agent AI Pipeline for Explainable Healthcare Analytics")
    logger.info("="*80)
    logger.info(f"Run ID: {config.run_id}")
    logger.info(f"Output directory: {config.run_dir}")
    logger.info(f"Selected dataset: {getattr(config.data, 'dataset_name', None) or (config.data.datasets[0] if getattr(config.data, 'datasets', None) else 'diabetes')}")

    try:
        base_run_dir = Path(config.run_dir)
        base_run_dir.mkdir(parents=True, exist_ok=True)

        # ============================================================
        # PART 1: Baseline pipeline (Data → Model → Explainability → Evaluation; no Feedback Agent)
        # ============================================================
        logger.info("\n" + "="*80)
        logger.info("PART 1: Executing Baseline Pipeline (run_baseline_pipeline)")
        logger.info("="*80)
        baseline_results = run_baseline_pipeline(config, base_run_dir)
        _write_run_log(base_run_dir, f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Baseline pipeline completed. Artifacts: baseline/\n")
        logger.info("Baseline pipeline completed. Artifacts: run_dir/baseline/")

        # ============================================================
        # PART 2: Multi-agent pipeline (Data → Model → Explainability → Evaluation → Feedback Agent)
        # ============================================================
        logger.info("\n" + "="*80)
        logger.info("PART 2: Executing Multi-Agent Pipeline (run_multi_agent_pipeline)")
        logger.info("="*80)
        multi_agent_results = run_multi_agent_pipeline(config, base_run_dir, baseline_results=baseline_results)
        _write_run_log(base_run_dir, f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Multi-agent pipeline completed. Artifacts: multi_agent/\n")
        logger.info("Multi-agent pipeline completed. Artifacts: run_dir/multi_agent/")
        # Copy multi_agent artifacts to main run_dir so dashboard pages (Overview, Modeling, etc.) can find them
        import shutil
        ma_dir = base_run_dir / 'multi_agent'
        for sub in ('models', 'data', 'explainability', 'figures'):
            src = ma_dir / sub
            dst = base_run_dir / sub
            if src.exists():
                dst.mkdir(parents=True, exist_ok=True)
                for f in src.iterdir():
                    if f.is_file():
                        shutil.copy2(f, dst / f.name)
                    elif f.is_dir():
                        (dst / f.name).mkdir(parents=True, exist_ok=True)
                        shutil.copytree(f, dst / f.name, dirs_exist_ok=True)
        # Copy feedback.json so dashboard header chips can display Feedback decision (research_outputs then reports)
        base_run_dir.mkdir(parents=True, exist_ok=True)
        reports_dst = base_run_dir / 'reports'
        reports_dst.mkdir(parents=True, exist_ok=True)
        feedback_src = ma_dir / 'research_outputs' / 'feedback.json'
        if not feedback_src.exists():
            feedback_src = ma_dir / 'reports' / 'feedback.json'
        if feedback_src.exists():
            shutil.copy2(feedback_src, reports_dst / 'feedback.json')

        # ============================================================
        # PART 3: Comprehensive Evaluation
        # ============================================================
        logger.info("\n" + "="*80)
        logger.info("PART 3: Comprehensive Evaluation")
        logger.info("="*80)
        
        # Ensure main run reports dir exists (comparison and result summaries go here)
        config.evaluation.results_dir.mkdir(parents=True, exist_ok=True)

        # A) Predictive Accuracy Comparison (compare the two result dicts only)
        logger.info("\n--- A) Predictive Accuracy Evaluation ---")
        metrics_evaluator = MetricsEvaluator()
        comparison_report = metrics_evaluator.generate_comparison_report(
            multi_agent_results,
            baseline_results,
            output_path=config.evaluation.results_dir / 'comparison_report.csv'
        )
        logger.info("Predictive accuracy comparison completed")

        # Merged table: baseline vs multi-agent vs multi-agent+crossdataset (execution time + metric deltas)
        cross_dataset_results = multi_agent_results.get("cross_dataset")
        merged_report = metrics_evaluator.generate_merged_comparison_report(
            baseline_results,
            multi_agent_results,
            cross_dataset_results=cross_dataset_results,
            output_path=config.evaluation.results_dir / "merged_comparison_report.csv",
        )
        logger.info("Merged comparison (baseline vs multi_agent vs multi_agent+crossdataset) completed")

        # Save serialized result dicts for benchmarking page (compare those two only)
        bl_ser, ma_ser = _serialize_results_for_dashboard(baseline_results, multi_agent_results)
        ma_ser["cross_dataset"] = multi_agent_results.get("cross_dataset_report") or multi_agent_results.get("cross_dataset")
        with open(config.evaluation.results_dir / 'baseline_results.json', 'w') as f:
            json.dump(bl_ser, f, indent=2, default=json_safe)
        with open(config.evaluation.results_dir / 'multi_agent_results.json', 'w') as f:
            json.dump(ma_ser, f, indent=2, default=json_safe)

        # Save classification metrics for selected best model (Evaluation Agent dashboard)
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            ma_models = multi_agent_results.get('models') or {}
            ma_data = multi_agent_results.get('data') or {}
            sel_name = ma_models.get('selected_model') or ma_models.get('best_model_name')
            models = ma_models.get('models') or {}
            val_df = ma_data.get('val_df')
            target_col = ma_data.get('target_column')
            if sel_name and sel_name in models and val_df is not None and target_col:
                X_val = val_df.drop(columns=[target_col])
                y_val = val_df[target_col]
                model = models[sel_name]
                y_pred = model.predict(X_val)
                y_prob = None
                if hasattr(model, 'predict_proba'):
                    try:
                        y_prob = model.predict_proba(X_val)[:, 1]
                    except Exception:
                        pass
                n_unique = len(set(y_val.ravel()))
                eval_metrics = {
                    'accuracy': float(accuracy_score(y_val, y_pred)),
                    'precision': float(precision_score(y_val, y_pred, average='binary', zero_division=0)),
                    'recall': float(recall_score(y_val, y_pred, average='binary', zero_division=0)),
                    'f1': float(f1_score(y_val, y_pred, average='binary', zero_division=0)),
                }
                if y_prob is not None and n_unique > 1:
                    eval_metrics['roc_auc'] = float(roc_auc_score(y_val, y_prob))
                else:
                    sel_m = ma_models.get('selected_metrics') or {}
                    r = sel_m.get('roc_auc') or sel_m.get('roc_auc_score')
                    if r is not None and not (isinstance(r, float) and np.isnan(r)):
                        eval_metrics['roc_auc'] = float(r)
                    else:
                        eval_metrics['roc_auc'] = 0.0
                with open(config.evaluation.results_dir / 'evaluation_metrics.json', 'w') as f:
                    json.dump(eval_metrics, f, indent=2)
                logger.info("evaluation_metrics.json saved (accuracy, precision, recall, f1, roc_auc)")
        except Exception as e:
            logger.warning(f"Could not save evaluation_metrics.json: {e}")

        # Save feedback + iterations for dashboard Overview (action, reason, iteration_1 vs iteration_2 metrics)
        def _flat_metrics(m):
            if not m or not isinstance(m, dict):
                return m
            return {k: float(v) if isinstance(v, (int, float)) else v for k, v in m.items()}
        fd = multi_agent_results.get('feedback_decision') or {}
        feedback_data = {
            'feedback_decision': fd,
            'action_taken': fd.get('action') or 'accept',
            'iteration_count': multi_agent_results.get('iteration_count', 1),
            'retraining_status': multi_agent_results.get('retraining_status', False),
            'iteration_2_ran': 'iteration_2' in multi_agent_results,
        }
        it1 = multi_agent_results.get('iteration_1') or {}
        it2 = multi_agent_results.get('iteration_2') or {}
        feedback_data['iteration_1_selected_model'] = (it1.get('models') or {}).get('selected_model') or (it1.get('models') or {}).get('best_model_name')
        feedback_data['iteration_1_selected_metrics'] = _flat_metrics((it1.get('models') or {}).get('selected_metrics'))
        if feedback_data['iteration_2_ran'] and it2:
            feedback_data['iteration_2_selected_model'] = (it2.get('models') or {}).get('selected_model') or (it2.get('models') or {}).get('best_model_name')
            feedback_data['iteration_2_selected_metrics'] = _flat_metrics((it2.get('models') or {}).get('selected_metrics'))
        with open(config.evaluation.results_dir / 'multi_agent_feedback.json', 'w') as f:
            json.dump(feedback_data, f, indent=2, default=json_safe)
        
        # B) Explainability Quality Evaluation
        logger.info("\n--- B) Explainability Quality Evaluation ---")
        explainability_evaluator = ExplainabilityEvaluator()
        
        # Prepare data for explainability evaluation
        X_val = multi_agent_results['data']['val_df'].drop(
            columns=[multi_agent_results['data']['target_column']]
        )
        
        explainability_report = explainability_evaluator.generate_explainability_report(
            explainability_results=multi_agent_results['explainability'],
            models=multi_agent_results['models']['models'],
            X_val=X_val
        )
        logger.info("Explainability quality evaluation completed")

        # Baseline vs multi-agent comparison JSON for dashboard Overview
        try:
            baseline_comparison = metrics_evaluator.generate_baseline_comparison_json(
                baseline_results,
                multi_agent_results,
                explainability_report=explainability_report,
                output_path=config.evaluation.results_dir / 'baseline_comparison.json',
            )
            logger.info("Baseline comparison JSON saved to research_outputs/baseline_comparison.json")
        except Exception as e:
            logger.warning(f"Could not save baseline_comparison.json: {e}")
        
        # C) Collaboration Efficiency Evaluation
        logger.info("\n--- C) Collaboration Efficiency Evaluation ---")
        collaboration_evaluator = CollaborationEvaluator()
        collaboration_report = collaboration_evaluator.generate_collaboration_report(
            collaboration_metrics=multi_agent_results['collaboration_metrics'],
            baseline_time=baseline_results['execution_time']
        )
        logger.info("Collaboration efficiency evaluation completed")

        # D) Research Scorecard (roc_auc, explainability, ERI, feedback, cross_dataset; null if missing)
        res_dir = Path(config.evaluation.results_dir)
        res_dir.mkdir(parents=True, exist_ok=True)
        ma_sel_name = multi_agent_results.get('models', {}).get('selected_model') or multi_agent_results.get('models', {}).get('best_model_name')
        ma_sel_metrics = multi_agent_results.get('models', {}).get('selected_metrics') or {}
        fd = (multi_agent_results.get('artifacts') or {}).get('feedback') or multi_agent_results.get('feedback_decision') or {}
        cross_res = multi_agent_results.get('cross_dataset') or {}

        def _v(x):
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return None
            return float(x) if isinstance(x, (int, float)) else x

        roc_auc = _v(ma_sel_metrics.get('roc_auc'))
        explainability_readability = None
        explainability_stability = None
        explanation_fidelity = None
        ERI = None
        if explainability_report:
            if ma_sel_name and (explainability_report.get('readability') or {}).get(ma_sel_name):
                r = explainability_report['readability'][ma_sel_name]
                explainability_readability = _v(r.get('mean_readability') if isinstance(r, dict) else r)
            if ma_sel_name and (explainability_report.get('shap_stability') or {}).get(ma_sel_name):
                s = explainability_report['shap_stability'][ma_sel_name]
                explainability_stability = _v(s.get('mean_rank_correlation') if isinstance(s, dict) else s)
            if ma_sel_name and (explainability_report.get('fidelity') or {}).get(ma_sel_name):
                f = explainability_report['fidelity'][ma_sel_name]
                explanation_fidelity = _v(f.get('correlation') if isinstance(f, dict) else f)
            if explainability_stability is not None and explanation_fidelity is not None:
                ERI = (explainability_stability + explanation_fidelity) / 2.0
            elif fd.get('trigger_metric_name') == 'eri':
                ERI = _v(fd.get('trigger_metric_value'))

        feedback_triggered = fd.get('decision') not in (None, 'none', '')
        action = (fd.get('decision') or fd.get('action') or 'none')
        if action not in ('switch_model', 'retrain_best_model', 'retrain_with_tuned_params', 'none'):
            action = 'none'
        cross_dataset_generalization_score = None
        if cross_res.get('results'):
            rocs = [r.get('cross_dataset_metrics', {}).get('roc_auc') for r in cross_res['results'] if isinstance(r.get('cross_dataset_metrics'), dict)]
            rocs = [x for x in rocs if x is not None]
            if rocs:
                cross_dataset_generalization_score = float(np.mean(rocs))

        scorecard = {
            'roc_auc': roc_auc,
            'explainability_readability': explainability_readability,
            'explainability_stability': explainability_stability,
            'explanation_fidelity': explanation_fidelity,
            'ERI': ERI,
            'feedback_triggered': feedback_triggered,
            'action': action,
            'cross_dataset_generalization_score': cross_dataset_generalization_score,
        }
        scorecard_path = res_dir / 'research_scorecard.csv'
        try:
            pd.DataFrame([scorecard]).to_csv(scorecard_path, index=False)
            logger.info(f"Research scorecard saved to {scorecard_path}")
        except Exception as e:
            logger.warning(f"Could not write research scorecard: {e}")

        for json_candidate in [res_dir / 'cross_dataset_report.json', config.run_dir / 'multi_agent' / 'research_outputs' / 'cross_dataset_report.json']:
            if json_candidate.exists():
                try:
                    with open(json_candidate, 'r') as f:
                        cross_report = json.load(f)
                    cross_report['research_scorecard'] = scorecard
                    with open(json_candidate, 'w') as f:
                        json.dump(cross_report, f, indent=2, default=json_safe)
                    logger.info(f"Updated research_scorecard in {json_candidate}")
                except Exception as e:
                    logger.warning(f"Could not update cross_dataset_report.json with scorecard: {e}")
                break

        # ============================================================
        # PART 4: Final Summary
        # ============================================================
        logger.info("\n" + "="*80)
        logger.info("FINAL SUMMARY")
        logger.info("="*80)
        
        # Print key results (selected_metrics only)
        ma_sel = multi_agent_results['models'].get('selected_model') or multi_agent_results['models']['best_model_name']
        bl_sel = baseline_results.get('selected_model') or baseline_results['best_model_name']
        ma_selected = multi_agent_results['models'].get('selected_metrics') or {}
        bl_selected = baseline_results.get('selected_metrics') or {}
        logger.info("\n--- Selected Models ---")
        logger.info(f"Multi-Agent selected: {ma_sel} (ROC-AUC: {ma_selected.get('roc_auc', 0):.4f})")
        logger.info(f"Baseline selected: {bl_sel} (ROC-AUC: {bl_selected.get('roc_auc', 0):.4f})")
        
        logger.info("\n--- Execution Times ---")
        ma_time = multi_agent_results['collaboration_metrics']['total_execution_time']
        bl_time = baseline_results['execution_time']
        logger.info(f"Multi-Agent: {ma_time:.2f} seconds")
        logger.info(f"Baseline: {bl_time:.2f} seconds")
        logger.info(f"Overhead: {ma_time - bl_time:.2f} seconds ({(ma_time - bl_time) / bl_time * 100:.1f}%)")
        
        logger.info("\n--- Explainability Quality ---")
        if explainability_report.get('readability'):
            for model_name, scores in explainability_report['readability'].items():
                logger.info(f"{model_name}: Readability = {scores['mean_readability']:.3f}")
        
        ma_sel = multi_agent_results['models'].get('selected_model') or multi_agent_results['models']['best_model_name']
        bl_sel = baseline_results.get('selected_model') or baseline_results['best_model_name']
        feedback_artifact = (multi_agent_results.get('artifacts') or {}).get('feedback') or multi_agent_results.get('feedback_decision') or {}
        run_metadata.update({
            'status': 'success',
            'total_execution_time': multi_agent_results['collaboration_metrics']['total_execution_time'],
            'best_model_multi_agent': ma_sel,
            'best_model_baseline': bl_sel,
            'selected_model_multi_agent': ma_sel,
            'selected_model_baseline': bl_sel,
            'best_model_roc_auc_ma': (multi_agent_results['models'].get('selected_metrics') or {}).get('roc_auc'),
            'best_model_roc_auc_bl': (baseline_results.get('selected_metrics') or {}).get('roc_auc'),
            'best_model_before_feedback': feedback_artifact.get('selected_model_before'),
            'selected_model_after_feedback': feedback_artifact.get('selected_model_after_feedback'),
            'retraining_performed': feedback_artifact.get('retrained', feedback_artifact.get('retraining_performed')),
            'feedback_decision': feedback_artifact.get('decision'),
            'metric_triggered_on': feedback_artifact.get('trigger_metric_name'),
            'enable_feedback': getattr(config.feedback, 'enabled', True) if getattr(config, 'feedback', None) else True,
            'enable_cross_dataset': getattr(config, 'enable_cross_dataset', False) or getattr(config, 'cross_dataset_enabled', False),
            'imbalance_strategy': (multi_agent_results.get('models') or {}).get('imbalance_strategy'),
        })
        artifact_manager.save_metadata(run_metadata)
        
        logger.info("\n--- Output Files ---")
        logger.info(f"Run directory: {config.run_dir}")
        logger.info(f"Results directory: {config.evaluation.results_dir}")
        logger.info(f"Models directory: {config.model.models_dir}")
        logger.info(f"Explanations directory: {config.explainability.explanations_dir}")
        logger.info(f"Plots directory: {config.explainability.plots_dir}")
        logger.info(f"Logs directory: {config.log_dir}")
        
        logger.info("\n" + "="*80)
        logger.info("Pipeline execution completed successfully!")
        logger.info(f"View results in dashboard: streamlit run dashboard/app.py")
        logger.info("="*80)
        
        return {
            'run_id': config.run_id,
            'multi_agent_results': multi_agent_results,
            'baseline_results': baseline_results,
            'comparison_report': comparison_report,
            'explainability_report': explainability_report,
            'collaboration_report': collaboration_report
        }
        
    except Exception as e:
        # Ensure run log is never empty: write failure and traceback to run.log
        err_msg = (
            f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Pipeline execution failed: {e}\n"
            f"{traceback.format_exc()}\n"
        )
        _write_run_log(config.run_dir, err_msg)
        logger.error(f"Pipeline execution failed: {e}")
        # Update metadata with failure - ensure it's saved
        try:
            run_metadata.update({
                'status': 'failed',
                'error': str(e),
                'total_execution_time': 0
            })
            artifact_manager.save_metadata(run_metadata)
        except Exception as save_error:
            logger.error(f"Failed to save metadata: {save_error}")
        raise


def _parse_args():
    """Parse CLI args: optional positional dataset, or --dataset, --cross_dataset (enables cross-dataset validation)."""
    parser = argparse.ArgumentParser(description="Multi-Agent Healthcare Analytics Pipeline")
    parser.add_argument(
        "dataset_positional",
        nargs="?",
        default=None,
        metavar="DATASET",
        help="Primary dataset (e.g. diabetes, heart_disease). Same as --dataset. Default: diabetes.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Primary dataset (e.g. diabetes, heart_disease). Overrides positional if both given.",
    )
    parser.add_argument(
        "--cross_dataset",
        type=str,
        default=None,
        metavar="TARGET",
        help="Enable cross-dataset validation; target dataset (e.g. heart_disease).",
    )
    parser.add_argument(
        "--cross-dataset",
        type=str,
        default=None,
        dest="cross_dataset_hyphen",
        metavar="TARGET",
        help="Same as --cross_dataset: enable cross-dataset validation and set target dataset.",
    )
    parser.add_argument(
        "--no-feature-engineering",
        action="store_true",
        help="Disable FeatureEngineeringAgent (use raw DataAgent output for modeling).",
    )
    args = parser.parse_args()
    # --dataset overrides positional (e.g. python main.py diabetes --dataset heart_disease → heart_disease)
    dataset = args.dataset if args.dataset is not None else args.dataset_positional
    cross_target = args.cross_dataset or args.cross_dataset_hyphen
    return dataset, cross_target, getattr(args, "no_feature_engineering", False)


if __name__ == '__main__':
    dataset_arg, cross_dataset_arg, no_fe = _parse_args()
    results = main(dataset=dataset_arg, cross_dataset_target=cross_dataset_arg, no_feature_engineering=no_fe)
