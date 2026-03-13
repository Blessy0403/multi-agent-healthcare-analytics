"""
Central pipeline orchestrator: enforces strict agent order and explicit artifact passing.

Order: DataAgent → FeatureEngineeringAgent (if enabled) → ModelAgent → ExplainabilityAgent → EvaluationAgent → FeedbackAgent → CrossDatasetAgent (if enabled).
Each agent has a run(artifacts) method and returns structured artifacts.
"""

import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd

from utils.config import PipelineConfig, get_config
from utils.json_utils import json_safe
from utils.logging import AgentLogger, setup_root_logger
from agents.data_agent import DataAgent
from agents.feature_engineering_agent import FeatureEngineeringAgent
from agents.model_agent import ModelAgent
from agents.explainability_agent import ExplainabilityAgent
from agents.evaluation_agent import EvaluationAgent
from agents.feedback_agent import FeedbackAgent
from agents.cross_dataset_agent import CrossDatasetAgent
from evaluation.explainability_eval import ExplainabilityEvaluator


def execute_pipeline(config: Optional[PipelineConfig] = None) -> Dict[str, Any]:
    """
    Run the full pipeline in strict order, passing artifacts explicitly between agents.

    1. DataAgent
    2. FeatureEngineeringAgent (if config.enable_feature_engineering)
    3. ModelAgent
    4. ExplainabilityAgent
    5. EvaluationAgent
    6. FeedbackAgent
    7. CrossDatasetAgent (if config.cross_dataset_enabled)

    Returns:
        Dict with keys: data, models, explainability, evaluation, feedback,
        cross_dataset (if enabled), collaboration_metrics, artifacts.
    """
    config = config or get_config()
    logger = AgentLogger("pipeline.orchestrator")

    collaboration_metrics = {
        "agent_execution_times": {},
        "handovers": [],
        "errors": [],
        "artifacts_passed": {},
    }
    artifacts: Dict[str, Any] = {}
    pipeline_start = time.time()

    def handover(from_agent: str, to_agent: str, elapsed: float, artifact_keys: list):
        collaboration_metrics["handovers"].append({
            "timestamp": datetime.now().isoformat(),
            "from": from_agent,
            "to": to_agent,
            "artifacts": artifact_keys,
            "execution_time": elapsed,
        })

    try:
        setup_root_logger(config.log_dir)
        artifacts["run_dir"] = Path(config.run_dir)
        # Base run dir (e.g. outputs/runs/run_xxx) for cross-dataset outputs: run_dir/research_outputs and run_dir/multi_agent/research_outputs
        _run_dir_path = Path(config.run_dir)
        artifacts["base_run_dir"] = (
            getattr(config, "base_run_dir", None)
            or (_run_dir_path.parent if _run_dir_path.name == "multi_agent" else _run_dir_path)
        )
        if artifacts["base_run_dir"] is not None:
            artifacts["base_run_dir"] = Path(artifacts["base_run_dir"])

        logger.info("=" * 80)
        logger.info("Pipeline: DataAgent → FeatureEngineeringAgent (if enabled) → ModelAgent → ExplainabilityAgent → EvaluationAgent → FeedbackAgent → CrossDatasetAgent (if enabled)")
        logger.info("=" * 80)

        enable_fe = getattr(config, "enable_feature_engineering", True)

        # -------------------------------------------------------------------------
        # 1. DataAgent
        # -------------------------------------------------------------------------
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: DataAgent")
        logger.info("=" * 80)
        t0 = time.time()
        data_agent = DataAgent(config.data)
        artifacts["data"] = data_agent.run(artifacts)
        elapsed = time.time() - t0
        collaboration_metrics["agent_execution_times"]["data_agent"] = elapsed
        logger.info(f"DataAgent completed in {elapsed:.2f}s")
        _copy_data_to_run_dir(config, artifacts["data"])
        next_agent = "feature_engineering_agent" if enable_fe else "model_agent"
        handover("data_agent", next_agent, elapsed, list((artifacts["data"].get("file_paths") or {}).keys()))

        # -------------------------------------------------------------------------
        # 2. FeatureEngineeringAgent (if enabled)
        # -------------------------------------------------------------------------
        if enable_fe:
            logger.info("\n" + "=" * 80)
            logger.info("STEP 2: FeatureEngineeringAgent")
            logger.info("=" * 80)
            t0 = time.time()
            fe_agent = FeatureEngineeringAgent(config)
            artifacts["features"] = fe_agent.run(artifacts)
            elapsed = time.time() - t0
            collaboration_metrics["agent_execution_times"]["feature_engineering_agent"] = elapsed
            logger.info(f"FeatureEngineeringAgent completed in {elapsed:.2f}s")
            handover("feature_engineering_agent", "model_agent", elapsed, list((artifacts["features"].get("file_paths") or {}).keys()))
        else:
            logger.info("\nFeatureEngineeringAgent skipped (enable_feature_engineering=False).")

        # -------------------------------------------------------------------------
        # 3. ModelAgent
        # -------------------------------------------------------------------------
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: ModelAgent")
        logger.info("=" * 80)
        t0 = time.time()
        artifacts["baseline_for_model_agent"] = getattr(config, "baseline_for_model_agent", None)
        model_agent = ModelAgent(config.model)
        artifacts["models"] = model_agent.run(artifacts)
        elapsed = time.time() - t0
        collaboration_metrics["agent_execution_times"]["model_agent"] = elapsed
        logger.info(f"ModelAgent completed in {elapsed:.2f}s")
        handover("model_agent", "explainability_agent", elapsed, list((artifacts["models"].get("file_paths") or {}).keys()))

        # -------------------------------------------------------------------------
        # 3. ExplainabilityAgent
        # -------------------------------------------------------------------------
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: ExplainabilityAgent")
        logger.info("=" * 80)
        t0 = time.time()
        explainability_agent = ExplainabilityAgent(config.explainability)
        artifacts["explainability"] = explainability_agent.run(artifacts)
        elapsed = time.time() - t0
        collaboration_metrics["agent_execution_times"]["explainability_agent"] = elapsed
        logger.info(f"ExplainabilityAgent completed in {elapsed:.2f}s")
        handover("explainability_agent", "evaluation_agent", elapsed, ["explainability"])

        # -------------------------------------------------------------------------
        # 5. EvaluationAgent
        # -------------------------------------------------------------------------
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: EvaluationAgent")
        logger.info("=" * 80)
        t0 = time.time()
        evaluation_agent = EvaluationAgent(config.evaluation, logger=logger)
        artifacts["evaluation"] = evaluation_agent.run(artifacts)
        if not artifacts["evaluation"].get("roc_auc") and isinstance(artifacts.get("models"), dict):
            sm = (artifacts["models"].get("selected_metrics") or {})
            if isinstance(sm, dict):
                artifacts["evaluation"]["roc_auc"] = sm.get("roc_auc")
        elapsed = time.time() - t0
        collaboration_metrics["agent_execution_times"]["evaluation_agent"] = elapsed
        logger.info(f"EvaluationAgent completed in {elapsed:.2f}s")
        handover("evaluation_agent", "feedback_agent", elapsed, ["evaluation"])

        # -------------------------------------------------------------------------
        # 4b. ExplainabilityEvaluator (for FeedbackAgent input: fidelity, shap_stability, readability)
        # (runs after EvaluationAgent; not a separate pipeline step)
        # -------------------------------------------------------------------------
        artifacts["explainability_report"] = {}
        try:
            data = artifacts.get("data") or {}
            val_df = data.get("val_df")
            target_col = data.get("target_column")
            if val_df is not None and target_col is not None:
                X_val = val_df.drop(columns=[target_col])
                models_dict = (artifacts.get("models") or {}).get("models") or {}
                explainability_evaluator = ExplainabilityEvaluator()
                artifacts["explainability_report"] = explainability_evaluator.generate_explainability_report(
                    explainability_results=artifacts.get("explainability") or {},
                    models=models_dict,
                    X_val=X_val,
                )
        except Exception as e:
            logger.warning(f"ExplainabilityEvaluator failed (FeedbackAgent will use proxies): {e}")

        # -------------------------------------------------------------------------
        # 6. FeedbackAgent
        # -------------------------------------------------------------------------
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: FeedbackAgent")
        logger.info("=" * 80)
        t0 = time.time()
        feedback_config = getattr(config, 'feedback', None)
        feedback_agent = FeedbackAgent(config=feedback_config, pipeline_config=config, logger=logger)
        run_context = {
            "evaluation": artifacts["evaluation"],
            "explainability": artifacts.get("explainability") or {},
            "explainability_report": artifacts.get("explainability_report") or {},
            "evaluation_results": {
                "all_models": (artifacts.get("models") or {}).get("all_models") or {},
                "best_model_name": (artifacts.get("models") or {}).get("best_model_name") or (artifacts.get("models") or {}).get("selected_model"),
            },
            "models": artifacts.get("models"),
        }
        artifacts["feedback"] = feedback_agent.run(run_context)
        elapsed = time.time() - t0
        collaboration_metrics["agent_execution_times"]["feedback_agent"] = elapsed
        logger.info(f"FeedbackAgent completed in {elapsed:.2f}s")
        cross_dataset_enabled = (
            getattr(config, "cross_dataset_enabled", False)
            or (getattr(config, "cross_dataset", None) and getattr(config.cross_dataset, "enabled", False))
        )
        handover("feedback_agent", "cross_dataset_agent" if cross_dataset_enabled else "end", elapsed, ["feedback"])

        # -------------------------------------------------------------------------
        # 7. CrossDatasetAgent (if config.cross_dataset.enabled)
        # -------------------------------------------------------------------------
        if cross_dataset_enabled:
            logger.info("\n" + "=" * 80)
            logger.info("STEP 7: CrossDatasetAgent")
            logger.info("=" * 80)
            t0 = time.time()
            try:
                cross_cfg = getattr(config, "cross_dataset", None)
                cross_dataset_agent = CrossDatasetAgent(config=cross_cfg, pipeline_config=config, logger=logger)
                artifacts["cross_dataset"] = cross_dataset_agent.run(artifacts)
                artifacts["cross_dataset_report"] = artifacts["cross_dataset"].get("report") if artifacts.get("cross_dataset") else None
                if artifacts.get("cross_dataset") and artifacts["cross_dataset"].get("report_paths"):
                    logger.info("Cross-dataset report saved to: " + ", ".join(artifacts["cross_dataset"]["report_paths"]))
            except Exception as e:
                logger.warning(f"CrossDatasetAgent failed (pipeline continues): {e}")
                error_report = {"status": "failed", "error": str(e), "timestamp": datetime.now().isoformat()}
                artifacts["cross_dataset"] = {"enabled": True, "report": error_report, "results": [], "summary": str(e)}
                artifacts["cross_dataset_report"] = error_report
                base_run = artifacts.get("base_run_dir")
                if base_run is None:
                    run_dir = Path(artifacts.get("run_dir") or config.run_dir or ".")
                    base_run = run_dir.parent if run_dir.name == "multi_agent" else run_dir
                base_run = Path(base_run)
                path1 = base_run / "research_outputs" / "cross_dataset_report.json"
                path2 = base_run / "multi_agent" / "research_outputs" / "cross_dataset_report.json"
                path1.parent.mkdir(parents=True, exist_ok=True)
                path2.parent.mkdir(parents=True, exist_ok=True)
                for p in (path1, path2):
                    try:
                        with open(p, "w") as f:
                            json.dump(error_report, f, indent=2)
                        logger.info(f"Cross-dataset report saved to: {str(p)}")
                    except Exception:
                        pass
                summary_txt = f"Cross-dataset: status=failed. Error: {e}"
                for d in (path1.parent, path2.parent):
                    try:
                        with open(d / "cross_dataset_summary.txt", "w") as f:
                            f.write(summary_txt)
                    except Exception:
                        pass
            elapsed = time.time() - t0
            collaboration_metrics["agent_execution_times"]["cross_dataset_agent"] = elapsed
            logger.info(f"CrossDatasetAgent completed in {elapsed:.2f}s")
        else:
            logger.info("\nCrossDatasetAgent skipped (config.cross_dataset.enabled=False).")

        # -------------------------------------------------------------------------
        # Finalize
        # -------------------------------------------------------------------------
        total = time.time() - pipeline_start
        collaboration_metrics["total_execution_time"] = total
        collaboration_metrics["pipeline_status"] = "success"

        logger.info("\n" + "=" * 80)
        logger.info("Pipeline completed successfully")
        logger.info("=" * 80)
        logger.info(f"Total: {total:.2f}s")

        _save_collaboration_metrics(config, collaboration_metrics)
        agents_for_log = {
            "data_agent": data_agent,
            "model_agent": model_agent,
            "explainability_agent": explainability_agent,
        }
        if enable_fe:
            agents_for_log["feature_engineering_agent"] = fe_agent
        for agent_name in list(agents_for_log.keys()):
            try:
                agent = agents_for_log.get(agent_name)
                if agent and hasattr(agent, "logger") and hasattr(agent.logger, "save_collaboration_log"):
                    agent.logger.save_collaboration_log()
            except Exception:
                pass

        return {
            "data": artifacts["data"],
            "models": artifacts["models"],
            "explainability": artifacts["explainability"],
            "evaluation": artifacts["evaluation"],
            "feedback": artifacts["feedback"],
            "collaboration_metrics": collaboration_metrics,
            "artifacts": {
                "data": artifacts["data"],
                "models": artifacts["models"],
                "explainability": artifacts["explainability"],
                "feedback": artifacts["feedback"],
                **({"features": artifacts["features"]} if artifacts.get("features") is not None else {}),
            },
            **({"cross_dataset": artifacts["cross_dataset"]} if artifacts.get("cross_dataset") is not None else {}),
            **({"cross_dataset_report": artifacts["cross_dataset_report"]} if artifacts.get("cross_dataset_report") is not None else {}),
        }

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        collaboration_metrics["pipeline_status"] = "failed"
        collaboration_metrics["errors"].append({
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "type": type(e).__name__,
        })
        _save_collaboration_metrics(config, collaboration_metrics)
        raise


def _build_data_profile(train_path: Path, target_column: str) -> Dict[str, Any]:
    """Build minimal data_profile for dashboard: columns, dtypes, min/max/median per feature, target."""
    try:
        df = pd.read_csv(train_path, nrows=10000)
    except Exception:
        return {}
    cols = [c for c in df.columns if c != target_column]
    if not cols:
        return {"columns": list(df.columns), "target_column": target_column, "feature_stats": {}}
    feature_stats = {}
    for c in cols:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            feature_stats[c] = {
                "min": float(s.min()),
                "max": float(s.max()),
                "median": float(s.median()),
                "dtype": str(s.dtype),
            }
        else:
            feature_stats[c] = {"dtype": str(s.dtype), "min": None, "max": None, "median": None}
    return {
        "columns": cols,
        "target_column": target_column,
        "feature_stats": feature_stats,
        "dtypes": {c: str(df[c].dtype) for c in cols},
    }


def _copy_data_to_run_dir(config: PipelineConfig, data_results: Dict[str, Any]) -> None:
    """Copy processed data and metadata to config.run_dir for dashboard access; write data_profile.json."""
    run_data_dir = config.run_dir / "data"
    run_data_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = getattr(config.data, "dataset_name", None) or (config.data.datasets[0] if getattr(config.data, "datasets", None) else "heart_disease")
    for split in ["train", "val", "test"]:
        src = (data_results.get("file_paths") or {}).get(split)
        if src and Path(src).exists():
            dst = run_data_dir / f"{dataset_name}_{split}.csv"
            shutil.copy2(src, dst)
    src_meta = (data_results.get("file_paths") or {}).get("metadata")
    target_col = data_results.get("target_column") or "target"
    if src_meta and Path(src_meta).exists():
        config.run_dir.mkdir(parents=True, exist_ok=True)
        reports = config.run_dir / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        dst_meta = reports / f"data_metadata_{dataset_name}.json"
        shutil.copy2(src_meta, dst_meta)
    # Save data_profile.json (columns, dtypes, min/max/median) for Prediction Explainer
    train_dst = run_data_dir / f"{dataset_name}_train.csv"
    if train_dst.exists():
        profile = _build_data_profile(train_dst, target_col)
        if profile:
            for base in (config.run_dir / "reports", config.run_dir / "research_outputs"):
                base.mkdir(parents=True, exist_ok=True)
                out_path = base / f"data_profile_{dataset_name}.json"
                with open(out_path, "w") as f:
                    json.dump(profile, f, indent=2, default=json_safe)
        # Class distribution for dashboard Data page (labels + counts)
        try:
            train_df = pd.read_csv(train_dst)
            if target_col in train_df.columns:
                vc = train_df[target_col].value_counts().sort_index()
                class_dist = {"labels": [str(k) for k in vc.index.tolist()], "counts": vc.values.tolist()}
                for base in (config.run_dir / "reports", config.run_dir / "research_outputs"):
                    base.mkdir(parents=True, exist_ok=True)
                    cd_path = base / f"class_distribution_{dataset_name}.json"
                    with open(cd_path, "w") as f:
                        json.dump(class_dist, f, indent=2, default=json_safe)
        except Exception:
            pass


def _save_collaboration_metrics(config: PipelineConfig, metrics: Dict[str, Any]) -> None:
    """Persist collaboration metrics to config.log_dir."""
    path = config.log_dir / "collaboration_metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "agent_execution_times": {k: float(v) for k, v in metrics.get("agent_execution_times", {}).items()},
        "handovers": metrics.get("handovers", []),
        "errors": metrics.get("errors", []),
        "total_execution_time": float(metrics.get("total_execution_time", 0)),
        "pipeline_status": metrics.get("pipeline_status", "unknown"),
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=json_safe)
