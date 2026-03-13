"""
Single-Model Baseline Pipeline: Traditional non-agent-based approach.

This module implements a traditional single-pipeline baseline for comparison
with the multi-agent approach. Uses the same dataset, preprocessing, and models,
but without agent separation.

Standalone mode: Data → Model → Explainability → Evaluation (no Feedback Agent),
with independent artifacts and logging under run_dir/baseline/.
"""

import pickle
import json
import time
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
import xgboost as xgb

from utils.config import get_config
from utils.json_utils import json_safe
from utils.logging import AgentLogger
from agents.data_agent import DataAgent
from agents.explainability_agent import ExplainabilityAgent


class SingleModelPipeline:
    """
    Traditional single-pipeline baseline without agent separation.
    
    This class performs all steps (data processing, modeling, evaluation)
    in a single monolithic pipeline for comparison with the multi-agent approach.
    """
    
    def __init__(self, config: Optional[Any] = None):
        """Initialize single-model pipeline. If config is provided (e.g. from main), use it so --dataset is respected."""
        self.config = config if config is not None else get_config()
        self.logger = AgentLogger('baseline_pipeline')
        
        # Results storage
        self.models = {}
        self.metrics = {}
        self.predictions = {}
    
    def _train_and_evaluate(self, data_results: Dict[str, Any]) -> tuple:
        """
        Train all models and evaluate on validation/test. Sets self.models, self.metrics, self.predictions.
        
        Returns:
            (execution_time, best_model_name, test_metrics)
        """
        pipeline_start = time.time()
        train_df = data_results['train_df']
        val_df = data_results['val_df']
        test_df = data_results['test_df']
        target_col = data_results['target_column']

        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_val = val_df.drop(columns=[target_col])
        y_val = val_df[target_col]
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        def _scale_pos_weight(yt: pd.Series) -> float:
            c = yt.value_counts()
            if len(c) < 2:
                return 1.0
            pos = int(c.get(1, c.get(1.0, 0)))
            neg = int(c.get(0, c.get(0.0, 0)))
            return max(0.1, neg / pos) if pos > 0 else 1.0
        _spw = _scale_pos_weight(y_train)

        self.logger.info("Training models...")
        # Logistic Regression
        if 'logistic_regression' in self.config.model.models:
            self.logger.info("Training Logistic Regression...")
            lr = LogisticRegression(random_state=42, solver='liblinear')
            grid_search = GridSearchCV(
                lr,
                self.config.model.lr_params,
                cv=self.config.model.cv_folds,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X_train, y_train)
            self.models['logistic_regression'] = grid_search.best_estimator_
            y_pred = self.models['logistic_regression'].predict(X_val)
            y_pred_proba = self.models['logistic_regression'].predict_proba(X_val)[:, 1]
            self.predictions['logistic_regression'] = y_pred
            self.metrics['logistic_regression'] = self._compute_metrics(
                y_val, y_pred, y_pred_proba
            )
        # Random Forest
        if 'random_forest' in self.config.model.models:
            self.logger.info("Training Random Forest...")
            rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
            grid_search = GridSearchCV(
                rf,
                self.config.model.rf_params,
                cv=self.config.model.cv_folds,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X_train, y_train)
            self.models['random_forest'] = grid_search.best_estimator_
            y_pred = self.models['random_forest'].predict(X_val)
            y_pred_proba = self.models['random_forest'].predict_proba(X_val)[:, 1]
            self.predictions['random_forest'] = y_pred
            self.metrics['random_forest'] = self._compute_metrics(
                y_val, y_pred, y_pred_proba
            )
        # XGBoost
        if 'xgboost' in self.config.model.models:
            self.logger.info("Training XGBoost...")
            xgb_model = xgb.XGBClassifier(
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False,
                scale_pos_weight=_spw
            )
            grid_search = GridSearchCV(
                xgb_model,
                self.config.model.xgb_params,
                cv=self.config.model.cv_folds,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X_train, y_train)
            self.models['xgboost'] = grid_search.best_estimator_
            y_pred = self.models['xgboost'].predict(X_val)
            y_pred_proba = self.models['xgboost'].predict_proba(X_val)[:, 1]
            self.predictions['xgboost'] = y_pred
            self.metrics['xgboost'] = self._compute_metrics(
                y_val, y_pred, y_pred_proba
            )
        # SVM
        if 'svm' in self.config.model.models:
            self.logger.info("Training SVM...")
            svm_model = SVC(probability=True, random_state=42, class_weight='balanced')
            grid_search = GridSearchCV(
                svm_model,
                self.config.model.svm_params,
                cv=self.config.model.cv_folds,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X_train, y_train)
            self.models['svm'] = grid_search.best_estimator_
            y_pred = self.models['svm'].predict(X_val)
            y_pred_proba = self.models['svm'].predict_proba(X_val)[:, 1]
            self.predictions['svm'] = y_pred
            self.metrics['svm'] = self._compute_metrics(y_val, y_pred, y_pred_proba)
        # Gradient Boosting
        if 'gradient_boosting' in self.config.model.models:
            self.logger.info("Training Gradient Boosting...")
            gb_model = GradientBoostingClassifier(random_state=42)
            grid_search = GridSearchCV(
                gb_model,
                self.config.model.gb_params,
                cv=self.config.model.cv_folds,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X_train, y_train)
            self.models['gradient_boosting'] = grid_search.best_estimator_
            y_pred = self.models['gradient_boosting'].predict(X_val)
            y_pred_proba = self.models['gradient_boosting'].predict_proba(X_val)[:, 1]
            self.predictions['gradient_boosting'] = y_pred
            self.metrics['gradient_boosting'] = self._compute_metrics(y_val, y_pred, y_pred_proba)
        # KNN
        if 'knn' in self.config.model.models:
            self.logger.info("Training KNN...")
            knn_model = KNeighborsClassifier()
            grid_search = GridSearchCV(
                knn_model,
                self.config.model.knn_params,
                cv=self.config.model.cv_folds,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X_train, y_train)
            self.models['knn'] = grid_search.best_estimator_
            y_pred = self.models['knn'].predict(X_val)
            y_pred_proba = self.models['knn'].predict_proba(X_val)[:, 1]
            self.predictions['knn'] = y_pred
            self.metrics['knn'] = self._compute_metrics(y_val, y_pred, y_pred_proba)
            
        # Select best model (baseline mode: strictly by ROC-AUC)
        best_model_name = max(
            self.metrics.keys(),
            key=lambda k: self.metrics[k]['roc_auc']
        )
        best_model = self.models[best_model_name]
        y_test_pred = best_model.predict(X_test)
        y_test_pred_proba = best_model.predict_proba(X_test)[:, 1]
        test_metrics = self._compute_metrics(y_test, y_test_pred, y_test_pred_proba)
        pipeline_end = time.time()
        execution_time = pipeline_end - pipeline_start
        return execution_time, best_model_name, test_metrics

    def _save_results(
        self,
        results_dir: Path,
        best_model_name: str,
        test_metrics: Dict[str, Any],
        execution_time: float
    ) -> Dict[str, Any]:
        """Save models and metrics to results_dir; return file_paths and results dict."""
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        model_paths = {}
        for model_name, model in self.models.items():
            model_path = results_dir / f'baseline_{model_name}.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            model_paths[model_name] = model_path
        metrics_path = results_dir / 'baseline_metrics.json'
        metrics_serializable = {
            k: {m: float(v) if isinstance(v, (np.integer, np.floating)) else v
                for m, v in metrics.items()}
            for k, metrics in self.metrics.items()
        }
        metrics_serializable['test_metrics'] = {
            k: float(v) if isinstance(v, (np.integer, np.floating)) else v
            for k, v in test_metrics.items()
        }
        metrics_serializable['best_model'] = best_model_name
        metrics_serializable['execution_time'] = execution_time
        with open(metrics_path, 'w') as f:
            json.dump(metrics_serializable, f, indent=2, default=json_safe)
        best_model = self.models[best_model_name]
        return {
            'models': self.models,
            'selected_model': best_model_name,
            'selected_metrics': self.metrics[best_model_name],
            'all_models': self.metrics,
            'test_metrics': test_metrics,
            'best_model_name': best_model_name,
            'best_model': best_model,
            'execution_time': execution_time,
            'file_paths': {'models': model_paths, 'metrics': metrics_path}
        }

    def execute(self, data_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the complete single-model pipeline using provided data (e.g. from multi-agent).
        
        Args:
            data_results: Results from data processing (can reuse from data agent)
            
        Returns:
            Dictionary containing models, metrics, and execution time
        """
        self.logger.info("="*80)
        self.logger.info("Executing Single-Model Baseline Pipeline (using provided data)")
        self.logger.info("="*80)
        with self.logger.execution_timer():
            execution_time, best_model_name, test_metrics = self._train_and_evaluate(data_results)
            self.logger.info(f"Baseline pipeline completed in {execution_time:.2f} seconds")
            self.logger.info(f"Best model: {best_model_name} (ROC-AUC: {self.metrics[best_model_name]['roc_auc']:.4f})")
            self.logger.info(f"Test set ROC-AUC: {test_metrics['roc_auc']:.4f}")
            return self._save_results(
                self.config.evaluation.results_dir,
                best_model_name,
                test_metrics,
                execution_time
            )

    def execute_standalone(self, run_dir: Path) -> Dict[str, Any]:
        """
        Run a fully independent single-pass baseline pipeline:
        Data → Model → Explainability → Evaluation (no Feedback Agent).
        Produces its own artifacts under run_dir/baseline/ and logs to run_dir/baseline/logs/.
        """
        run_dir = Path(run_dir)
        baseline_dir = run_dir / 'baseline'
        for sub in ('models', 'explainability', 'figures', 'reports', 'data', 'logs'):
            (baseline_dir / sub).mkdir(parents=True, exist_ok=True)
        baseline_log_dir = baseline_dir / 'logs'
        self.logger = AgentLogger('baseline_pipeline', log_dir=baseline_log_dir)

        self.logger.info("="*80)
        self.logger.info("Executing Standalone Baseline Pipeline (Data → Model → Explainability)")
        self.logger.info("="*80)

        pipeline_start = time.time()
        # Step 1: Data (independent of multi-agent)
        self.logger.info("Step 1: Data (baseline data run)")
        data_agent = DataAgent(self.config.data)
        data_results = data_agent.process()
        if data_results.get('train_df') is None:
            raise RuntimeError("Baseline data step produced no training data")
        # Optionally copy baseline data into run for reproducibility
        dataset_name = getattr(
            self.config.data, 'dataset_name',
            (self.config.data.datasets or ['heart_disease'])[0]
        )
        for split in ('train', 'val', 'test'):
            src = data_results.get('file_paths', {}).get(split)
            if src and Path(src).exists():
                (baseline_dir / 'data').mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, baseline_dir / 'data' / f'{dataset_name}_{split}.csv')

        # Step 2: Model
        self.logger.info("Step 2: Model (baseline training)")
        execution_time, best_model_name, test_metrics = self._train_and_evaluate(data_results)
        self.logger.info(f"Baseline training completed in {execution_time:.2f}s; best: {best_model_name}")

        # Save baseline artifacts: metrics to baseline/reports, models to baseline/models
        results = self._save_results(
            baseline_dir / 'reports',
            best_model_name,
            test_metrics,
            execution_time
        )
        (baseline_dir / 'models').mkdir(parents=True, exist_ok=True)
        for model_name in list(results['file_paths']['models'].keys()):
            src = Path(results['file_paths']['models'][model_name])
            dst = baseline_dir / 'models' / src.name
            if src != dst and src.exists():
                shutil.copy2(src, dst)
            results['file_paths']['models'][model_name] = dst

        # Step 3: Explainability (write to baseline dirs)
        orig_explanations = getattr(self.config.explainability, 'explanations_dir', None)
        orig_plots = getattr(self.config.explainability, 'plots_dir', None)
        try:
            self.config.explainability.explanations_dir = baseline_dir / 'explainability'
            self.config.explainability.plots_dir = baseline_dir / 'figures'
            self.logger.info("Step 3: Explainability (baseline)")
            X_train = data_results['train_df'].drop(columns=[data_results['target_column']])
            X_val = data_results['val_df'].drop(columns=[data_results['target_column']])
            y_train = data_results['train_df'][data_results['target_column']]
            y_val = data_results['val_df'][data_results['target_column']]
            feature_names = list(X_train.columns)
            explain_agent = ExplainabilityAgent(self.config.explainability)
            explain_agent.process(
                models=self.models,
                X_train=X_train,
                X_val=X_val,
                y_train=y_train,
                y_val=y_val,
                feature_names=feature_names,
                best_model_name=best_model_name,
            )
        finally:
            if orig_explanations is not None:
                self.config.explainability.explanations_dir = orig_explanations
            if orig_plots is not None:
                self.config.explainability.plots_dir = orig_plots

        total_time = time.time() - pipeline_start
        with open(baseline_log_dir / 'baseline.log', 'a') as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Baseline pipeline completed in {total_time:.2f}s; best_model={best_model_name}\n")
        self.logger.info(f"Standalone baseline pipeline completed in {total_time:.2f}s")
        return results

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='binary', zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.0,
            'average_precision': average_precision_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.0,
        }
        
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp'] = cm.ravel()
        
        return metrics
