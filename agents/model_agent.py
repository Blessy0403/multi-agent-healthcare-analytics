"""
Modeling Agent: Trains and evaluates multiple ML models for healthcare prediction.

This agent is responsible for:
- Training a thesis-grade model suite: Logistic Regression, Random Forest, XGBoost,
  SVM, Gradient Boosting, K-Nearest Neighbors
- Hyperparameter tuning via GridSearchCV
- Model evaluation on validation set
- Saving trained models and predictions
- Generating model performance metrics
"""

import pickle
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, classification_report
)
import xgboost as xgb

from utils.config import ModelConfig, get_config
from utils.json_utils import json_safe
from utils.logging import AgentLogger


class ModelAgent:
    """
    Agent responsible for model training and evaluation.
    
    Trains multiple models:
    - Logistic Regression
    - Random Forest
    - XGBoost
    
    Performs hyperparameter tuning and evaluates on validation set.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize Model Agent.
        
        Args:
            config: Model configuration (defaults to global config)
        """
        self.config = config or get_config().model
        self.logger = AgentLogger('model_agent')
        
        # Trained models storage
        self.models = {}
        self.best_params = {}
        self.predictions = {}
        self.metrics = {}
        # Selection reasoning for UI (Master's-level: why this model vs baseline)
        self.selection_reasoning = None

    def select_best_model(
        self,
        model_results: Dict[str, Dict[str, Any]],
        baseline_results: Optional[Dict[str, Any]] = None,
        n_features: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Master's-level logic: Compare multi-agent (engineered) results against standard baseline
        and store the reasoning for the dashboard.
        Selects by highest ROC-AUC to harmonize with the comparison table.
        """
        if not model_results:
            return None, {}
        best_model_name = max(
            model_results.keys(),
            key=lambda x: model_results[x].get("roc_auc") or model_results[x].get("roc_auc_score") or 0.0,
        )
        selected = model_results[best_model_name]
        best_roc = float(selected.get("roc_auc") or selected.get("roc_auc_score") or 0.0)

        # Baseline ROC-AUC: support standard_xgboost or selected_metrics
        baseline_roc = None
        if baseline_results:
            if isinstance(baseline_results.get("standard_xgboost"), dict):
                baseline_roc = baseline_results["standard_xgboost"].get("roc_auc") or baseline_results["standard_xgboost"].get("roc_auc_score")
            if baseline_roc is None and isinstance(baseline_results.get("selected_metrics"), dict):
                baseline_roc = baseline_results["selected_metrics"].get("roc_auc") or baseline_results["selected_metrics"].get("roc_auc_score")
        if baseline_roc is not None:
            baseline_roc = float(baseline_roc)
            improvement = best_roc - baseline_roc
            improvement_pct = improvement * 100.0
            feat_text = f"{n_features} engineered features and " if n_features is not None else ""
            self.selection_reasoning = (
                f"Selected {best_model_name} because it achieved the highest ROC-AUC ({best_roc:.4f}). "
                f"This represents a {improvement_pct:.2f}% improvement over the single-agent baseline by leveraging "
                f"the {feat_text}augmented dataset."
            )
        else:
            feat_text = f"{n_features} engineered features and " if n_features is not None else ""
            self.selection_reasoning = (
                f"Selected {best_model_name} because it achieved the highest ROC-AUC ({best_roc:.4f}). "
                f"Improvement over the single-agent baseline by leveraging the {feat_text}augmented dataset."
            )
        self.logger.info(f"Selection reasoning: {self.selection_reasoning}")
        return best_model_name, selected

    def prepare_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame, target_col: str) -> Tuple:
        """
        Prepare data for model training.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            target_col: Name of target column
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val)
        """
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_val = val_df.drop(columns=[target_col])
        y_val = val_df[target_col]
        
        return X_train, y_train, X_val, y_val

    def _scale_pos_weight(self, y):
        """
        scale_pos_weight = (#negative / #positive)
        For XGBoost class imbalance. Safe defaults if y has only one class.
        """
        import numpy as np
        y = np.asarray(y).ravel()
        # convert to 0/1 if possible
        try:
            y = y.astype(int)
        except Exception:
            pass
        pos = np.sum(y == 1)
        neg = np.sum(y == 0)
        if pos == 0:
            return 1.0
        return float(neg / pos)

    def train_logistic_regression(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Dict[str, Any]:
        """
        Train Logistic Regression model with hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Dictionary with model, predictions, and metrics
        """
        self.logger.info("Training Logistic Regression...")
        
        # class_weight='balanced' for imbalanced classification
        lr = LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced')
        grid_search = GridSearchCV(
            lr,
            self.config.lr_params,
            cv=self.config.cv_folds,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        self.logger.info(f"Best LR parameters: {best_params}")
        self.logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Predictions
        y_pred = best_model.predict(X_val)
        y_pred_proba = best_model.predict_proba(X_val)[:, 1]
        
        # Metrics
        metrics = self._compute_metrics(y_val, y_pred, y_pred_proba)
        
        result = {
            'model': best_model,
            'best_params': best_params,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'metrics': metrics
        }
        
        self.logger.info(f"LR Validation Accuracy: {metrics['accuracy']:.4f}")
        self.logger.info(f"LR Validation ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return result
    
    def train_random_forest(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Dict[str, Any]:
        """
        Train Random Forest model with hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Dictionary with model, predictions, and metrics
        """
        self.logger.info("Training Random Forest...")
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
        grid_search = GridSearchCV(
            rf,
            self.config.rf_params,
            cv=self.config.cv_folds,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        self.logger.info(f"Best RF parameters: {best_params}")
        self.logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Predictions
        y_pred = best_model.predict(X_val)
        y_pred_proba = best_model.predict_proba(X_val)[:, 1]
        
        # Metrics
        metrics = self._compute_metrics(y_val, y_pred, y_pred_proba)
        
        result = {
            'model': best_model,
            'best_params': best_params,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'metrics': metrics
        }
        
        self.logger.info(f"RF Validation Accuracy: {metrics['accuracy']:.4f}")
        self.logger.info(f"RF Validation ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return result
    
    def train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Dict[str, Any]:
        """
        Train XGBoost model with hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Dictionary with model, predictions, and metrics
        """
        self.logger.info("Training XGBoost...")
        
        # Grid search for hyperparameters
        xgb_model = xgb.XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )
        grid_search = GridSearchCV(
            xgb_model,
            self.config.xgb_params,
            cv=self.config.cv_folds,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        self.logger.info(f"Best XGB parameters: {best_params}")
        self.logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Predictions
        y_pred = best_model.predict(X_val)
        y_pred_proba = best_model.predict_proba(X_val)[:, 1]
        
        # Metrics
        metrics = self._compute_metrics(y_val, y_pred, y_pred_proba)
        
        result = {
            'model': best_model,
            'best_params': best_params,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'metrics': metrics
        }
        
        self.logger.info(f"XGB Validation Accuracy: {metrics['accuracy']:.4f}")
        self.logger.info(f"XGB Validation ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return result
    
    def train_svm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Dict[str, Any]:
        """Train Support Vector Machine with hyperparameter tuning (thesis-standard for healthcare ML)."""
        self.logger.info("Training SVM...")
        svm_model = SVC(probability=True, random_state=42, class_weight='balanced')
        grid_search = GridSearchCV(
            svm_model,
            self.config.svm_params,
            cv=self.config.cv_folds,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_val)
        y_pred_proba = best_model.predict_proba(X_val)[:, 1]
        metrics = self._compute_metrics(y_val, y_pred, y_pred_proba)
        self.logger.info(f"SVM Validation ROC-AUC: {metrics['roc_auc']:.4f}")
        return {
            'model': best_model,
            'best_params': grid_search.best_params_,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'metrics': metrics
        }
    
    def train_gradient_boosting(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Dict[str, Any]:
        """Train Gradient Boosting Classifier (sklearn) with hyperparameter tuning."""
        self.logger.info("Training Gradient Boosting...")
        gb_model = GradientBoostingClassifier(random_state=42)
        grid_search = GridSearchCV(
            gb_model,
            self.config.gb_params,
            cv=self.config.cv_folds,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_val)
        y_pred_proba = best_model.predict_proba(X_val)[:, 1]
        metrics = self._compute_metrics(y_val, y_pred, y_pred_proba)
        self.logger.info(f"Gradient Boosting Validation ROC-AUC: {metrics['roc_auc']:.4f}")
        return {
            'model': best_model,
            'best_params': grid_search.best_params_,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'metrics': metrics
        }
    
    def train_knn(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Dict[str, Any]:
        """Train K-Nearest Neighbors with hyperparameter tuning (interpretable baseline)."""
        self.logger.info("Training KNN...")
        knn_model = KNeighborsClassifier()
        grid_search = GridSearchCV(
            knn_model,
            self.config.knn_params,
            cv=self.config.cv_folds,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_val)
        y_pred_proba = best_model.predict_proba(X_val)[:, 1]
        metrics = self._compute_metrics(y_val, y_pred, y_pred_proba)
        self.logger.info(f"KNN Validation ROC-AUC: {metrics['roc_auc']:.4f}")
        return {
            'model': best_model,
            'best_params': grid_search.best_params_,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'metrics': metrics
        }
    
    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='binary', zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.0,
            'average_precision': average_precision_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.0,
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp'] = cm.ravel()
        
        return metrics

    def run(
        self,
        artifacts: Dict[str, Any],
        force_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Orchestrator entry: run model training using artifacts["features"] if present, else artifacts["data"].

        Args:
            artifacts: Must contain "data" or "features" with train_df, val_df, target_column.
            force_model: Optional override for selected model (e.g. from FeedbackAgent).

        Returns:
            Structured model artifacts (models, best_model_name, selected_metrics, file_paths, etc.).
        """
        data = artifacts.get("features") or artifacts.get("data") or {}
        train_df = data.get("train_df")
        val_df = data.get("val_df")
        target_col = data.get("target_column") or (data.get("metadata") or {}).get("target_column")
        if train_df is None or val_df is None or not target_col:
            raise ValueError("ModelAgent.run requires artifacts['data'] or artifacts['features'] with train_df, val_df, target_column.")
        return self.train_all_models(
            train_df=train_df,
            val_df=val_df,
            target_col=target_col,
            mode="multi_agent",
            force_model=force_model,
        )

    def train_all_models(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        target_col: str,
        mode: str = "multi_agent",
        force_model: Optional[str] = None,
        baseline_for_model_agent: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Train all configured models.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            target_col: Name of target column
            mode: "baseline" = select by ROC-AUC only; "multi_agent" = select by ROC-AUC + selection reasoning vs baseline
            force_model: If set, override selection to this model name (must be in trained models).
            baseline_for_model_agent: Optional baseline metrics for selection_reasoning (selected_metrics, best_model_name).
            
        Returns:
            Dictionary containing all trained models; metrics and selected_model refer to the chosen model only
        """
        with self.logger.execution_timer():
            # Prepare data
            X_train, y_train, X_val, y_val = self.prepare_data(
                train_df, val_df, target_col
            )
            n_features = len(X_train.columns) if X_train is not None else None
            results = {}
            
            # Train each model
            if 'logistic_regression' in self.config.models:
                results['logistic_regression'] = self.train_logistic_regression(
                    X_train, y_train, X_val, y_val
                )
                self.models['logistic_regression'] = results['logistic_regression']['model']
                self.best_params['logistic_regression'] = results['logistic_regression']['best_params']
                self.predictions['logistic_regression'] = results['logistic_regression']['predictions']
                self.metrics['logistic_regression'] = results['logistic_regression']['metrics']
            
            if 'random_forest' in self.config.models:
                results['random_forest'] = self.train_random_forest(
                    X_train, y_train, X_val, y_val
                )
                self.models['random_forest'] = results['random_forest']['model']
                self.best_params['random_forest'] = results['random_forest']['best_params']
                self.predictions['random_forest'] = results['random_forest']['predictions']
                self.metrics['random_forest'] = results['random_forest']['metrics']
            
            if 'xgboost' in self.config.models:
                results['xgboost'] = self.train_xgboost(
                    X_train, y_train, X_val, y_val
                )
                self.models['xgboost'] = results['xgboost']['model']
                self.best_params['xgboost'] = results['xgboost']['best_params']
                self.predictions['xgboost'] = results['xgboost']['predictions']
                self.metrics['xgboost'] = results['xgboost']['metrics']
            
            if 'svm' in self.config.models:
                results['svm'] = self.train_svm(X_train, y_train, X_val, y_val)
                self.models['svm'] = results['svm']['model']
                self.best_params['svm'] = results['svm']['best_params']
                self.predictions['svm'] = results['svm']['predictions']
                self.metrics['svm'] = results['svm']['metrics']
            
            if 'gradient_boosting' in self.config.models:
                results['gradient_boosting'] = self.train_gradient_boosting(
                    X_train, y_train, X_val, y_val
                )
                self.models['gradient_boosting'] = results['gradient_boosting']['model']
                self.best_params['gradient_boosting'] = results['gradient_boosting']['best_params']
                self.predictions['gradient_boosting'] = results['gradient_boosting']['predictions']
                self.metrics['gradient_boosting'] = results['gradient_boosting']['metrics']
            
            if 'knn' in self.config.models:
                results['knn'] = self.train_knn(X_train, y_train, X_val, y_val)
                self.models['knn'] = results['knn']['model']
                self.best_params['knn'] = results['knn']['best_params']
                self.predictions['knn'] = results['knn']['predictions']
                self.metrics['knn'] = results['knn']['metrics']
            
            # Select best model by mode (or force_model override). Multi-agent uses ROC-AUC + selection reasoning.
            if force_model is not None and force_model in self.models:
                best_model_name = force_model
                self.selection_reasoning = self.selection_reasoning or f"Selected {best_model_name} (forced by feedback/override)."
                self.logger.info(f"Best model (forced): {best_model_name}")
            elif mode == "baseline":
                best_model_name = max(
                    self.metrics.keys(),
                    key=lambda k: self.metrics[k].get('roc_auc') or self.metrics[k].get('roc_auc_score') or 0
                )
                self.logger.info(f"Best model (ROC-AUC): {best_model_name} (ROC-AUC: {self.metrics[best_model_name].get('roc_auc', 0):.4f})")
            else:
                # multi_agent: select by highest ROC-AUC and store selection reasoning vs baseline (dynamic benchmarking)
                best_model_name, _ = self.select_best_model(
                    self.metrics,
                    baseline_results=baseline_for_model_agent,
                    n_features=n_features,
                )
                if best_model_name is None:
                    best_model_name = max(
                        self.metrics.keys(),
                        key=lambda k: self.metrics[k].get('roc_auc') or self.metrics[k].get('roc_auc_score') or 0
                    )
                m = self.metrics[best_model_name]
                self.logger.info(f"Best model (ROC-AUC): {best_model_name} (ROC-AUC: {m.get('roc_auc', 0):.4f})")
            
            # Save models
            model_paths = self._save_models()
            
            # Save metrics (full metrics + selection_reasoning for dashboard)
            metrics_path = self._save_metrics(best_model_name=best_model_name)
            
            # Save predictions
            predictions_path = self._save_predictions(X_val, y_val)
            
            # Imbalance strategy used (for run_metadata / dashboard)
            scale_pw = self._scale_pos_weight(y_train)
            imbalance_strategy = {
                'class_weight_balanced': ['logistic_regression', 'random_forest', 'svm'],
                'scale_pos_weight_xgb': scale_pw if 'xgboost' in self.models else None,
                'stratified_split': True,
            }
            
            result = {
                'models': self.models,
                'best_model_name': best_model_name,
                'selected_model': best_model_name,
                'selected_metrics': self.metrics[best_model_name],
                'selection_reasoning': self.selection_reasoning,
                'all_models': self.metrics,
                'best_model': self.models[best_model_name],
                'predictions': self.predictions,
                'best_params': self.best_params,
                'file_paths': {
                    'models': model_paths,
                    'metrics': metrics_path,
                    'predictions': predictions_path
                },
                'feature_names': list(X_train.columns),
                'imbalance_strategy': imbalance_strategy,
            }
            
            self.logger.info("Model training pipeline completed successfully")
            
            return result
    
    def _save_models(self) -> Dict[str, Path]:
        """Save all trained models to disk."""
        model_paths = {}
        
        for model_name, model in self.models.items():
            model_path = self.config.models_dir / f'{model_name}.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            model_paths[model_name] = model_path
            self.logger.log_artifact('model', str(model_path), {'model_type': model_name})
        
        return model_paths
    
    def _save_metrics(self, best_model_name: Optional[str] = None) -> Path:
        """Save model metrics to JSON (including selection_reasoning for dashboard)."""
        metrics_path = self.config.models_dir / 'model_metrics.json'
        
        # Convert numpy types to native Python types for JSON serialization
        metrics_serializable = {}
        for model_name, metrics in self.metrics.items():
            metrics_serializable[model_name] = {
                k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                for k, v in metrics.items()
            }
        if self.selection_reasoning is not None:
            metrics_serializable["selection_reasoning"] = self.selection_reasoning
        if best_model_name is not None and best_model_name in self.metrics:
            m = self.metrics[best_model_name]
            metrics_serializable["selected_roc_auc"] = float(m.get("roc_auc") or m.get("roc_auc_score") or 0)
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_serializable, f, indent=2, default=json_safe)
        
        self.logger.log_artifact('metrics', str(metrics_path))
        
        return metrics_path
    
    def _save_predictions(self, X_val: pd.DataFrame, y_val: pd.Series) -> Path:
        """Save validation predictions to CSV."""
        predictions_path = self.config.models_dir / 'validation_predictions.csv'
        
        pred_df = pd.DataFrame({
            'true_label': y_val.values
        })
        
        for model_name, preds in self.predictions.items():
            pred_df[f'{model_name}_pred'] = preds
        
        pred_df.to_csv(predictions_path, index=False)
        
        self.logger.log_artifact('predictions', str(predictions_path))
        
        return predictions_path
