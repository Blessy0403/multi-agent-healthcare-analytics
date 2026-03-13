"""
Explainability Agent: Generates model explanations using SHAP and LIME.

This agent is responsible for:
- SHAP global and local explanations
- LIME local explanations
- Natural language explanation generation
- Saving explanation plots and structured data
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import shap
import lime
import lime.lime_tabular
from matplotlib import pyplot as plt
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

from utils.config import ExplainabilityConfig, get_config
from utils.json_utils import json_safe
from utils.logging import AgentLogger

# Signature features per dataset: used to infer actual dataset from feature names so SHAP filenames match plot content
_DATASET_SIGNATURE_FEATURES = {
    "heart_disease": {"thalach", "thal", "cp", "ca", "exang", "oldpeak", "slope", "restecg", "trestbps"},
    "diabetes": {"pregnancies", "glucose", "blood_pressure", "skin_thickness", "insulin", "bmi", "diabetes_pedigree"},
}


def _infer_dataset_slug_from_features(feature_names: List[str]) -> Optional[str]:
    """Infer dataset from feature names so SHAP file prefix always matches plot content (avoids cross-dataset leakage)."""
    if not feature_names:
        return None
    names_lower = {str(n).strip().lower() for n in feature_names}
    for slug, sig in _DATASET_SIGNATURE_FEATURES.items():
        if sig & names_lower:  # at least one signature feature present
            return slug
    return None


class ExplainabilityAgent:
    """
    Agent responsible for generating model explanations.
    
    Uses:
    - SHAP for global and local feature importance
    - LIME for local interpretability
    - Natural language generation for human-readable summaries
    """
    
    def __init__(self, config: Optional[ExplainabilityConfig] = None):
        """
        Initialize Explainability Agent.
        
        Args:
            config: Explainability configuration (defaults to global config)
        """
        self.config = config or get_config().explainability
        self.logger = AgentLogger('explainability_agent')
        
        # Storage for explanations
        self.shap_explanations = {}
        self.lime_explanations = {}
        self.natural_language_explanations = {}
        
    def _get_binary_positive_shap(self, shap_values):
        """
        Normalize SHAP output to (n_samples, n_features) for binary classification.
        Handles list output, 3D output, and (n_features, 2) per-instance style.
        """
        # Case 1: list of arrays (older API)
        if isinstance(shap_values, list):
            if len(shap_values) == 2:
                return shap_values[1]
            return shap_values[0]

        arr = np.array(shap_values)

        # Case 2: 3D (n_samples, n_features, n_outputs)
        if arr.ndim == 3:
            if arr.shape[-1] == 2:
                return arr[:, :, 1]
            return arr[:, :, 0]
        
        # Case 3: 2D but last dim is outputs (n_features, 2) for one instance
        if arr.ndim == 2 and arr.shape[-1] == 2:
            return arr[:, 1]

        return arr
    
    def explain_with_shap(
        self,
        model: Any,
        X_train: pd.DataFrame,
        X_explain: pd.DataFrame,
        model_name: str,
        model_type: str = 'auto'
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanations for a model.
        
        Args:
            model: Trained model
            X_train: Training data (for background)
            X_explain: Data to explain
            model_name: Name of the model
            model_type: Type of model ('tree', 'linear', 'kernel', 'auto')
            
        Returns:
            Dictionary containing SHAP values and explainer
        """
        self.logger.info(f"Generating SHAP explanations for {model_name}...")
        
        # Sample background data for efficiency
        if len(X_train) > self.config.shap_sample_size:
            background = X_train.sample(
                n=self.config.shap_sample_size,
                random_state=42
            )
        else:
            background = X_train
        
        # Select explainer based on model type
        if model_type == 'auto':
            # Auto-detect model type
            if isinstance(model, (xgb.XGBClassifier,)):
                model_type = 'tree'
            elif hasattr(model, 'tree_') or hasattr(model, 'estimators_'):
                model_type = 'tree'
            else:
                model_type = 'kernel'
        
        shap_values = None
        explainer = None
        x_explain_used = None  # When KernelExplainer caps instances, we store the subset for plots/NL
        explained_indices = None  # Row indices in original X_explain that were used for SHAP (when capped)

        try:
            if model_type == "tree":
                # Try fast TreeExplainer first (best for RF / XGB when it works)
                if hasattr(model, "get_booster"):
                    explainer = shap.TreeExplainer(model.get_booster())
                else:
                    explainer = shap.TreeExplainer(model)

                shap_values = explainer.shap_values(X_explain)

            elif model_type == "linear":
                # LinearExplainer needs background data
                background = shap.sample(X_explain, 50, random_state=42)
                explainer = shap.LinearExplainer(model, background)
                shap_values = explainer.shap_values(X_explain)
            
            else:
                # Non-tree models (e.g., logistic regression)
                explainer = shap.Explainer(model, X_explain)
                shap_values = explainer(X_explain).values
                
        except Exception as e:
            # -------- 2) Fallback: KernelExplainer (slow but universal) --------
            self.logger.warning(f"Fast SHAP explainer failed: {e}. Falling back to KernelExplainer.")
            explain_n = getattr(self.config, 'explain_n', None) or getattr(self.config, 'shap_max_explain_instances', 200)
            nsamples = getattr(self.config, 'shap_kernel_nsamples', 50)
            total_rows = len(X_explain)
            if total_rows > explain_n:
                X_explain = X_explain.sample(n=explain_n, random_state=42)
                x_explain_used = X_explain
                # Store exact row indices from original X used for SHAP (for fidelity and plots)
                explained_indices = X_explain.index.tolist()
                self.logger.info(
                    f"KernelExplainer: total eval rows={total_rows}, explain_n={explain_n}, "
                    f"actual explained count={len(explained_indices)}, nsamples={nsamples}"
                )

            # KernelExplainer needs a predict function that outputs probability for positive class
            def predict_fn(x):
                # x might be numpy array; make sure model can handle it
                proba = model.predict_proba(x)
                return proba[:, 1]

            background = shap.sample(X_explain, min(50, len(X_explain)), random_state=42)
            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(X_explain, nsamples=nsamples)

        # If SHAP returns [class0, class1], use positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Final sanity check
        if shap_values is None:
            raise RuntimeError("SHAP failed: shap_values is None after explainer attempts.")

        self.logger.info(f"SHAP values computed for {model_name}")
                
            
            # Store explanations
            
    # Store explanations
        explanation = {
            'shap_values': shap_values,
            'explainer': explainer,
            'feature_names': list(X_explain.columns),
            'expected_value': explainer.expected_value
            if hasattr(explainer, 'expected_value') else None
        }
        if x_explain_used is not None:
            explanation['X_explain_used'] = x_explain_used
        if explained_indices is not None:
            explanation['explained_indices'] = explained_indices

        self.shap_explanations[model_name] = explanation
        return explanation

       

    
    def explain_with_lime(
        self,
        model: Any,
        X_train: pd.DataFrame,
        X_explain: pd.DataFrame,
        y_train: pd.Series,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Generate LIME explanations for a model.
        
        Args:
            model: Trained model
            X_train: Training data
            X_explain: Data to explain
            y_train: Training labels
            model_name: Name of the model
            
        Returns:
            Dictionary containing LIME explanations
        """
        self.logger.info(f"Generating LIME explanations for {model_name}...")
        
        try:
            # Create LIME explainer
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train.values,
                feature_names=list(X_train.columns),
                class_names=['No Disease', 'Disease'],
                mode='classification',
                discretize_continuous=True
            )
            
            # Generate explanations for sample instances
            num_samples = min(self.config.shap_plot_samples, len(X_explain))
            sample_indices = np.random.choice(
                len(X_explain),
                size=num_samples,
                replace=False
            )
            
            lime_explanations = {}
            
            for idx in sample_indices:
                instance = X_explain.iloc[idx].values
                explanation = explainer.explain_instance(
                    instance,
                    model.predict_proba,
                    num_features=self.config.lime_num_features
                )
                
                lime_explanations[int(idx)] = {
                    'explanation': explanation,
                    'instance': instance.tolist(),
                    'prediction': model.predict_proba([instance])[0].tolist()
                }
            
            self.logger.info(f"LIME explanations computed for {model_name} ({num_samples} instances)")
            
            result = {
                'explainer': explainer,
                'explanations': lime_explanations,
                'feature_names': list(X_explain.columns)
            }
            
            self.lime_explanations[model_name] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating LIME explanations: {e}")
            return {}
    
    def generate_shap_plots(
        self,
        model_name: str,
        shap_explanation: Dict[str, Any],
        X_explain: pd.DataFrame,
        max_explainable_index: Optional[int] = None,
        dataset_slug: str = "dataset",
    ) -> Dict[str, Path]:
        """
        Generate and save SHAP plots. Filenames include dataset and model keys so the dashboard
        can load the correct artifact: {dataset_key}_{model_key}_shap_summary.png.
        
        Args:
            model_name: Name of the model (e.g. 'gradient_boosting' or 'Gradient Boosting')
            shap_explanation: SHAP explanation dictionary
            X_explain: Data that was explained
            max_explainable_index: Max valid index into shap_values (e.g. len(shap_values)-1 when capped). Optional.
            dataset_slug: Dataset identifier for filenames (e.g. 'diabetes', 'heart_disease').
            
        Returns:
            Dictionary of plot file paths
        """
        # Canonical figures directory: run_dir / multi_agent / figures (set by pipeline runner)
        figures_dir = Path(self.config.plots_dir) if self.config.plots_dir else None
        if figures_dir is None:
            self.logger.warning("plots_dir not set; SHAP summary will not be saved")
        else:
            figures_dir.mkdir(parents=True, exist_ok=True)
        # Normalize model name to key for filenames (dashboard expects e.g. diabetes_gradient_boosting_shap_summary.png)
        model_key = str(model_name).lower().replace(" ", "_").replace("-", "_")
        
        self.logger.info(f"Generating SHAP plots for {model_name} (dataset: {dataset_slug})...")
        
        plot_paths = {}
        shap_values = self._get_binary_positive_shap(shap_explanation['shap_values'])

        
        # Ensure shap_values is 2D
        if len(shap_values.shape) == 1:
            shap_values = shap_values.reshape(1, -1)
        n_shap = shap_values.shape[0]
        if max_explainable_index is None:
            max_explainable_index = n_shap - 1
        else:
            max_explainable_index = min(max_explainable_index, n_shap - 1)
        # Align X to SHAP row count so summary plot can always be generated (required for dashboard)
        X_explain_used = shap_explanation.get("X_explain_used")
        if X_explain_used is None or len(X_explain_used) != n_shap:
            if X_explain is not None and len(X_explain) >= n_shap:
                X_explain_used = X_explain.iloc[:n_shap]
            else:
                X_explain_used = None
        if X_explain_used is None or len(X_explain_used) != n_shap:
            self.logger.warning(f"SHAP summary: no X with {n_shap} rows (X_explain: {len(X_explain) if X_explain is not None else 0})")
        X_explain = X_explain_used  # use aligned X for summary and waterfall below
        
        # Summary plot (beeswarm): always generate and save when we have SHAP values (dashboard requires this file)
        if figures_dir is not None:
            figures_dir.mkdir(parents=True, exist_ok=True)
            summary_filename = f"{dataset_slug}_{model_key}_shap_summary.png"
            summary_path = figures_dir / summary_filename
            self.logger.info(f"SHAP summary filename (saving): {summary_filename} -> {summary_path}")
            summary_saved = False
            try:
                if X_explain is not None and len(X_explain) == n_shap:
                    plt.figure(figsize=(14, 8), dpi=200)
                    shap.summary_plot(
                        shap_values,
                        X_explain,
                        show=False,
                        max_display=12,
                        feature_names=shap_explanation['feature_names'],
                    )
                    plt.tight_layout()
                    plt.savefig(summary_path, bbox_inches="tight", dpi=200)
                    plt.close()
                    plot_paths['summary'] = summary_path
                    summary_saved = True
                    self.logger.info(f"Saved SHAP summary plot -> {summary_path}")
                    self.logger.log_artifact('plot', str(summary_path), {'type': 'shap_summary'})
                else:
                    # Fallback: bar of mean |SHAP| so dashboard always gets an artifact
                    plt.figure(figsize=(14, 8), dpi=200)
                    mean_abs = np.abs(shap_values).mean(axis=0)
                    feat_names = shap_explanation['feature_names']
                    n_f = min(len(feat_names), len(mean_abs))
                    order = np.argsort(mean_abs)[::-1][:min(12, n_f)]
                    plt.barh([feat_names[i] for i in order], mean_abs[order])
                    plt.xlabel("Mean |SHAP value|")
                    plt.tight_layout()
                    plt.savefig(summary_path, bbox_inches="tight", dpi=200)
                    plt.close()
                    plot_paths['summary'] = summary_path
                    summary_saved = True
                    self.logger.info(f"Saved SHAP summary plot (bar fallback) -> {summary_path}")
                    self.logger.log_artifact('plot', str(summary_path), {'type': 'shap_summary'})
            except Exception as e:
                self.logger.warning(f"Could not generate SHAP summary plot: {e}")
                try:
                    plt.close()
                except Exception:
                    pass
            if not summary_saved:
                self.logger.warning("SHAP summary plot was not saved; dashboard may show no explanation.")
        
        # Bar plot (mean absolute SHAP values)
        if figures_dir is not None:
            try:
                plt.figure(figsize=(10, 8))
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                plt.barh(shap_explanation['feature_names'], mean_abs_shap)
                plt.xlabel("Mean |SHAP value|")
                plt.tight_layout()
                bar_filename = f"{dataset_slug}_{model_key}_shap_bar.png"
                bar_path = figures_dir / bar_filename
                self.logger.info(f"SHAP bar filename (saving): {bar_filename} -> {bar_path}")
                plt.savefig(bar_path, bbox_inches='tight', dpi=150)
                plt.close()
                plot_paths['bar'] = bar_path
                self.logger.log_artifact('plot', str(bar_path), {'type': 'shap_bar'})
            except Exception as e:
                self.logger.warning(f"Could not generate SHAP bar plot: {e}")
                try:
                    plt.close()
                except Exception:
                    pass
        
        # Waterfall plots: sample ONLY from range(len(shap_values)) for safe bounded indexing
        if figures_dir is not None:
            pool_size = max_explainable_index + 1
            num_plots = min(self.config.shap_plot_samples, pool_size)
            if X_explain is not None and len(X_explain) >= pool_size:
                sample_indices = np.random.choice(pool_size, size=num_plots, replace=False)
            else:
                sample_indices = np.array([], dtype=int)

            sv_2d = self._get_binary_positive_shap(shap_values)
            ev = shap_explanation.get("expected_value", 0)
            if isinstance(ev, (list, tuple, np.ndarray)):
                base_value = float(np.array(ev).reshape(-1)[-1])
            else:
                base_value = float(ev)

            for idx in sample_indices:
                try:
                    idx_safe = min(int(idx), max_explainable_index)
                    if idx_safe < 0 or idx_safe >= n_shap:
                        self.logger.warning(f"Skipping waterfall plot: index {idx_safe} out of bounds for shap_values rows {n_shap}")
                        continue
                    plt.figure(figsize=(10, 6))
                    sv_i = np.array(sv_2d[idx_safe]).reshape(-1)
                    exp = shap.Explanation(
                        values=sv_i,
                        base_values=base_value,
                        data=X_explain.iloc[idx_safe].values,
                        feature_names=shap_explanation["feature_names"],
                    )
                    shap.plots.waterfall(exp, show=False)
                    waterfall_path = figures_dir / f"{dataset_slug}_{model_key}_shap_waterfall_{idx_safe}.png"
                    plt.savefig(waterfall_path, bbox_inches="tight", dpi=150)
                    plt.close()
                    plot_paths[f"waterfall_{idx_safe}"] = waterfall_path
                except Exception as e:
                    self.logger.warning(f"Could not generate waterfall plot for instance {idx}: {e}")
                    try:
                        plt.close()
                    except Exception:
                        pass

        
        self.logger.info(f"Generated {len(plot_paths)} SHAP plots for {model_name}")
        if plot_paths:
            for k, p in plot_paths.items():
                self.logger.info(f"  SHAP plot saved: {k} -> {p}")
        
        return plot_paths
    
    def generate_natural_language_explanation(
        self,
        model_name: str,
        shap_explanation: Dict[str, Any],
        prediction: float,
        instance_idx: int,
        X_explain: pd.DataFrame,
        feature_names: List[str]
    ) -> Optional[str]:
        """
        Generate natural language explanation from SHAP values.
        
        Args:
            model_name: Name of the model
            shap_explanation: SHAP explanation dictionary
            prediction: Predicted probability
            instance_idx: Index of the instance
            X_explain: Explained data
            feature_names: List of feature names
            
        Returns:
            Natural language explanation string, or None if instance_idx is out of bounds.
        """
        shap_values = shap_explanation['shap_values']
        n_shap = shap_values.shape[0] if hasattr(shap_values, "shape") and len(shap_values.shape) >= 1 else len(shap_values)
        if instance_idx >= n_shap:
            return None
        instance_idx = min(int(instance_idx), n_shap - 1)

        # Get SHAP values for this instance
        if len(shap_values.shape) > 1:
            instance_shap = shap_values[instance_idx]
            # Handle multi-class output (take positive class)
            if isinstance(instance_shap, np.ndarray) and len(instance_shap.shape) > 1:
                instance_shap = instance_shap[1] if instance_shap.shape[0] > 1 else instance_shap[0]
        else:
            instance_shap = shap_values
            if isinstance(instance_shap, np.ndarray) and len(instance_shap.shape) > 1:
                instance_shap = instance_shap[1] if instance_shap.shape[0] > 1 else instance_shap[0]
        
        # Ensure instance_shap is 1D
        if isinstance(instance_shap, np.ndarray):
            instance_shap = instance_shap.flatten()
        
        # Get feature values
        instance_values = X_explain.iloc[instance_idx].values
        if isinstance(instance_values, np.ndarray):
            instance_values = instance_values.flatten()
        
        # Sort features by absolute SHAP value
        feature_importance = list(zip(feature_names, instance_shap, instance_values))
        feature_importance.sort(key=lambda x: abs(float(x[1])), reverse=True)
        
        # Generate explanation
        prediction_class = "Disease" if prediction > 0.5 else "No Disease"
        confidence = abs(prediction - 0.5) * 2 * 100  # Convert to percentage
        
        explanation_parts = [
            f"The {model_name} model predicts **{prediction_class}** "
            f"with {confidence:.1f}% confidence (probability: {prediction:.3f})."
        ]
        
        # Top contributing features
        top_features = feature_importance[:5]
        
        explanation_parts.append("\n**Key contributing factors:**")
        
        for i, (feature, shap_val, feature_val) in enumerate(top_features, 1):
            direction = "increased" if shap_val > 0 else "decreased"
            impact_pct = abs(shap_val) * 100
            
            # Normalize feature value for readability (if it's scaled)
            if abs(feature_val) < 3:  # Likely scaled
                feature_desc = f"{feature} (value: {feature_val:.2f})"
            else:
                feature_desc = f"{feature} (value: {feature_val:.2f})"
            
            explanation_parts.append(
                f"{i}. {feature_desc} {direction} the risk by approximately {impact_pct:.1f}%."
            )
        
        return "\n".join(explanation_parts)

    def run(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestrator entry: run explainability using artifacts["features"] or artifacts["data"] and artifacts["models"].

        Returns:
            Structured explainability artifacts (shap_explanations, lime_explanations, plot_paths, etc.).
        """
        data = artifacts.get("features") or artifacts.get("data") or {}
        model_results = artifacts.get("models") or {}
        models = model_results.get("models") or {}
        target_col = data.get("target_column") or (data.get("metadata") or {}).get("target_column")
        if not target_col:
            target_col = data.get("target_column")
        train_df = data.get("train_df")
        val_df = data.get("val_df")
        if train_df is None or val_df is None or not target_col or not models:
            raise ValueError("ExplainabilityAgent.run requires artifacts['data'] or artifacts['features'] and artifacts['models'].")
        X_train = train_df.drop(columns=[target_col])
        X_val = val_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        y_val = val_df[target_col]
        feature_names = list(model_results.get("feature_names") or X_train.columns)
        best_model_name = model_results.get("best_model_name") or model_results.get("selected_model")
        metadata = data.get("metadata") or {}
        dataset_name = (metadata.get("dataset_name") or "dataset").strip()
        dataset_slug = str(dataset_name).lower().replace(" ", "_").replace("-", "_")
        if dataset_slug == "dataset" or not dataset_slug:
            try:
                from utils.config import get_config
                cfg = get_config()
                if getattr(cfg, "data", None):
                    dataset_slug = (getattr(cfg.data, "dataset_name", None) or (cfg.data.datasets[0] if getattr(cfg.data, "datasets", None) else None)) or "dataset"
                    if dataset_slug:
                        dataset_slug = str(dataset_slug).lower().replace(" ", "_").replace("-", "_")
            except Exception:
                pass
            if not dataset_slug or dataset_slug == "dataset":
                dataset_slug = "diabetes"
        # Override with inference from actual feature names so SHAP filename always matches plot content (no cross-dataset leakage)
        inferred = _infer_dataset_slug_from_features(feature_names)
        if inferred:
            dataset_slug = inferred
            self.logger.info(f"Using dataset slug from features for SHAP filenames: {dataset_slug}")
        return self.process(
            models=models,
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val,
            feature_names=feature_names,
            best_model_name=best_model_name,
            dataset_slug=dataset_slug,
        )

    def process(
        self,
        models: Dict[str, Any],
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        feature_names: List[str],
        best_model_name: str,
        dataset_slug: str = "dataset",
    ) -> Dict[str, Any]:
        """
        Execute full explainability pipeline.
        
        Args:
            models: Dictionary of trained models
            X_train: Training features
            X_val: Validation features
            y_train: Training labels
            y_val: Validation labels
            feature_names: List of feature names
            best_model_name: Name of the best performing model
            
        Returns:
            Dictionary containing all explanations and file paths
        """
        with self.logger.execution_timer():
            results = {
                "shap_explanations": {},
                "lime_explanations": {},
                "natural_language": {},
                "plot_paths": {},
                "explanation_paths": {},
            }

            # Explain each model
            for model_name, model in models.items():
                self.logger.info(f"Explaining model: {model_name}")

                # SHAP explanations
                shap_explanation = {}
                if self.config.use_shap:
                    shap_explanation = self.explain_with_shap(
                        model=model,
                        X_train=X_train,
                        X_explain=X_val,
                        model_name=model_name,
                        model_type="auto",
                    )

                    if shap_explanation:
                        results["shap_explanations"][model_name] = shap_explanation
                        # Safe bound for all downstream indexing (summary, waterfall, natural language)
                        sv = shap_explanation["shap_values"]
                        max_explainable_index = (sv.shape[0] - 1) if hasattr(sv, "shape") and len(sv.shape) >= 1 and sv.shape[0] > 0 else 0

                        # Generate plots (use subset when KernelExplainer capped instances)
                        X_for_shap = shap_explanation.get("X_explain_used")
                        if X_for_shap is None:
                            X_for_shap = X_val
                        plot_paths = self.generate_shap_plots(
                            model_name=model_name,
                            shap_explanation=shap_explanation,
                            X_explain=X_for_shap,
                            max_explainable_index=max_explainable_index,
                            dataset_slug=dataset_slug,
                        )
                        results["plot_paths"][model_name] = plot_paths

                # LIME explanations
                if self.config.use_lime:
                    lime_explanation = self.explain_with_lime(
                        model=model,
                        X_train=X_train,
                        X_explain=X_val,
                        y_train=y_train,
                        model_name=model_name,
                    )
                    if lime_explanation:
                        results["lime_explanations"][model_name] = lime_explanation

                # Natural language explanations for best model (sample ONLY from explainable range)
                if model_name == best_model_name and self.config.use_shap and shap_explanation:
                    sv = shap_explanation["shap_values"]
                    max_explainable_index = (sv.shape[0] - 1) if hasattr(sv, "shape") and len(sv.shape) >= 1 and sv.shape[0] > 0 else 0
                    n_explainable = max_explainable_index + 1
                    X_for_nl = shap_explanation.get("X_explain_used")
                    if X_for_nl is None:
                        X_for_nl = X_val
                    # Ensure we only sample indices that exist in shap_values
                    num_explanations = min(10, n_explainable)
                    sample_indices = np.random.choice(n_explainable, size=num_explanations, replace=False)
                    # Use data aligned with SHAP (subset when capped)
                    if len(X_for_nl) > n_explainable:
                        X_for_nl = X_for_nl.iloc[:n_explainable]

                    nl_explanations = {}
                    for idx in sample_indices:
                        idx_safe = min(int(idx), max_explainable_index)
                        prediction = model.predict_proba(X_for_nl.iloc[[idx_safe]])[0][1]
                        nl_explanation = self.generate_natural_language_explanation(
                            model_name=model_name,
                            shap_explanation=shap_explanation,
                            prediction=prediction,
                            instance_idx=idx_safe,
                            X_explain=X_for_nl,
                            feature_names=feature_names,
                        )
                        if nl_explanation is not None:
                            nl_explanations[int(idx_safe)] = nl_explanation

                    results["natural_language"][model_name] = nl_explanations

            # Save structured explanations
            explanations_path = self.config.explanations_dir / "explanations.json"

            explanations_serializable = {
                "shap": {},
                "lime": {},
                "natural_language": results["natural_language"],
            }

            # Save SHAP summary stats and metadata (explained_indices for evaluator)
            shap_metadata = {}
            for model_name, shap_exp in results["shap_explanations"].items():
                shap_values = shap_exp["shap_values"]
                entry = {
                    "mean_abs_shap": np.abs(shap_values).mean(axis=0).tolist(),
                    "feature_names": shap_exp["feature_names"],
                }
                if shap_exp.get("explained_indices") is not None:
                    entry["explained_indices"] = shap_exp["explained_indices"]
                explanations_serializable["shap"][model_name] = entry
                shap_metadata[model_name] = {
                    "shap_rows": int(shap_values.shape[0]) if hasattr(shap_values, "shape") else 0,
                    "explained_indices": shap_exp.get("explained_indices"),
                }

            # Save LIME metadata and full LIME explanations (JSON-serializable) per model
            for model_name, lime_exp in results["lime_explanations"].items():
                explanations_serializable["lime"][model_name] = {
                    "num_explanations": len(lime_exp["explanations"]),
                    "feature_names": lime_exp["feature_names"],
                }
                # Save full LIME explanations for dashboard (feature weights via as_list())
                lime_export = {
                    "feature_names": lime_exp["feature_names"],
                    "explanations": {},
                }
                for idx, data in lime_exp["explanations"].items():
                    exp_obj = data.get("explanation")
                    try:
                        feature_weights = exp_obj.as_list() if exp_obj and hasattr(exp_obj, "as_list") else []
                    except Exception:
                        feature_weights = []
                    lime_export["explanations"][str(idx)] = {
                        "instance": data.get("instance", []),
                        "prediction": data.get("prediction", []),
                        "feature_weights": [[str(f), float(w)] for f, w in feature_weights],
                    }
                lime_path = self.config.explanations_dir / f"lime_explanations_{model_name}.json"
                with open(lime_path, "w") as f:
                    json.dump(lime_export, f, indent=2, default=json_safe)
                self.logger.log_artifact("lime", str(lime_path))

            with open(explanations_path, "w") as f:
                json.dump(explanations_serializable, f, indent=2, default=json_safe)

            results["explanation_paths"]["structured"] = explanations_path
            self.logger.log_artifact("explanation", str(explanations_path))

            # Save explainability metadata (explained_indices per model for evaluator)
            metadata_path = self.config.explanations_dir / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(shap_metadata, f, indent=2, default=json_safe)
            self.logger.log_artifact("explanation", str(metadata_path))

            # Save natural language explanations
            nl_path = self.config.explanations_dir / "natural_language_explanations.txt"
            with open(nl_path, "w") as f:
                for model_name, explanations in results["natural_language"].items():
                    f.write("\n" + "=" * 80 + "\n")
                    f.write(f"Model: {model_name}\n")
                    f.write("=" * 80 + "\n\n")
                    for idx, explanation in explanations.items():
                        f.write(f"Instance {idx}:\n")
                        f.write(f"{explanation}\n\n")

            results["explanation_paths"]["natural_language"] = nl_path
            self.logger.log_artifact("explanation", str(nl_path))

            self.logger.info("Explainability pipeline completed successfully")
            return results
