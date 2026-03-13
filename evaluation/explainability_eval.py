"""
Explainability Evaluation: Assesses quality and stability of model explanations.

This module provides:
- Explanation fidelity measurement
- SHAP value stability analysis
- Qualitative readability scoring
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from scipy.stats import spearmanr

from utils.config import get_config
from utils.json_utils import json_safe
from utils.logging import AgentLogger


def _align_X_to_model(
    model: Any,
    X: Union[pd.DataFrame, np.ndarray],
    expected_features: Optional[List[str]] = None,
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Align X to the feature names expected by the model (e.g. after feature engineering).
    Returns a DataFrame with columns in the order expected by the model, missing columns filled with 0.
    If X is numpy and no column mapping is available, returns X unchanged.
    """
    # 1) Get expected feature names
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
    elif expected_features is not None and len(expected_features) > 0:
        expected = list(expected_features)
    else:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        return X

    # 2) Align X
    if isinstance(X, pd.DataFrame):
        X_aligned = X.reindex(columns=expected, fill_value=0.0)
        # Ensure numeric
        for c in X_aligned.columns:
            if not np.issubdtype(X_aligned[c].dtype, np.number):
                X_aligned[c] = pd.to_numeric(X_aligned[c], errors="coerce").fillna(0.0)
        return X_aligned
    # numpy: skip alignment (no column->index mapping in this module)
    return X


class ExplainabilityEvaluator:
    """
    Evaluator for explainability quality.
    
    Measures:
    - Fidelity of explanations
    - Stability of SHAP values
    - Readability of natural language explanations
    """
    
    def __init__(self):
        """Initialize explainability evaluator."""
        self.config = get_config()
        self.logger = AgentLogger('explainability_evaluator')
    
    def evaluate_shap_stability(
        self,
        shap_explanations: Dict[str, Any],
        num_samples: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate stability of SHAP values.
        
        Stability is measured as the consistency of feature importance
        rankings across different samples.
        
        Args:
            shap_explanations: SHAP explanation dictionary
            num_samples: Number of samples to use for stability test
            
        Returns:
            Dictionary with stability metrics
        """
        self.logger.info("Evaluating SHAP stability...")
        
        stability_metrics = {}
        
        for model_name, explanation in shap_explanations.items():
            shap_values = explanation['shap_values']
            feature_names = explanation['feature_names']
            
            # Compute mean absolute SHAP values per feature
            # Handle multi-dimensional SHAP values
            if len(shap_values.shape) > 2:
                shap_values = shap_values.reshape(shap_values.shape[0], -1)
            elif len(shap_values.shape) == 2 and shap_values.shape[1] != len(feature_names):
                # Take first N features if shape doesn't match
                shap_values = shap_values[:, :len(feature_names)]
            
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            
            # Rank features by importance
            feature_ranks = np.argsort(mean_abs_shap)[::-1]
            feature_ranks = [int(i) for i in feature_ranks]  # Convert to list of ints
            
            # Sample different subsets and check rank consistency
            # Handle multi-dimensional SHAP values
            if len(shap_values.shape) > 2:
                # If 3D, take first class or flatten
                shap_values = shap_values[:, :, 0] if shap_values.shape[2] > 1 else shap_values[:, :, 0]
            elif len(shap_values.shape) == 2 and shap_values.shape[1] > len(feature_names):
                # If shape doesn't match, take first N features
                shap_values = shap_values[:, :len(feature_names)]
            
            if len(shap_values) > num_samples:
                sample_indices = np.random.choice(
                    len(shap_values),
                    size=min(num_samples, len(shap_values)),
                    replace=False
                )
                
                sample_ranks = []
                for idx in sample_indices:
                    sample_shap = np.abs(shap_values[int(idx)])
                    # Ensure we have the right shape
                    if len(sample_shap.shape) > 1:
                        sample_shap = sample_shap.flatten()[:len(feature_names)]
                    sample_rank = np.argsort(sample_shap)[::-1]
                    sample_ranks.append(sample_rank)
                
                # Compute rank correlation between samples
                rank_correlations = []
                for i in range(len(sample_ranks)):
                    for j in range(i + 1, len(sample_ranks)):
                        try:
                            corr_result = spearmanr(sample_ranks[i], sample_ranks[j])
                            # Handle both tuple and single value returns
                            if isinstance(corr_result, tuple):
                                corr = corr_result[0]
                            else:
                                corr = corr_result
                            
                            # Convert to scalar if array
                            if isinstance(corr, np.ndarray):
                                if corr.size == 1:
                                    corr = float(corr.item())
                                else:
                                    corr = float(corr[0])
                            
                            # Check if correlation is valid (not NaN)
                            if isinstance(corr, (int, float)) and not (np.isnan(corr) if isinstance(corr, (np.floating, float)) else False):
                                rank_correlations.append(float(corr))
                        except Exception:
                            # Skip if correlation computation fails
                            continue
                
                stability = np.mean(rank_correlations) if rank_correlations else 0.0
                
                top_5_indices = feature_ranks[:min(5, len(feature_ranks), len(feature_names))]
                stability_metrics[model_name] = {
                    'mean_rank_correlation': float(stability),
                    'std_rank_correlation': float(np.std(rank_correlations)) if rank_correlations else 0.0,
                    'top_features': [feature_names[i] for i in top_5_indices if i < len(feature_names)]
                }
            else:
                top_5_indices = feature_ranks[:min(5, len(feature_ranks), len(feature_names))]
                stability_metrics[model_name] = {
                    'mean_rank_correlation': 1.0,
                    'std_rank_correlation': 0.0,
                    'top_features': [feature_names[i] for i in top_5_indices if i < len(feature_names)]
                }
        
        return stability_metrics
    
    def evaluate_explanation_fidelity(
        self,
        model: Any,
        X_explain: pd.DataFrame,
        shap_values: np.ndarray,
        expected_value: Optional[float] = None,
        explained_indices: Optional[List[int]] = None,
        expected_features: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate fidelity of SHAP explanations (like-for-like when SHAP was computed on a subset).
        
        Args:
            model: Trained model
            X_explain: Full validation/data to explain (or subset if no explained_indices)
            shap_values: SHAP values (length = number of explained rows)
            expected_value: Expected SHAP value (baseline)
            explained_indices: If SHAP was computed on a subset, row indices in X_explain that were used
            expected_features: Optional list of feature names the model was trained on (for alignment)
        """
        self.logger.info("Evaluating explanation fidelity...")
        total_eval_rows = len(X_explain)

        # Align X_explain to model's expected feature names (avoids sklearn feature_names mismatch after feature engineering)
        X_explain = _align_X_to_model(model, X_explain, expected_features)
        if not isinstance(X_explain, pd.DataFrame):
            X_explain = pd.DataFrame(X_explain)

        # Like-for-like: if SHAP was on a subset, slice model predictions to same rows
        model_predictions = None
        if explained_indices is not None and len(explained_indices) > 0:
            try:
                X_subset = X_explain.iloc[explained_indices]
            except Exception:
                X_subset = None
            if X_subset is not None and len(X_subset) == len(explained_indices):
                X_subset = _align_X_to_model(model, X_subset, expected_features)
                if not isinstance(X_subset, pd.DataFrame):
                    X_subset = pd.DataFrame(X_subset)
                try:
                    model_predictions = model.predict_proba(X_subset)[:, 1]
                    self.logger.info(
                        "Fidelity: subset applied (explained_indices len=" + str(len(explained_indices)) + "), total eval set rows=" + str(total_eval_rows)
                    )
                except Exception:
                    model_predictions = None
                if model_predictions is None:
                    explained_indices = None
            else:
                explained_indices = None
        if model_predictions is None:
            try:
                model_predictions = model.predict_proba(X_explain)[:, 1]
            except Exception:
                self.logger.warning(
                    "Fidelity: model.predict_proba failed after alignment; returning fidelity as nan"
                )
                return {
                    "correlation": float("nan"),
                    "mean_absolute_error": float("nan"),
                    "expected_value": float(expected_value) if expected_value is not None else float("nan"),
                    "fidelity_subset_applied": False,
                    "n_compared": 0,
                }

        # Reconstruct predictions from SHAP values
        if expected_value is None:
            expected_value = float(np.mean(model_predictions))
        if isinstance(expected_value, np.ndarray):
            expected_value = float(expected_value.item() if expected_value.size == 1 else expected_value.flat[0])
        elif not isinstance(expected_value, (int, float)):
            expected_value = float(expected_value)

        # Handle multi-dimensional SHAP values
        if len(shap_values.shape) > 2:
            shap_values = shap_values[:, :, 1] if shap_values.shape[2] > 1 else shap_values[:, :, 0]
        n_shap_rows = shap_values.shape[0] if len(shap_values.shape) >= 1 else 0
        if len(shap_values.shape) == 2 and shap_values.shape[1] > len(X_explain.columns):
            shap_values = shap_values[:, :len(X_explain.columns)]

        if len(shap_values.shape) == 2:
            shap_sum = shap_values.sum(axis=1)
        else:
            shap_sum = np.array(shap_values).flatten()
        reconstructed_predictions = np.ravel(expected_value + shap_sum)
        model_predictions = np.ravel(model_predictions)

        # Regression check: ensure identical length before spearmanr
        n_model, n_recon = len(model_predictions), len(reconstructed_predictions)
        if n_model != n_recon:
            n_use = min(n_model, n_recon)
            self.logger.warning(
                f"Fidelity length mismatch: model_predictions={n_model}, reconstructed={n_recon}; "
                f"truncating to {n_use} (explained_indices may be missing or stale)"
            )
            model_predictions = model_predictions[:n_use]
            reconstructed_predictions = reconstructed_predictions[:n_use]

        # Regression check: do not crash if lengths still differ
        if len(model_predictions) != len(reconstructed_predictions):
            self.logger.warning(
                "Fidelity: skipping correlation (lengths still differ after alignment); returning safe defaults"
            )
            fidelity_metrics = {
                'correlation': 0.0,
                'mean_absolute_error': float('nan'),
                'expected_value': float(expected_value),
                'fidelity_subset_applied': explained_indices is not None,
                'n_compared': 0,
            }
            return fidelity_metrics
        correlation, _ = spearmanr(model_predictions, reconstructed_predictions)
        mae = float(np.mean(np.abs(model_predictions - reconstructed_predictions)))
        fidelity_metrics = {
            'correlation': float(correlation) if not np.isnan(correlation) else 0.0,
            'mean_absolute_error': mae,
            'expected_value': float(expected_value),
            'fidelity_subset_applied': explained_indices is not None,
            'n_compared': len(model_predictions),
        }
        return fidelity_metrics
    
    def evaluate_readability(
        self,
        natural_language_explanations: Dict[str, Dict[int, str]]
    ) -> Dict[str, float]:
        """
        Evaluate readability of natural language explanations.
        
        Uses simple heuristics:
        - Length (not too short, not too long)
        - Number of features mentioned
        - Clarity of language
        
        Args:
            natural_language_explanations: Dictionary of NL explanations
            
        Returns:
            Dictionary with readability scores
        """
        self.logger.info("Evaluating explanation readability...")
        
        readability_scores = {}
        
        for model_name, explanations in natural_language_explanations.items():
            scores = []
            
            for idx, explanation in explanations.items():
                # Length score (optimal around 200-500 characters)
                length = len(explanation)
                if 200 <= length <= 500:
                    length_score = 1.0
                elif length < 200:
                    length_score = length / 200
                else:
                    length_score = max(0, 1 - (length - 500) / 500)
                
                # Feature count score (optimal 3-7 features)
                feature_count = explanation.count('.')
                if 3 <= feature_count <= 7:
                    feature_score = 1.0
                else:
                    feature_score = max(0, 1 - abs(feature_count - 5) / 5)
                
                # Clarity score (presence of key terms)
                clarity_terms = ['predicted', 'confidence', 'risk', 'increased', 'decreased']
                clarity_score = sum(1 for term in clarity_terms if term.lower() in explanation.lower()) / len(clarity_terms)
                
                # Combined score
                combined_score = (length_score * 0.3 + feature_score * 0.3 + clarity_score * 0.4)
                scores.append(combined_score)
            
            readability_scores[model_name] = {
                'mean_readability': float(np.mean(scores)),
                'std_readability': float(np.std(scores)),
                'num_explanations': len(scores)
            }
        
        return readability_scores
    
    def generate_explainability_report(
        self,
        explainability_results: Dict[str, Any],
        models: Dict[str, Any],
        X_val: pd.DataFrame,
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive explainability evaluation report.
        
        Args:
            explainability_results: Results from explainability agent
            models: Trained models
            X_val: Validation data
            output_path: Path to save report
            
        Returns:
            Dictionary with all explainability metrics
        """
        self.logger.info("Generating explainability evaluation report...")
        
        report = {
            'shap_stability': {},
            'fidelity': {},
            'readability': {}
        }
        
        # Evaluate SHAP stability
        if 'shap_explanations' in explainability_results:
            report['shap_stability'] = self.evaluate_shap_stability(
                explainability_results['shap_explanations']
            )
            
            # Evaluate fidelity for each model (like-for-like when SHAP was on subset)
            ec = getattr(self.config, 'explainability', None)
            explain_n = getattr(ec, 'explain_n', None) or getattr(ec, 'shap_max_explain_instances', 200)
            total_eval_rows = len(X_val)
            for model_name, shap_exp in explainability_results['shap_explanations'].items():
                if model_name in models:
                    explained_indices = shap_exp.get("explained_indices")
                    if explained_indices is not None:
                        X_for_fidelity = X_val
                        actual_explained = len(explained_indices)
                    else:
                        X_for_fidelity = shap_exp.get("X_explain_used") or X_val
                        explained_indices = None
                        sv = shap_exp.get("shap_values")
                        actual_explained = int(sv.shape[0]) if hasattr(sv, "shape") and len(sv.shape) >= 1 else len(X_for_fidelity)
                    self.logger.info(
                        f"Fidelity for {model_name}: total eval rows={total_eval_rows}, "
                        f"explain_n={explain_n}, actual explained={actual_explained}, "
                        f"subset_applied={explained_indices is not None}"
                    )
                    fidelity = self.evaluate_explanation_fidelity(
                        model=models[model_name],
                        X_explain=X_for_fidelity,
                        shap_values=shap_exp['shap_values'],
                        expected_value=shap_exp.get('expected_value'),
                        explained_indices=explained_indices,
                        expected_features=shap_exp.get('feature_names'),
                    )
                    report['fidelity'][model_name] = fidelity
        
        # Evaluate readability
        if 'natural_language' in explainability_results:
            report['readability'] = self.evaluate_readability(
                explainability_results['natural_language']
            )
        
        # Save report
        if output_path is None:
            output_path = self.config.evaluation.results_dir / 'explainability_evaluation.json'
        
        # Convert numpy types for JSON serialization
        report_serializable = json.loads(json.dumps(report, default=json_safe))
        
        with open(output_path, 'w') as f:
            json.dump(report_serializable, f, indent=2, default=json_safe)
        
        self.logger.info(f"Explainability evaluation report saved to {output_path}")
        
        return report
