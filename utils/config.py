"""
Configuration management for Multi-Agent Healthcare Analytics Pipeline.

Config lives here: utils.config (PipelineConfig, get_config()).
Optional schema reference: configs/default.yaml (keys only; load in code if needed).

Single schema: dataset, enable_baseline, enable_cross_dataset, explain_n, feedback (enabled, trigger_metric, threshold, action).
Research outputs → run_dir/research_outputs; dashboard-only artifacts → run_dir/dashboard_outputs.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Literal

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class DataConfig:
    """Configuration for data sources and processing."""
    # Dataset selection: list of datasets to process
    datasets: List[str] = None
    # Backward compatibility: single dataset name
    dataset_name: Optional[str] = None
    
    # Data paths (will be set per run)
    raw_data_dir: Optional[Path] = None
    processed_data_dir: Optional[Path] = None
    
    # Train/val/test split ratios
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Data augmentation settings
    use_augmentation: bool = True
    augmentation_factor: float = 3.0  # Multiply dataset by this factor (e.g., 3.0 = triple the data)
    
    # Random seed for reproducibility
    random_seed: int = 42
    
    # Dataset URLs with fallback mirrors
    dataset_urls: Dict[str, List[str]] = None
    
    def __post_init__(self):
        """Initialize dataset URLs after dataclass creation."""
        # Backward compatibility: if dataset_name is set, use it
        if self.dataset_name is not None and self.datasets is None:
            self.datasets = [self.dataset_name]
        
        if self.datasets is None:
            # Allow override via env: DATA_DATASETS=heart_disease or DATA_DATASETS=diabetes,heart_disease
            env_datasets = os.environ.get('DATA_DATASETS', '').strip()
            if env_datasets:
                self.datasets = [s.strip() for s in env_datasets.split(',') if s.strip()]
            if not self.datasets:
                # Use diabetes by default (~768 samples; with augmentation gives 1500+)
                # Note: only the first dataset is processed per run; one run = one dataset = one run.log
                self.datasets = ['diabetes']
        
        # Initialize augmentation settings if not set
        if not hasattr(self, 'use_augmentation'):
            self.use_augmentation = True
        if not hasattr(self, 'augmentation_factor'):
            self.augmentation_factor = 2.0  # Double the dataset size
        
        # Set dataset_name for backward compatibility
        if self.dataset_name is None and self.datasets:
            self.dataset_name = self.datasets[0]
        
        if self.dataset_urls is None:
            self.dataset_urls = {
                'heart_disease': [
                    'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data',
                    'https://raw.githubusercontent.com/plotly/datasets/master/heart.csv'  # Fallback
                ],
                'diabetes': [
                    'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv',
                    'https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data'  # Fallback
                ],
                'framingham': [
                    'https://raw.githubusercontent.com/GauravPadawe/Framingham-Heart-Study/master/framingham.csv',
                ],
                'breast_cancer': [
                    'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',
                    'https://raw.githubusercontent.com/jbrownlee/Datasets/master/breast-cancer.csv'  # Fallback
                ],
                'liver_disease': [
                    'https://archive.ics.uci.edu/ml/machine-learning-databases/liver-disorders/bupa.data',
                    'https://raw.githubusercontent.com/jbrownlee/Datasets/master/liver.csv'  # Fallback
                ],
                'hepatitis': [
                    'https://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data',
                    'https://raw.githubusercontent.com/jbrownlee/Datasets/master/hepatitis.csv'  # Fallback
                ],
                'parkinsons': [
                    'https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data',
                    'https://raw.githubusercontent.com/jbrownlee/Datasets/master/parkinsons.csv'  # Fallback
                ],
                'thyroid': [
                    'https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/new-thyroid.data',
                    'https://raw.githubusercontent.com/jbrownlee/Datasets/master/thyroid.csv'  # Fallback
                ],
                'heart_failure': [
                    'https://raw.githubusercontent.com/plotly/datasets/master/heart-failure.csv',
                    'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-failure/heart_failure_clinical_records_dataset.csv'  # Fallback
                ],
                'stroke': [
                    'https://raw.githubusercontent.com/plotly/datasets/master/stroke.csv',
                    'https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/download?datasetVersionNumber=2'  # Fallback
                ],
                'kidney_disease': [
                    'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat',
                    'https://raw.githubusercontent.com/jbrownlee/Datasets/master/kidney.csv'  # Fallback
                ],
                'mammographic': [
                    'https://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data',
                    'https://raw.githubusercontent.com/jbrownlee/Datasets/master/mammographic.csv'  # Fallback
                ],
                'blood_transfusion': [
                    'https://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/transfusion.data',
                    'https://raw.githubusercontent.com/jbrownlee/Datasets/master/blood-transfusion.csv'  # Fallback
                ],
                'cervical_cancer': [
                    'https://archive.ics.uci.edu/ml/machine-learning-databases/00383/risk_factors_cervical_cancer.csv',
                    'https://raw.githubusercontent.com/jbrownlee/Datasets/master/cervical-cancer.csv'  # Fallback
                ],
                'lung_cancer': [
                    'https://archive.ics.uci.edu/ml/machine-learning-databases/lung-cancer/lung-cancer.data',
                    'https://raw.githubusercontent.com/jbrownlee/Datasets/master/lung-cancer.csv'  # Fallback
                ],
                'prostate_cancer': [
                    'https://archive.ics.uci.edu/ml/machine-learning-databases/prostate-cancer/prostate-cancer.data',
                    'https://raw.githubusercontent.com/jbrownlee/Datasets/master/prostate-cancer.csv'  # Fallback
                ],
                'dermatology': [
                    'https://archive.ics.uci.edu/ml/machine-learning-databases/dermatology/dermatology.data',
                    'https://raw.githubusercontent.com/jbrownlee/Datasets/master/dermatology.csv'  # Fallback
                ],
                'arrhythmia': [
                    'https://archive.ics.uci.edu/ml/machine-learning-databases/arrhythmia/arrhythmia.data',
                    'https://raw.githubusercontent.com/jbrownlee/Datasets/master/arrhythmia.csv'  # Fallback
                ],
                'primary_tumor': [
                    'https://archive.ics.uci.edu/ml/machine-learning-databases/primary-tumor/primary-tumor.data',
                    'https://raw.githubusercontent.com/jbrownlee/Datasets/master/primary-tumor.csv'  # Fallback
                ],
                'lymphography': [
                    'https://archive.ics.uci.edu/ml/machine-learning-databases/lymphography/lymphography.data',
                    'https://raw.githubusercontent.com/jbrownlee/Datasets/master/lymphography.csv'  # Fallback
                ],
                'appendicitis': [
                    'https://archive.ics.uci.edu/ml/machine-learning-databases/appendicitis/appendicitis.data',
                    'https://raw.githubusercontent.com/jbrownlee/Datasets/master/appendicitis.csv'  # Fallback
                ],
                'ecoli': [
                    'https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data',
                    'https://raw.githubusercontent.com/jbrownlee/Datasets/master/ecoli.csv'  # Fallback
                ],
                'yeast': [
                    'https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data',
                    'https://raw.githubusercontent.com/jbrownlee/Datasets/master/yeast.csv'  # Fallback
                ],
                'sick': [
                    'https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/sick.data',
                    'https://raw.githubusercontent.com/jbrownlee/Datasets/master/sick.csv'  # Fallback
                ],
                'hypothyroid': [
                    'https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/hypothyroid.data',
                    'https://raw.githubusercontent.com/jbrownlee/Datasets/master/hypothyroid.csv'  # Fallback
                ],
                'hyperthyroid': [
                    'https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/hyperthyroid.data',
                    'https://raw.githubusercontent.com/jbrownlee/Datasets/master/hyperthyroid.csv'  # Fallback
                ],
                'splice': [
                    'https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/splice-junction-gene-sequences/splice.data',
                    'https://raw.githubusercontent.com/jbrownlee/Datasets/master/splice.csv'  # Fallback
                ],
                'mushroom': [
                    'https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data',
                    'https://raw.githubusercontent.com/jbrownlee/Datasets/master/mushroom.csv'  # Fallback
                ],
                'cleveland_heart': [
                    'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data',
                    'https://raw.githubusercontent.com/plotly/datasets/master/heart.csv'  # Fallback
                ],
                'hungarian_heart': [
                    'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data',
                    'https://raw.githubusercontent.com/plotly/datasets/master/heart.csv'  # Fallback
                ]
            }
        
        # Set default paths if not provided
        if self.raw_data_dir is None:
            self.raw_data_dir = PROJECT_ROOT / 'data' / 'raw'
        if self.processed_data_dir is None:
            self.processed_data_dir = PROJECT_ROOT / 'data' / 'processed'
        
        # Create directories
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    """Configuration for model training and hyperparameters (thesis-grade model suite)."""
    # Model types to train: interpretable + ensemble + kernel methods (6 models)
    models: list = None
    
    # Hyperparameter grids
    lr_params: Dict[str, Any] = None
    rf_params: Dict[str, Any] = None
    xgb_params: Dict[str, Any] = None
    svm_params: Dict[str, Any] = None
    gb_params: Dict[str, Any] = None
    knn_params: Dict[str, Any] = None
    
    # Cross-validation
    cv_folds: int = 5
    
    # Model output directory
    models_dir: Path = PROJECT_ROOT / 'outputs' / 'models'
    
    def __post_init__(self):
        """Initialize default model configurations."""
        if self.models is None:
            self.models = [
                'logistic_regression',
                'random_forest',
                'xgboost',
                'svm',
                'gradient_boosting',
                'knn',
            ]
        
        if self.lr_params is None:
            self.lr_params = {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l2'],
                'max_iter': [1000]
            }
        
        if self.rf_params is None:
            self.rf_params = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5]
            }
        
        if self.xgb_params is None:
            self.xgb_params = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3]
            }
        
        if self.svm_params is None:
            self.svm_params = {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto'],
            }
        
        if self.gb_params is None:
            self.gb_params = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1],
            }
        
        if self.knn_params is None:
            self.knn_params = {
                'n_neighbors': [5, 11, 21, 31],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan'],
            }
        
        self.models_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class ExplainabilityConfig:
    """Configuration for explainability methods."""
    # Methods to use
    use_shap: bool = True
    use_lime: bool = True
    
    # SHAP configuration
    shap_sample_size: int = 100  # For KernelExplainer background
    shap_plot_samples: int = 20  # Number of individual explanations to plot
    explain_n: int = 200  # Max instances to explain when using KernelExplainer (cap); used consistently by agent and evaluator
    shap_max_explain_instances: int = 200  # Alias for explain_n (legacy)
    shap_kernel_nsamples: int = 50  # KernelExplainer nsamples per instance (lower = faster, less accurate)
    
    # LIME configuration
    lime_num_features: int = 10  # Top features to show in LIME
    lime_num_samples: int = 5000  # Samples for LIME perturbation
    
    # Output directories
    explanations_dir: Path = PROJECT_ROOT / 'outputs' / 'explanations'
    plots_dir: Path = PROJECT_ROOT / 'outputs' / 'plots'
    
    def __post_init__(self):
        """Create output directories."""
        self.explanations_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics."""
    # Metrics to compute
    metrics: list = None
    
    # Output directory (research outputs: stability, fidelity, ERI, etc.)
    results_dir: Path = PROJECT_ROOT / 'outputs' / 'results'
    
    def __post_init__(self):
        """Initialize default metrics."""
        if self.metrics is None:
            self.metrics = [
                'accuracy', 'precision', 'recall', 'f1_score',
                'roc_auc', 'confusion_matrix'
            ]
        if self.results_dir is not None:
            self.results_dir.mkdir(parents=True, exist_ok=True)


# Single config schema: feedback (used by FeedbackAgent and pipeline)
FeedbackTriggerMetric = Literal['fidelity', 'shap_stability', 'eri']
FeedbackAction = Literal['retrain_best_model', 'switch_model', 'retrain_with_tuned_params']


@dataclass
class FeedbackConfig:
    """Configuration for FeedbackAgent (research vs demo separation). Safe defaults for missing keys."""
    enabled: bool = True
    trigger_metric: str = "eri"
    threshold: float = 0.6
    action: str = "none"


@dataclass
class CrossDatasetConfig:
    """Configuration for CrossDatasetAgent (cross-dataset validation). Safe defaults for missing keys."""
    enabled: bool = False
    target_dataset: str = "heart_disease"
    eval_split: str = "test"
    train_dataset: Optional[str] = None
    test_datasets: List[str] = field(default_factory=list)


@dataclass
class PipelineConfig:
    """Main pipeline configuration combining all sub-configs.
    Single schema: dataset, enable_baseline, enable_cross_dataset, explain_n, feedback, cross_dataset.
    All sub-configs use default_factory so PipelineConfig() and dict-loaded config never miss keys.
    """
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    cross_dataset: CrossDatasetConfig = field(default_factory=CrossDatasetConfig)

    run_id: Optional[str] = None
    run_dir: Optional[Path] = None
    log_dir: Optional[Path] = None
    log_level: str = "INFO"
    track_collaboration: bool = True

    dataset: Optional[str] = None
    enable_baseline: bool = True
    enable_feature_engineering: bool = True
    enable_cross_dataset: bool = False
    cross_dataset_enabled: bool = False

    def __post_init__(self):
        """Initialize all sub-configurations. Ensure feedback and cross_dataset exist (backwards compat for dict/yaml load)."""
        from utils.paths import get_run_id, get_run_dir, get_run_subdirs

        if not hasattr(self, "feedback") or self.feedback is None:
            self.feedback = FeedbackConfig()
        if not hasattr(self, "cross_dataset") or self.cross_dataset is None:
            self.cross_dataset = CrossDatasetConfig()
        if getattr(self.cross_dataset, "test_datasets", None) is None:
            self.cross_dataset.test_datasets = []
        if not hasattr(self.cross_dataset, "target_dataset"):
            self.cross_dataset.target_dataset = "heart_disease"
        if not hasattr(self.cross_dataset, "eval_split"):
            self.cross_dataset.eval_split = "test"

        if self.run_id is None:
            self.run_id = get_run_id()
        if self.run_dir is None:
            self.run_dir = get_run_dir(self.run_id)
        if getattr(self, "data", None) is None:
            self.data = DataConfig()
        if getattr(self, "model", None) is None:
            self.model = ModelConfig()
        if getattr(self, "explainability", None) is None:
            self.explainability = ExplainabilityConfig()
        if getattr(self, "evaluation", None) is None:
            self.evaluation = EvaluationConfig()

        # Sync dataset string from data
        if self.dataset is None and getattr(self.data, 'datasets', None):
            self.dataset = self.data.datasets[0] if self.data.datasets else None
        if self.dataset is None and getattr(self.data, 'dataset_name', None):
            self.dataset = self.data.dataset_name
        
        # Sync cross_dataset_enabled with enable_cross_dataset and cross_dataset.enabled
        if hasattr(self, 'enable_cross_dataset'):
            self.cross_dataset_enabled = bool(self.enable_cross_dataset)
        if getattr(self.cross_dataset, 'enabled', False):
            self.cross_dataset_enabled = True
        
        # Update paths to use run-based structure (research_outputs for reports)
        run_subdirs = get_run_subdirs(self.run_id)
        
        if self.log_dir is None:
            self.log_dir = run_subdirs['logs']
        
        self.model.models_dir = run_subdirs['models']
        self.explainability.explanations_dir = run_subdirs['explainability']
        self.explainability.plots_dir = run_subdirs['figures']
        # Research outputs: stability, fidelity, ERI, feedback decisions, baseline comparison, cross-dataset
        self.evaluation.results_dir = run_subdirs['research_outputs']
        self.cross_dataset_outputs_dir = run_subdirs['research_outputs'] / 'cross_dataset'
        # Dashboard/UI-only artifacts (do not mix UI-only into research report logic)
        self.dashboard_outputs_dir = run_subdirs.get('dashboard_outputs', run_subdirs['reports'])


def get_config() -> PipelineConfig:
    """Get the default pipeline configuration. Safe when called with no args or when config keys are missing."""
    config = PipelineConfig()
    # Sanity check: ensure cross_dataset is always present (no AttributeError)
    try:
        _ = config.cross_dataset.enabled
        _ = config.cross_dataset.test_datasets
        _ = getattr(config.cross_dataset, "target_dataset", None)
    except AttributeError:
        config.cross_dataset = CrossDatasetConfig()
    return config
