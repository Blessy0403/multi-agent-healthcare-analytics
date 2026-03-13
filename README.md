# Multi-Agent AI Collaboration for Explainable Healthcare Analytics

A production-ready, end-to-end multi-agent AI pipeline for explainable healthcare analytics, designed for research and benchmarking against traditional single-model baselines. This repository supports a **6-month thesis scope** with a 4-agent architecture, a **6-model comparative suite**, and three evaluation dimensions (accuracy, explainability, collaboration).

## 📋 Table of Contents

- [Overview](#overview)
- [Research Question Mapping](#research-question-mapping)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Agents](#agents)
- [Evaluation Framework](#evaluation-framework)
- [Installation](#installation)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Research Contributions](#research-contributions)

## 🎯 Overview

This pipeline implements a **multi-agent AI system** for healthcare analytics with explicit agent roles, structured collaboration, and comprehensive explainability. The system is designed to:

1. **Process real healthcare datasets** (UCI Heart Disease, UCI Diabetes, extensible to others)
2. **Train a thesis-grade 6-model suite** with GridSearchCV: Logistic Regression, Random Forest, XGBoost, SVM, Gradient Boosting, K-Nearest Neighbors
3. **Generate explainable predictions** using SHAP (tree and kernel explainers) and LIME
4. **Compare multi-agent vs. single-model** baselines
5. **Evaluate across three dimensions**: Predictive accuracy, explainability quality, and collaboration efficiency

## 🔬 Research Question Mapping

This implementation directly addresses the research questions from the Master Thesis exposé:

### RQ1: Can multi-agent collaboration improve predictive accuracy?
- **Evaluation**: Compare ROC-AUC, accuracy, precision, recall, F1-score between multi-agent and baseline
- **Implementation**: `evaluation/metrics.py` - `MetricsEvaluator.compare_predictive_accuracy()`

### RQ2: Does agent-based explainability enhance interpretability?
- **Evaluation**: SHAP stability, explanation fidelity, natural language readability
- **Implementation**: `evaluation/explainability_eval.py` - `ExplainabilityEvaluator`

### RQ3: What is the efficiency cost of multi-agent collaboration?
- **Evaluation**: Execution time, handover latency, collaboration overhead
- **Implementation**: `evaluation/collaboration_eval.py` - `CollaborationEvaluator`

## 🏗️ Architecture

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      ORCHESTRATOR AGENT                          │
│  - Coordinates agent execution                                   │
│  - Manages artifact passing                                     │
│  - Tracks collaboration metrics                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Orchestrates
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌──────────────────┐
│  DATA AGENT   │───▶│ MODEL AGENT   │───▶│ EXPLAINABILITY   │
│               │    │               │    │     AGENT        │
│ - Download    │    │ - Train 6     │    │ - SHAP           │
│ - Clean       │    │   models: LR, │    │ - LIME           │
│ - Encode      │    │   RF, XGB, SVM│    │ - NL Generation  │
│ - Scale       │    │   GB, KNN+tune│    │ - Plot Generation│
│ - Split       │    │ - Evaluate    │    │                  │
└───────────────┘    └───────────────┘    └──────────────────┘
        │                     │                     │
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   EVALUATION     │
                    │   FRAMEWORK      │
                    │                  │
                    │ - Accuracy       │
                    │ - Explainability │
                    │ - Collaboration  │
                    └──────────────────┘
```

### Agent Communication Flow

```
Data Agent
    │
    ├─▶ Produces: train.csv, val.csv, test.csv, metadata.json
    │
    ▼
Model Agent
    │
    ├─▶ Receives: Processed datasets
    ├─▶ Produces: Trained models (.pkl), metrics.json, predictions.csv
    │
    ▼
Explainability Agent
    │
    ├─▶ Receives: Trained models, validation data
    ├─▶ Produces: SHAP plots, LIME explanations, NL explanations
    │
    ▼
Evaluation Framework
    │
    ├─▶ Receives: All artifacts from agents
    ├─▶ Produces: Comparison reports, evaluation metrics
```

### Comparison with Baseline

```
┌─────────────────────────────────────────────────────────────┐
│              MULTI-AGENT PIPELINE                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐             │
│  │   Data   │─▶│  Model   │─▶│Explainability│             │
│  │  Agent   │  │  Agent   │  │    Agent     │             │
│  └──────────┘  └──────────┘  └──────────────┘             │
│       │              │              │                       │
│       └──────────────┼──────────────┘                       │
│                      ▼                                       │
│              ┌──────────────┐                                │
│              │ Orchestrator │                                │
│              └──────────────┘                                │
└─────────────────────────────────────────────────────────────┘
                        │
                        │ Compare
                        │
┌─────────────────────────────────────────────────────────────┐
│              SINGLE-MODEL BASELINE                           │
│  ┌──────────────────────────────────────────┐               │
│  │  Monolithic Pipeline                     │               │
│  │  - Data processing                       │               │
│  │  - Model training                        │               │
│  │  - Evaluation                            │               │
│  └──────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────┘
```

## 📊 Dataset

### Supported Datasets

1. **UCI Heart Disease Dataset** (Cleveland)
   - **Source**: UCI ML Repository
   - **Task**: Binary classification (disease / no disease)
   - **Features**: 13 features (age, sex, cp, trestbps, chol, etc.)
   - **Samples**: ~300 instances
   - **Preprocessing**: Missing value imputation, categorical encoding, scaling

2. **UCI Diabetes Dataset** (Pima Indians)
   - **Source**: UCI ML Repository
   - **Task**: Binary classification (diabetes / no diabetes)
   - **Features**: 8 features (pregnancies, glucose, blood_pressure, etc.)
   - **Samples**: ~768 instances
   - **Preprocessing**: Zero-value handling, missing value imputation, scaling

### Data Processing Pipeline

```
Raw Data (UCI Repository)
    │
    ▼
[Data Agent]
    ├─▶ Download dataset
    ├─▶ Load & clean (handle missing values)
    ├─▶ Encode categorical features
    ├─▶ Scale numerical features (StandardScaler)
    ├─▶ Split: Train (70%) / Val (15%) / Test (15%)
    │
    └─▶ Output: train.csv, val.csv, test.csv, metadata.json
```

## 🤖 Agents

### Agent 1: Data Agent (`agents/data_agent.py`)

**Responsibilities:**
- Download datasets from UCI ML Repository
- Clean data (handle missing values, outliers)
- Encode categorical features (Label Encoding)
- Scale features (StandardScaler)
- Split into train/validation/test sets
- Generate data metadata

**Key Methods:**
- `download_dataset()`: Downloads raw data
- `clean_data()`: Handles missing values
- `encode_features()`: Categorical encoding
- `scale_features()`: Feature scaling
- `split_data()`: Train/val/test split
- `process()`: Full pipeline execution

**Output Artifacts:**
- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`
- `data/processed/data_metadata.json`

### Agent 2: Model Agent (`agents/model_agent.py`)

**Responsibilities:**
- Train a **thesis-grade 6-model suite** with hyperparameter tuning (GridSearchCV):
  - **Logistic Regression** (interpretable baseline)
  - **Random Forest** (ensemble)
  - **XGBoost** (gradient boosting)
  - **SVM** (Support Vector Machine; common in healthcare ML)
  - **Gradient Boosting** (sklearn)
  - **K-Nearest Neighbors** (interpretable baseline)
- Model evaluation on validation set
- Select best model based on ROC-AUC

**Key Methods:**
- `train_logistic_regression()`, `train_random_forest()`, `train_xgboost()`
- `train_svm()`, `train_gradient_boosting()`, `train_knn()`
- `train_all_models()`: Train all configured models
- `_compute_metrics()`: Accuracy, precision, recall, F1, ROC-AUC

**Output Artifacts:**
- `outputs/models/{logistic_regression,random_forest,xgboost,svm,gradient_boosting,knn}.pkl`
- `outputs/models/model_metrics.json`
- `outputs/models/validation_predictions.csv`

### Agent 3: Explainability Agent (`agents/explainability_agent.py`)

**Responsibilities:**
- Generate SHAP explanations (global and local)
- Generate LIME explanations (local)
- Create natural language explanations
- Generate visualization plots

**Key Methods:**
- `explain_with_shap()`: SHAP TreeExplainer/KernelExplainer
- `explain_with_lime()`: LIME TabularExplainer
- `generate_shap_plots()`: Summary, bar, waterfall plots
- `generate_natural_language_explanation()`: Convert SHAP to text

**Output Artifacts:**
- `outputs/plots/{model}_shap_summary.png`
- `outputs/plots/{model}_shap_bar.png`
- `outputs/plots/{model}_shap_waterfall_{idx}.png`
- `outputs/explanations/explanations.json`
- `outputs/explanations/natural_language_explanations.txt`

### Agent 4: Orchestrator (`agents/orchestrator.py`)

**Responsibilities:**
- Coordinate agent execution order
- Pass artifacts between agents
- Track collaboration metrics
- Log agent handovers
- Measure execution times

**Key Methods:**
- `execute_pipeline()`: Execute full multi-agent pipeline
- `get_collaboration_summary()`: Generate collaboration summary
- `_save_collaboration_metrics()`: Save metrics to JSON

**Output Artifacts:**
- `outputs/logs/collaboration_metrics.json`
- `outputs/logs/{agent}_collaboration.json`

## 📈 Evaluation Framework

### Three-Dimensional Evaluation

#### A) Predictive Accuracy (`evaluation/metrics.py`)

**Metrics:**
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

**Comparison:**
- Multi-agent vs. Baseline for each model type
- Statistical significance testing
- Performance difference analysis

**Output:**
- `outputs/results/comparison_report.csv`
- `outputs/results/comparison_summary.json`

#### B) Explainability Quality (`evaluation/explainability_eval.py`)

**Metrics:**
- **SHAP Stability**: Rank correlation across samples
- **Explanation Fidelity**: Correlation between SHAP reconstructions and actual predictions
- **Readability**: Length, feature count, clarity of natural language explanations

**Output:**
- `outputs/results/explainability_evaluation.json`

#### C) Collaboration Efficiency (`evaluation/collaboration_eval.py`)

**Metrics:**
- Execution time per agent
- Total pipeline execution time
- Handover latency
- Error rate
- Collaboration overhead

**Comparison:**
- Multi-agent vs. Baseline execution time
- Overhead percentage

**Output:**
- `outputs/results/collaboration_evaluation.json`

## 🚀 Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. **Clone or navigate to the project directory:**
```bash
cd /path/to/blessy_thesis
```

2. **Create virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Basic Execution

Run the complete pipeline:

```bash
python main.py
```

Or use the Makefile:
```bash
make run
```

This will:
1. Execute the multi-agent pipeline
2. Execute the baseline pipeline
3. Run comprehensive evaluation
4. Generate all reports and visualizations in `outputs/runs/<run_id>/`

### Dashboard

Launch the Streamlit dashboard to visualize results:

```bash
streamlit run dashboard/app.py
```

Or use the Makefile:
```bash
make dashboard
```

The dashboard provides interactive exploration of:
- Pipeline overview and KPIs
- Data profiles and distributions
- Model performance metrics
- Explainability visualizations
- Baseline comparisons
- Collaboration efficiency metrics
- Interactive prediction explainer

### Configuration

Modify `utils/config.py` to customize:
- Dataset selection (`dataset_name`: 'heart_disease' or 'diabetes')
- Model hyperparameters
- Train/val/test split ratios
- Explainability settings

### Output Structure

```
blessy_thesis/
├── data/
│   ├── raw/                    # Downloaded raw datasets
│   └── processed/              # Cleaned and processed datasets
│       ├── train.csv
│       ├── val.csv
│       ├── test.csv
│       └── data_metadata.json
│
├── outputs/
│   ├── models/                 # Trained models
│   │   ├── logistic_regression.pkl
│   │   ├── random_forest.pkl
│   │   ├── xgboost.pkl
│   │   ├── model_metrics.json
│   │   └── validation_predictions.csv
│   │
│   ├── explanations/           # Model explanations
│   │   ├── explanations.json
│   │   └── natural_language_explanations.txt
│   │
│   ├── plots/                  # Visualization plots
│   │   ├── {model}_shap_summary.png
│   │   ├── {model}_shap_bar.png
│   │   └── {model}_shap_waterfall_{idx}.png
│   │
│   ├── results/                # Evaluation results
│   │   ├── comparison_report.csv
│   │   ├── comparison_summary.json
│   │   ├── explainability_evaluation.json
│   │   └── collaboration_evaluation.json
│   │
│   └── logs/                   # Execution logs
│       ├── collaboration_metrics.json
│       ├── {agent}_collaboration.json
│       └── {agent}_{timestamp}.log
```

## 📝 Research Contributions

### Key Features

1. **True Multi-Agent Architecture**
   - Independent agent modules with explicit roles
   - Structured artifact passing
   - Collaboration tracking and logging

2. **Comprehensive Explainability**
   - SHAP (global and local)
   - LIME (local)
   - Natural language generation
   - Visualization plots

3. **Rigorous Evaluation**
   - Three-dimensional evaluation framework
   - Statistical comparison with baseline
   - Reproducible results

4. **Production-Ready**
   - Modular design
   - Comprehensive logging
   - Error handling
   - Versioned datasets

### Thesis Integration

This implementation directly supports:

- **Chapter 3: Methodology**
  - Multi-agent architecture design
  - Agent role definitions
  - Collaboration protocols

- **Chapter 4: Implementation**
  - Code structure and organization
  - Agent implementation details
  - Evaluation framework

- **Chapter 5: Results**
  - Predictive accuracy comparison
  - Explainability quality analysis
  - Collaboration efficiency metrics

- **Chapter 6: Discussion**
  - Interpretation of results
  - Comparison with baseline
  - Limitations and future work

## 🔧 Technical Details

### Agent Communication

Agents communicate via **structured artifacts** (JSON, CSV, pickle files):
- No direct function calls between agents
- All communication through file system
- Enables independent agent development and testing

### Reproducibility

- Fixed random seeds (`random_seed=42`)
- Versioned datasets (downloaded from UCI)
- Comprehensive logging
- Structured output formats

### Extensibility

The architecture supports:
- Adding new agents (e.g., Feature Engineering Agent)
- Adding new models (e.g., Neural Networks)
- Adding new explainability methods (e.g., Integrated Gradients)
- Adding new datasets (extend `DataAgent`)

## 📚 References

- UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/
- SHAP: Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions.
- LIME: Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?" Explaining the predictions of any classifier.

## 📄 License

This project is developed for academic research purposes as part of a Master Thesis.

## 👤 Author

Developed for the Master Thesis: "Multi-Agent AI Collaboration for Explainable Healthcare Analytics"

---

**Note**: This is a research implementation. For production use, additional considerations (security, scalability, deployment) should be addressed.
