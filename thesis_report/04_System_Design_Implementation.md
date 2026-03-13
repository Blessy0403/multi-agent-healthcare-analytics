# Chapter 4: System Design and Implementation

## 4.1 Technology Stack

- **Language:** Python 3.x  
- **ML:** scikit-learn, XGBoost  
- **Explainability:** SHAP, LIME  
- **Dashboard:** Streamlit  
- **Other:** pandas, numpy, pickle for model serialisation, JSON for metadata and reports.

## 4.2 Repository and Run Structure

Each pipeline run creates a unique run directory under `outputs/runs/<run_id>/`:

```
outputs/runs/<run_id>/
├── data/           # train, val, test CSV; copies of processed data
├── models/         # *.pkl, model_metrics.json, validation_predictions.csv
├── explainability/# explanations.json, lime_explanations_*.json, natural_language_explanations.txt
├── figures/       # SHAP summary/bar/waterfall PNGs
├── reports/       # data_metadata_*.json, comparison_report.csv, *_evaluation.json, run_metadata.json
└── logs/          # run.log, collaboration_metrics.json
```

The **orchestrator** sets run_id (e.g., timestamp + short hash), creates this structure, and passes paths to each agent via configuration.

## 4.3 Agent Implementation Summary

- **Data Agent** (`agents/data_agent.py`): Implements load_raw_data, clean_data, encode_features, scale_features, split_data; saves metadata and CSVs to the run directory.  
- **Model Agent** (`agents/model_agent.py`): Implements train_* for each of the six models; _compute_metrics (accuracy, precision, recall, F1, ROC-AUC, confusion matrix); _save_models, _save_metrics, _save_predictions.  
- **Explainability Agent** (`agents/explainability_agent.py`): explain_with_shap (Tree/Kernel), explain_with_lime, generate_shap_plots, generate_natural_language_explanation; writes to explainability/ and figures/.  
- **Orchestrator** (`agents/orchestrator.py`): Executes Data → Model → Explainability; collects collaboration_metrics (agent_execution_times, handovers, errors, pipeline_status); returns results for main.py.  
- **Baseline** (`baseline/single_model_pipeline.py`): Reuses data outputs; trains same six models; saves baseline metrics and execution time.  
- **Evaluation** (`evaluation/metrics.py`, `explainability_eval.py`, `collaboration_eval.py`): Generate comparison_report.csv, explainability_evaluation.json, collaboration_evaluation.json.

## 4.4 Dashboard (Streamlit)

The dashboard (`dashboard/app.py` and pages under `dashboard/pages/`) provides:

- **Overview:** Run summary, best models per dataset, KPIs, comparison table, run log.  
- **Data:** Schema, missing values, feature distributions, class distribution (with thesis-style Plotly charts).  
- **Modeling:** Model leaderboard, metrics for selected model, confusion matrix, ROC curve (model-specific colours).  
- **Explainability:** SHAP summary/bar, LIME per-instance, natural-language explanations.  
- **Benchmarking:** Baseline vs. multi-agent comparison table and bar chart (distinct colours for Multi Agent, Baseline, Difference).  
- **Efficiency:** Agent execution timeline (per-agent colours), handover analysis, error analysis, efficiency metrics (pipeline status, success rate, total agents, overhead).  
- **Prediction Explainer:** Select instance or manual input; show prediction (styled card: No Disease / Disease with confidence); SHAP and natural-language explanation.

Charts are implemented in `dashboard/components/charts.py` (Plotly) with a consistent thesis theme; model-specific and agent-specific colours differentiate series.

## 4.5 Configuration and Reproducibility

Configuration (`utils/config.py`) centralises datasets, model lists, hyperparameter grids, cross-validation folds, and paths. Run ID and random seed (e.g., 42) are set at the start of `main.py` to ensure reproducible splits and training. Logging is written to `logs/run.log` for each run so that failures and timings are traceable.

---

*[INSERT FIGURE: Screenshot of dashboard home with pipeline flow and quick stats.]*  
*[INSERT FIGURE: Screenshot of Modeling page with leaderboard and ROC curve.]*
