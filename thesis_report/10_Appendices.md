# Chapter 10: Appendices

## Appendix A: Configuration (Excerpt)

The pipeline is configured via `utils/config.py`. Key settings:

- **Datasets:** `data.datasets` (e.g., `['diabetes']` or `['heart_disease']`).  
- **Models:** `model.models` — list of six model names.  
- **Hyperparameter grids:** `model.lr_params`, `model.rf_params`, `model.xgb_params`, `model.svm_params`, `model.gb_params`, `model.knn_params`.  
- **Cross-validation:** `model.cv_folds` (e.g., 5).  
- **Explainability:** `use_shap`, `use_lime`, `shap_sample_size`, `lime_num_features`, etc.  
- **Paths:** Run directory under `outputs/runs/<run_id>/`; subdirs: data, models, explainability, figures, reports, logs.

Environment variable `DATA_DATASETS` can override the default dataset list (e.g., `export DATA_DATASETS=diabetes`).

## Appendix B: Directory Structure (Run Output)

```
outputs/runs/<run_id>/
├── data/
│   ├── diabetes_train.csv
│   ├── diabetes_val.csv
│   └── diabetes_test.csv
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── svm.pkl
│   ├── gradient_boosting.pkl
│   ├── knn.pkl
│   ├── model_metrics.json
│   └── validation_predictions.csv
├── explainability/
│   ├── explanations.json
│   ├── lime_explanations_<model>.json  (per model)
│   └── natural_language_explanations.txt
├── figures/
│   ├── <model>_shap_summary.png
│   ├── <model>_shap_bar.png
│   └── <model>_shap_waterfall_<idx>.png
├── reports/
│   ├── data_metadata_<dataset>.json
│   ├── comparison_report.csv
│   ├── explainability_evaluation.json
│   ├── collaboration_evaluation.json
│   └── run_metadata.json
└── logs/
    ├── run.log
    └── collaboration_metrics.json
```

## Appendix C: Sample model_metrics.json (Excerpt)

```json
{
  "logistic_regression": {
    "accuracy": 0.78,
    "precision": 0.72,
    "recall": 0.65,
    "f1_score": 0.68,
    "roc_auc": 0.82,
    "confusion_matrix": [[tn, fp], [fn, tp]]
  },
  "random_forest": { ... },
  ...
}
```

## Appendix D: Sample comparison_report.csv (Excerpt)

Columns typically include: `model`, `metric`, `multi_agent`, `baseline`, `difference` (and optionally `percent_difference`). Rows: one per (model, metric) combination, plus optional execution_time row.

## Appendix E: How to Run and Reproduce

1. Clone or download the repository.  
2. Create a virtual environment: `python -m venv venv` and activate.  
3. Install dependencies: `pip install -r requirements.txt`.  
4. Run the pipeline: `python main.py`.  
5. Start the dashboard: `streamlit run dashboard/app.py`.  
6. In the dashboard, select the latest run and the desired dataset/model to view metrics and export or screenshot figures for the thesis.

## Appendix F: List of Figures and Tables (For Thesis Index)

- **Chapter 1:** Figure 1.1 – Four-agent pipeline diagram.  
- **Chapter 2:** Figure 2.1 – XAI methods summary (optional).  
- **Chapter 3:** Figure 3.1 – Agent flow; Table 3.1 – Hyperparameter grids.  
- **Chapter 4:** Figure 4.1 – Dashboard home; Figure 4.2 – Modeling page.  
- **Chapter 5:** Table 5.1 – Dataset statistics.  
- **Chapter 6:** Table 6.1 – Validation metrics; Figure 6.1 – ROC; Figure 6.2 – Confusion matrix; Table 6.2 – Comparison; Figure 6.3 – Comparison chart; Figure 6.4 – Leaderboard; Figures 6.5–6.9 – SHAP/LIME/NL; Table 6.3 – Execution times; Figure 6.10 – Agent timeline.

Use the **FIGURES_AND_CHARTS_GUIDE.md** in this folder to map each figure to the exact dashboard page and how to capture it.
