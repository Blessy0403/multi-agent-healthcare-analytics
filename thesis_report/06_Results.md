# Chapter 6: Results

## 6.1 Predictive Accuracy

### 6.1.1 Multi-Agent Pipeline

The multi-agent pipeline trains all six models (Logistic Regression, Random Forest, XGBoost, SVM, Gradient Boosting, KNN) on the preprocessed dataset and selects the best model by validation ROC-AUC. Table 6.1 summarises the validation metrics for each model on the Diabetes dataset (example run). [Fill in with your run’s actual numbers from `model_metrics.json` or the dashboard Modeling page.]

**Table 6.1 – Validation metrics (multi-agent pipeline, Diabetes dataset)**

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | —        | —         | —      | —        | —       |
| Random Forest       | —        | —         | —      | —        | —       |
| XGBoost             | —        | —         | —      | —        | —       |
| SVM                 | —        | —         | —      | —        | —       |
| Gradient Boosting   | —        | —         | —      | —        | —       |
| KNN                 | —        | —         | —      | —        | —       |

*Source: outputs/runs/<run_id>/models/model_metrics.json*

**Figure 6.1** shows the ROC curve for the best-performing model (e.g., KNN). Each model is assigned a distinct colour in the dashboard (blue for LR, green for RF, orange for XGBoost, purple for SVM, magenta for Gradient Boosting, olive for KNN) to facilitate comparison.

*[INSERT FIGURE 6.1: ROC curve for the best model (screenshot from dashboard Modeling page or export from Plotly).]*

**Figure 6.2** shows the confusion matrix for the selected model. The heatmap uses a model-specific colour scale so that different models are visually distinguishable when switching in the dashboard.

*[INSERT FIGURE 6.2: Confusion matrix (screenshot from dashboard Modeling page).]*

### 6.1.2 Baseline vs. Multi-Agent Comparison

Table 6.2 compares the best-model ROC-AUC (and optionally other metrics) between the multi-agent pipeline and the single-pipeline baseline. The comparison report is saved as `comparison_report.csv` in the run’s reports folder and is displayed in the Benchmarking page with distinct colours for Multi Agent (blue), Baseline (gray), and Difference (purple).

**Table 6.2 – Multi-agent vs. baseline (example)**

| Model               | Multi-Agent (ROC-AUC) | Baseline (ROC-AUC) | Difference |
|---------------------|------------------------|--------------------|------------|
| (per model or best) | —                      | —                  | —          |

*[INSERT FIGURE 6.3: Bar chart of comparison (Multi Agent vs. Baseline vs. Difference) from dashboard Benchmarking page.]*

### 6.1.3 Model Leaderboard

The dashboard Modeling page shows a leaderboard of all six models sorted by ROC-AUC. For the thesis, include a screenshot or table of this leaderboard for at least one run (e.g., Diabetes) and comment on which model family (e.g., KNN, Gradient Boosting) performed best and why this might be (e.g., dataset size, feature scale, class balance).

*[INSERT FIGURE 6.4: Model leaderboard table or screenshot.]*

---

## 6.2 Explainability Results

### 6.2.1 SHAP

SHAP summary plots show the mean absolute SHAP value per feature, indicating which features drive model predictions globally. Bar plots and waterfall plots are generated per model and saved in the run’s `figures/` directory. The dashboard Explainability page displays these for the selected model.

*[INSERT FIGURE 6.5: SHAP summary plot for one model (e.g., Random Forest or best model).]*  
*[INSERT FIGURE 6.6: SHAP bar plot (mean |SHAP|) for the same model.]*  
*[INSERT FIGURE 6.7: SHAP waterfall for one instance (optional).]*

### 6.2.2 LIME

LIME explanations are stored in `lime_explanations_<model>.json` with feature weights per instance. The dashboard allows selecting an instance and viewing the top contributing features. For the report, include a short table or list of LIME feature weights for one or two instances and one model.

*[INSERT FIGURE 6.8: LIME feature weights for a selected instance (table or screenshot).]*

### 6.2.3 Natural-Language Explanations

The Explainability Agent generates natural-language summaries (e.g., “The model predicts Disease with probability 0.72 because features X, Y, Z contribute positively”). The dashboard displays these in the Explainability and Prediction Explainer pages. Include an example in the thesis to illustrate the kind of interpretability offered.

*[INSERT FIGURE 6.9: Example natural-language explanation (screenshot or pasted text).]*

### 6.2.4 Explainability Evaluation Metrics

If the explainability evaluator was run, report fidelity and readability scores from `explainability_evaluation.json` (e.g., per-model readability). A short table or paragraph is sufficient.

---

## 6.3 Collaboration Efficiency

### 6.3.1 Execution Times

Table 6.3 reports the execution time per agent (Data, Model, Explainability) and total pipeline time for the multi-agent run, and the total time for the baseline. The dashboard Efficiency page shows an agent timeline bar chart with distinct colours per agent.

**Table 6.3 – Execution times (example run)**

| Component        | Time (s) |
|------------------|----------|
| Data Agent       | —        |
| Model Agent      | —        |
| Explainability   | —        |
| Total (multi-agent) | —    |
| Baseline (total) | —        |
| Overhead         | — %      |

*[INSERT FIGURE 6.10: Agent execution times bar chart (Efficiency page).]*

### 6.3.2 Handover and Error Analysis

Report the number of handovers and the average handover time (if recorded), and the number of errors and error rate. For a successful run, errors should be zero and success rate 100%. The dashboard displays these in the Efficiency page under Handover Analysis and Error Analysis.

### 6.3.3 Efficiency Metrics Summary

Include pipeline status, success rate, total agents (4), and collaboration overhead (%). These values are shown in the Efficiency Metrics section of the dashboard and can be copied into the thesis.

---

## 6.4 Summary of Results

- **RQ1 (Accuracy):** [Summarise whether multi-agent achieved comparable or better accuracy than baseline; mention best model and ROC-AUC.]  
- **RQ2 (Explainability):** [Summarise SHAP/LIME outputs and any fidelity/readability metrics.]  
- **RQ3 (Efficiency):** [Summarise execution times and overhead; note that multi-agent incurs coordination and explainability cost.]

*[You can duplicate Section 6.1–6.3 for a second dataset (e.g., Heart Disease) to show generality; add tables and figures for that dataset as needed.]*
