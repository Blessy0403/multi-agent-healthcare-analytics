# Chapter 3: Methodology

## 3.1 Overall Architecture

The system is organised as a **four-agent pipeline** orchestrated in sequence:

1. **Data Agent** – Ingestion, cleaning, encoding, scaling, and train/validation/test splitting.  
2. **Model Agent** – Training and hyperparameter tuning of six classifiers; selection of the best model by ROC-AUC.  
3. **Explainability Agent** – SHAP (tree and kernel), LIME, natural-language explanations, and plot generation.  
4. **Evaluation Agent** – Comparative evaluation (multi-agent vs. baseline), explainability metrics, and collaboration efficiency reporting.

Artifacts (CSV, JSON, pickle, plots) are passed between agents via the file system under a run-specific directory. The orchestrator records execution times and handovers for later analysis.

## 3.2 Data Agent

**Inputs:** Dataset name (e.g., diabetes, heart_disease), configuration (split ratios, augmentation factor).  
**Process:**  
- Download from UCI or configured URL; load with explicit column names if the file is headerless.  
- Clean: handle missing values (median for numeric, mode for categorical); for diabetes, treat invalid zeros as missing.  
- Encode categorical variables (label encoding); ensure binary target for classification.  
- Optionally augment data (e.g., 3×) to improve robustness.  
- Scale features (StandardScaler).  
- Split into train (70%), validation (15%), test (15%).  
**Outputs:** `train.csv`, `val.csv`, `test.csv`, `data_metadata.json` (schema, target, missing-value stats).

## 3.3 Model Agent

**Inputs:** Train and validation DataFrames, target column name.  
**Process:**  
- For each of the six models, run **GridSearchCV** with 5-fold cross-validation, scoring on ROC-AUC.  
- **Models and hyperparameter grids:**  
  - **Logistic Regression:** C, penalty, max_iter.  
  - **Random Forest:** n_estimators, max_depth, min_samples_split.  
  - **XGBoost:** n_estimators, max_depth, learning_rate.  
  - **SVM:** C, kernel (RBF, linear), gamma.  
  - **Gradient Boosting:** n_estimators, max_depth, learning_rate.  
  - **KNN:** n_neighbors, weights, metric.  
- Select the best model by validation ROC-AUC.  
- Save models (.pkl), metrics (accuracy, precision, recall, F1, ROC-AUC, confusion matrix), and validation predictions.  
**Outputs:** `model_metrics.json`, `validation_predictions.csv`, one `.pkl` per model.

## 3.4 Explainability Agent

**Inputs:** Trained models, validation (or sample) data, feature names.  
**Process:**  
- **SHAP:** For tree models use TreeExplainer; for others use KernelExplainer or Explainer API. Compute SHAP values and generate summary plot, bar plot (mean |SHAP|), and waterfall plots for selected instances.  
- **LIME:** LimeTabularExplainer; for each model, explain a subset of instances and save feature weights per instance in JSON.  
- **Natural language:** Generate short textual explanations (e.g., “The model predicts Disease with probability X because features A, B, C contribute positively/negatively”) and save per model.  
**Outputs:** SHAP plots (PNG), `lime_explanations_{model}.json`, `natural_language_explanations.txt`, `explanations.json`.

## 3.5 Evaluation Agent (Evaluation Framework)

**Inputs:** Multi-agent results (metrics, models, collaboration_metrics), baseline results (metrics, execution time).  
**Process:**  
- **Predictive accuracy:** Compare multi-agent vs. baseline per model (ROC-AUC, F1, etc.); produce comparison report (CSV) and summary.  
- **Explainability:** Evaluate fidelity (e.g., correlation between SHAP and model output), readability scores for natural-language explanations.  
- **Collaboration efficiency:** Per-agent execution times, handover count, error count, success rate, overhead vs. baseline.  
**Outputs:** `comparison_report.csv`, `explainability_evaluation.json`, `collaboration_evaluation.json`, updated `run_metadata.json`.

## 3.6 Baseline Pipeline

A **single-pipeline baseline** runs the same data preprocessing (reusing Data Agent outputs) and trains the same six models in a monolithic script without agent boundaries. No Explainability Agent is run in the baseline; only predictive metrics and total execution time are recorded. This allows a fair comparison: any difference in accuracy is due to pipeline structure (or random variation); any difference in time is attributed to coordination and explainability overhead.

## 3.7 Evaluation Dimensions and Metrics

| Dimension | Metrics |
|-----------|---------|
| **Predictive accuracy** | ROC-AUC, accuracy, precision, recall, F1; per model and best model. |
| **Explainability** | Fidelity (SHAP/LIME vs. model); readability (natural language); optional stability. |
| **Collaboration efficiency** | Total execution time; per-agent time; number of handovers; error count; success rate; overhead (%). |

---

*[INSERT FIGURE: Flow diagram of the four agents and artifact flow.]*  
*[INSERT TABLE: Hyperparameter grids for each of the six models.]*
