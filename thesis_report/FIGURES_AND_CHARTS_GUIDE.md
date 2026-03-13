# Figures and Charts Guide for Thesis Report

This guide lists every chart and table you can capture from the dashboard (or from run files) for your thesis. Use it to fill in the *[INSERT FIGURE]* placeholders in Chapters 1–6 and 10.

---

## Dashboard Home (app.py)

| Figure | Description | How to capture |
|--------|-------------|----------------|
| **Pipeline flow** | Four agents: Data → Model → Explainability → Evaluation (coloured pills) | Screenshot the flow bar below the header. |
| **Quick Stats** | Total runs, selected run, dataset | Optional; include if you want to show the dashboard in use. |
| **Research scope expander** | Methodology summary in expandable section | Expand and screenshot for Appendix or Chapter 1. |

---

## Overview (1_Overview.py)

| Figure | Description | How to capture |
|--------|-------------|----------------|
| **Run summary metrics** | Run ID, Created, Status, Total Time (4 metric cards) | Screenshot the first row of metrics. |
| **Best Models cards** | Best model per dataset with ROC-AUC | Screenshot the “Best Models” section. |
| **KPIs** | Accuracy, ROC-AUC, F1, Precision (4 metrics) | Screenshot the “Key Performance Indicators” section. |
| **Comparison table** | DataFrame of comparison_report.csv | Screenshot or copy table to Word/LaTeX. |
| **Run log** | Text area with run.log content | Optional; for reproducibility. |

---

## Data (2_Data.py)

| Figure | Description | How to capture |
|--------|-------------|----------------|
| **Schema table** | Column names and types (Numeric/Categorical/Integer) | Screenshot or export as table. |
| **Missing values** | Bar chart or “No missing values” message | Screenshot. |
| **Feature distribution** | Histogram for selected feature | Select a feature (e.g., Glucose), screenshot Plotly chart. |
| **Class distribution** | Pie chart (train set) | Screenshot the “Class Distribution” chart. |

**Suggested for thesis:** Schema table (Table 5.1 or Appendix), one feature histogram, class distribution pie.

---

## Modeling (3_Modeling.py)

| Figure | Description | How to capture |
|--------|-------------|----------------|
| **Model leaderboard** | Table of all 6 models with Accuracy, ROC-AUC, F1, Precision, Recall | Screenshot or copy → **Table 6.1 / Figure 6.4**. |
| **Metrics for selected model** | 5 metric cards (Accuracy, Precision, Recall, F1, ROC-AUC) | Optional. |
| **Confusion matrix** | Heatmap (model-specific colour) | Screenshot → **Figure 6.2**. |
| **ROC curve** | ROC with AUC; model-specific colour | Screenshot → **Figure 6.1**. |

**Suggested:** Leaderboard table, confusion matrix for best model, ROC curve for best model. Repeat for second dataset if needed.

---

## Explainability (4_Explainability.py)

| Figure | Description | How to capture |
|--------|-------------|----------------|
| **SHAP summary** | Summary plot (beeswarm or bar) for selected model | Screenshot → **Figure 6.5**. |
| **SHAP bar** | Mean \|SHAP\| bar chart | Screenshot → **Figure 6.6**. |
| **SHAP waterfall** | Waterfall for one instance (select in dropdown) | Screenshot → **Figure 6.7**. |
| **LIME** | Feature weights for selected instance | Screenshot or copy table → **Figure 6.8**. |
| **Natural language** | Text area with NL explanations for model | Screenshot or paste text → **Figure 6.9**. |

**Suggested:** At least one SHAP summary, one SHAP bar, one LIME example, one NL explanation.

---

## Benchmarking (5_Benchmarking.py)

| Figure | Description | How to capture |
|--------|-------------|----------------|
| **Comparison table** | comparison_report.csv as DataFrame | Screenshot or copy → **Table 6.2**. |
| **Summary metrics** | Multi-Agent Avg ROC-AUC, Baseline Avg ROC-AUC, Difference | Optional. |
| **Detailed comparison chart** | Grouped bar chart (Multi Agent = blue, Baseline = gray, Difference = purple) | Screenshot → **Figure 6.3**. |

**Suggested:** Comparison table and the comparison bar chart.

---

## Efficiency (6_Efficiency.py)

| Figure | Description | How to capture |
|--------|-------------|----------------|
| **Agent execution times** | Bar chart (one bar per agent, distinct colours) | Screenshot → **Figure 6.10**. |
| **Handover analysis** | Number of handovers, Avg handover time | Optional; can go in Table 6.3. |
| **Error analysis** | Number of errors, Error rate | Optional. |
| **Efficiency metrics** | Pipeline status, Success rate, Total agents, Overhead | Screenshot or copy into **Table 6.3**. |

**Suggested:** Agent timeline bar chart and the four efficiency metrics (for Table 6.3).

---

## Prediction Explainer (7_Prediction_Explainer.py)

| Figure | Description | How to capture |
|--------|-------------|----------------|
| **Prediction card** | Styled “Predicted class” (No Disease / Disease) with confidence | Screenshot → optional figure for Chapter 6 or 7. |
| **SHAP explanation** | Bar or waterfall for the selected instance | Optional; can reuse style from Explainability page. |
| **Natural language** | Explanation for that prediction | Optional. |

**Suggested:** One prediction card example (green for No Disease, amber for Disease).

---

## Files You Can Use Directly (No Screenshot)

- **model_metrics.json** → Fill **Table 6.1** (validation metrics).  
- **comparison_report.csv** → Fill **Table 6.2** (multi-agent vs baseline).  
- **collaboration_evaluation.json** or **logs/collaboration_metrics.json** → Fill **Table 6.3** (execution times, overhead).  
- **data_metadata_<dataset>.json** → Dataset schema and stats for **Table 5.1**.  
- **figures/*.png** → SHAP summary/bar/waterfall can be inserted as figures directly from the run folder.

---

## Suggested Order for a 100-Page Thesis

1. **Chapters 1–2:** 1–2 figures (pipeline diagram, optional XAI table).  
2. **Chapter 3:** 1 flow diagram, 1 hyperparameter table.  
3. **Chapter 4:** 2 dashboard screenshots (home, Modeling).  
4. **Chapter 5:** 1 dataset table.  
5. **Chapter 6:** 10–12 figures + 3–4 tables (main results).  
6. **Appendices:** 1–2 structure/configuration figures or tables.

Adding full narrative (as in the chapter files), figure captions, and references will bring the document to a substantial length. Expand Results (Section 6) with your actual numbers and short analyses to reach your target page count.
