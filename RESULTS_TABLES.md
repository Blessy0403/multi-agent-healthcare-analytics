# Pipeline comparison: Baseline vs Multi-Agent

Tables below are generated from the latest pipeline runs. Re-run `python scripts/generate_results_tables.py` after new runs to refresh.

## Dataset links

- **Diabetes (Pima Indians):**
  - Primary: https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv
  - UCI fallback: https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data

- **Heart Disease (Cleveland):**
  - Primary: https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data
  - Fallback: https://raw.githubusercontent.com/plotly/datasets/master/heart.csv

---

## Diabetes Dataset

| Metric | Baseline Pipeline | Multi-Agent Pipeline |
|--------|-------------------|----------------------|
| Accuracy | 0.8237 | 0.8295 |
| Precision | 0.7273 | 0.7664 |
| Recall | 0.7586 | 0.7069 |
| F1 Score | 0.7426 | 0.7354 |
| ROC-AUC | 0.8945 | 0.8991 |

## Heart Disease Dataset

| Metric | Baseline Pipeline | Multi-Agent Pipeline |
|--------|-------------------|----------------------|
| Accuracy | 0.6934 | 0.7226 |
| Precision | 0.6667 | 0.7400 |
| Recall | 0.6452 | 0.5968 |
| F1 Score | 0.6557 | 0.6607 |
| ROC-AUC | 0.7886 | 0.7841 |
