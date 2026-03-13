# Chapter 5: Experimental Setup

## 5.1 Datasets

### 5.1.1 UCI Diabetes (Pima Indians)

- **Source:** UCI ML Repository / Brownlee’s mirror.  
- **Task:** Binary classification (onset of diabetes within five years).  
- **Instances:** 768 (after loading); optional augmentation (e.g., 3×) increases training size.  
- **Features:** 8 (Pregnancies, Glucose, Blood pressure, Skin thickness, Insulin, BMI, Diabetes pedigree function, Age).  
- **Target:** Outcome (0/1).  
- **Preprocessing:** Invalid zeros (e.g., Glucose, BMI) treated as missing; median imputation; scaling; 70/15/15 split.

### 5.1.2 UCI Heart Disease (Cleveland)

- **Source:** UCI ML Repository.  
- **Task:** Binary classification (presence of heart disease).  
- **Features:** 13 (age, sex, cp, trestbps, chol, fbs, restecg, thalch, exang, oldpeak, slope, ca, thal).  
- **Target:** Original 0–4 mapped to binary (0 = no disease, >0 = disease).  
- **Preprocessing:** Missing value handling, encoding, scaling, same split ratio.

Additional datasets (e.g., breast cancer) can be added via configuration.

## 5.2 Hardware and Software

- **Environment:** Python 3.x, virtual environment recommended.  
- **Libraries:** See `requirements.txt` (pandas, numpy, scikit-learn, xgboost, shap, lime, streamlit, etc.).  
- **Hardware:** Experiments can be run on a single machine; GridSearchCV and multiple models may take several minutes per run (e.g., ~2–3 minutes for a full pipeline on diabetes with six models).

## 5.3 Run Protocol

1. Set dataset (e.g., diabetes) and random seed in configuration or environment.  
2. Execute `python main.py` once per run.  
3. Each run produces one run_id directory with all artifacts.  
4. Multiple runs can be compared in the dashboard by selecting different runs in the sidebar.

For the thesis, at least one full run on each dataset (e.g., diabetes, heart_disease) is recommended, with the dashboard used to capture figures and tables for the report.

## 5.4 Metrics Recorded

- **Per model:** Accuracy, precision, recall, F1, ROC-AUC, confusion matrix (TP, TN, FP, FN).  
- **Comparison:** Multi-agent vs. baseline for each metric; difference and percent difference where applicable.  
- **Explainability:** Fidelity and readability scores (see evaluation module).  
- **Efficiency:** Total time, per-agent time, handovers, errors, success rate, overhead percentage.

---

*[INSERT TABLE: Dataset statistics (number of samples, features, class balance) for diabetes and heart disease.]*
