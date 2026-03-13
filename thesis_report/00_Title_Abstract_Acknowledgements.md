# Multi-Agent AI Collaboration for Explainable Healthcare Analytics

---

## Title Page

**Title:** Multi-Agent AI Collaboration for Explainable Healthcare Analytics

**Author:** Blessy Evangeline Aaron

**Enrolment Number:** 11038695

**Supervisor:** Prof. Mehrdad Jalali

**Degree:** Master of Science

**Institution:** [Your University Name]

**Submission Date:** [Month Year]

---

## Abstract

This thesis investigates the application of multi-agent AI systems in healthcare data analytics, with a focus on explainability and comparative performance against traditional single-model approaches. Healthcare decision-support systems increasingly rely on data-driven models; however, many such systems operate as "black boxes," offering limited transparency to clinicians and hindering adoption in high-stakes clinical environments.

We propose and implement a four-agent pipeline comprising a Data Agent, Model Agent, Explainability Agent, and Evaluation Agent. Each agent performs a specialised role: data ingestion and preprocessing, training and tuning of a six-model comparative suite (Logistic Regression, Random Forest, XGBoost, Support Vector Machine, Gradient Boosting, and K-Nearest Neighbors), generation of local and global explanations using SHAP and LIME, and comprehensive evaluation. The system is evaluated along three dimensions—predictive accuracy, explainability quality, and collaboration efficiency—on public healthcare datasets (UCI Diabetes, UCI Heart Disease).

Results show that the multi-agent pipeline achieves competitive predictive performance (ROC-AUC, F1, accuracy) relative to a single-model baseline while providing structured, interpretable outputs. Explainability is delivered through SHAP summary and waterfall plots, LIME feature weights, and natural-language explanations. Collaboration overhead (execution time, handover analysis) is quantified and reported. The framework is implemented in Python with a Streamlit dashboard for reproducibility and thesis-level visualisation.

**Keywords:** Multi-agent systems, explainable AI, healthcare analytics, SHAP, LIME, machine learning, agentic AI, interpretability, diabetes prediction, heart disease classification.

---

## Acknowledgements

[Add your personal acknowledgements here: supervisor, family, colleagues, and any support received during the thesis work.]

I would like to thank my supervisor, Prof. Mehrdad Jalali, for his guidance and feedback throughout this research. I also thank [names] for their support during the development and evaluation phases. This work would not have been possible without the open-source healthcare datasets and the developers of the libraries used in this project.

---

## Table of Contents

1. Introduction  
2. Literature Review and State of the Art  
3. Methodology  
4. System Design and Implementation  
5. Experimental Setup  
6. Results  
7. Discussion  
8. Conclusion and Future Work  
9. References  
10. Appendices  

---

*[INSERT FIGURE: Optional – high-level pipeline diagram or dashboard overview screenshot as frontispiece.]*
