# Chapter 1: Introduction

## 1.1 Background and Motivation

Healthcare is increasingly dependent on data-driven solutions for diagnostics, risk stratification, and treatment decisions. Machine learning (ML) models can achieve high accuracy on tasks such as disease prediction, readmission risk, and outcome forecasting. Nevertheless, many deployed systems remain opaque: they produce predictions without exposing the reasoning or the features that drove the outcome. This lack of interpretability limits trust among clinicians, complicates regulatory and ethical review, and can hinder adoption in sensitive, high-stakes environments such as hospitals and primary care.

The motivation for this thesis stems from the need for *trustworthy* AI in healthcare—systems that are not only accurate but also transparent and auditable. Multi-agent AI systems, in which specialised agents assume distinct roles (e.g., data preparation, model training, explanation generation), offer a structured way to separate concerns and to integrate explainability as a first-class component of the workflow. By coupling predictive models with post-hoc explainability methods such as SHAP (Lundberg & Lee, 2017) and LIME (Ribeiro et al., 2016), we aim to bridge the gap between black-box predictions and human-centred decision support.

## 1.2 Problem Statement

The central challenge addressed in this work is the **lack of transparency** in current healthcare ML systems. Single-agent or monolithic pipelines often deliver accurate predictions but do not explain their decisions in a way that is meaningful to medical professionals. This leads to:

- **Low trust** and limited adoption in clinical practice  
- **Regulatory and ethical concerns** regarding accountability and fairness  
- **Difficulty in debugging and improving** models when errors occur  

A further challenge is to understand whether a **multi-agent**, modular design—with explicit handovers between data, modelling, and explainability—can match or exceed the predictive performance of a traditional single-pipeline baseline while providing superior interpretability and a clear audit trail.

## 1.3 Research Questions

This thesis is guided by the following research questions:

**RQ1.** Can a multi-agent AI system, with clearly defined roles for data processing, diagnosis (predictive modelling), and explanation, achieve **predictive accuracy** comparable to or better than a traditional single-model pipeline on healthcare classification tasks?

**RQ2.** Does the integration of explainability agents (SHAP, LIME, natural-language explanations) **enhance interpretability** in a way that is measurable (e.g., fidelity, stability, readability) and useful for stakeholders?

**RQ3.** What is the **efficiency cost** of multi-agent collaboration—in terms of execution time, coordination overhead, and handover latency—and how does it compare to a monolithic baseline?

## 1.4 Contributions

The main contributions of this work are:

1. **Design and implementation of a four-agent pipeline** for healthcare analytics: Data Agent, Model Agent, Explainability Agent, and Evaluation Agent, with defined interfaces and artifact passing.

2. **A six-model comparative suite** with hyperparameter tuning (GridSearchCV): Logistic Regression, Random Forest, XGBoost, SVM, Gradient Boosting, and K-Nearest Neighbors, enabling a thorough comparison of interpretable and ensemble methods.

3. **Integration of explainability** into the agent workflow: SHAP (tree and kernel explainers), LIME, and natural-language explanation generation, with outputs suitable for thesis-level reporting and dashboards.

4. **A three-dimensional evaluation framework**: predictive accuracy (ROC-AUC, F1, precision, recall), explainability quality (fidelity, readability), and collaboration efficiency (per-agent times, success rate, overhead).

5. **An open, reproducible implementation** in Python with a Streamlit dashboard for visualisation, comparison reports, and interactive prediction explanation, supporting thesis writing and future extension.

## 1.5 Scope and Limitations

- **Datasets:** The experiments use publicly available UCI datasets (Diabetes, Heart Disease). Results are not intended to replace clinical validation on proprietary or regulatory-grade data.  
- **Models:** The focus is on classical ML (no deep learning); the pipeline is extensible to neural networks in future work.  
- **Explainability:** SHAP and LIME are post-hoc methods; we do not address inherently interpretable architectures (e.g., decision trees as sole model) in depth.  
- **Deployment:** The system is research-oriented; production deployment, security, and integration with hospital systems are out of scope.

## 1.6 Thesis Structure

- **Chapter 2** reviews related work in explainable AI, multi-agent systems, and healthcare ML.  
- **Chapter 3** presents the methodology: agent roles, model suite, explainability methods, and evaluation dimensions.  
- **Chapter 4** describes the system design and implementation (agents, orchestrator, baseline, dashboard).  
- **Chapter 5** details the experimental setup (datasets, metrics, hardware, runs).  
- **Chapter 6** reports results (accuracy, explainability, efficiency) with tables and references to figures.  
- **Chapter 7** discusses findings, limitations, and implications.  
- **Chapter 8** concludes and outlines future work.  
- **Appendices** provide configuration, directory structure, and sample outputs.

---

*[INSERT FIGURE: Conceptual diagram of the four-agent pipeline (Data → Model → Explainability → Evaluation).]*
