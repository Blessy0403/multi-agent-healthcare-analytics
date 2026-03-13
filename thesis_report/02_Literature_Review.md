# Chapter 2: Literature Review and State of the Art

## 2.1 Explainable AI in Healthcare

The need for interpretable and explainable AI in medicine has been widely recognised. Adadi and Berrada (2018) survey the field of explainable AI (XAI), distinguishing between *transparency* (model-inherent interpretability) and *post-hoc explainability* (methods applied after prediction). In healthcare, both regulatory considerations and clinician acceptance depend on the ability to justify and critique model outputs.

Holzinger et al. (2019) introduce the concept of *causability*—the extent to which an explanation supports human causal reasoning—and argue that explainability in medicine must go beyond statistical correlation to support decision-making under uncertainty. Our use of SHAP and LIME aligns with the post-hoc, local-explanation paradigm that many clinical applications require: explaining *why* a specific patient received a given prediction.

**SHAP (SHapley Additive exPlanations)** (Lundberg & Lee, 2017) grounds explanations in game-theoretic Shapley values, providing a consistent and locally accurate attribution of feature contributions. TreeSHAP enables efficient computation for tree-based models (Random Forest, XGBoost, Gradient Boosting); Kernel SHAP supports model-agnostic explanation for Logistic Regression and SVM. **LIME (Local Interpretable Model-agnostic Explanations)** (Ribeiro et al., 2016) approximates the model locally with an interpretable surrogate (e.g., linear model) and is widely used for tabular and text data. Both methods are well-suited to healthcare tabular datasets and are integrated into our Explainability Agent.

## 2.2 Multi-Agent Systems and Agentic AI

Multi-agent systems (MAS) distribute tasks among autonomous agents that communicate via messages or shared artifacts. In AI, "agentic" workflows have gained traction for research automation, code generation, and decision support. Agents can specialise in data gathering, reasoning, or tool use, with an orchestrator coordinating the flow.

In healthcare, multi-agent designs have been proposed for care coordination, diagnosis support, and resource scheduling. Our contribution is to apply an agentic decomposition specifically to the *analytics pipeline*: one agent for data, one for modelling, one for explanation, and one for evaluation. This separation allows each component to be developed, tested, and explained independently while maintaining a single runnable pipeline.

## 2.3 Machine Learning for Healthcare Prediction

Classical ML for healthcare classification commonly employs logistic regression (interpretable baseline), tree-based methods (Random Forest, Gradient Boosting, XGBoost), and kernel methods (SVM). Deep learning is increasingly used for imaging and sequences but often at the cost of interpretability. Our six-model suite—LR, RF, XGBoost, SVM, Gradient Boosting, KNN—covers interpretable baselines, ensembles, and kernel methods, enabling a comparative analysis that is standard in thesis and publication work.

Benchmarking multi-agent vs. single-pipeline performance is essential to answer RQ1. We implement a baseline that uses the same data, preprocessing, and models but without agent separation, so that any difference in metrics can be attributed to the pipeline structure (and overhead) rather than to the algorithms themselves.

## 2.4 Evaluation of Explainability and Collaboration

Evaluating explainability is challenging. Common approaches include *fidelity* (how well the explanation matches the model’s behaviour), *stability* (consistency under perturbation), and *readability* (human assessment or proxy metrics). Our Evaluation Agent and explainability evaluator compute fidelity and readability scores where applicable. Collaboration efficiency is measured by execution time per agent, number of handovers, error rate, and overhead relative to the baseline—addressing RQ3.

## 2.5 Research Gap and Position of This Work

Existing work often either focuses on accuracy without explainability, or on explainability without a multi-agent structure. This thesis combines (1) a clear multi-agent architecture, (2) a broad model suite, (3) integrated SHAP/LIME and natural-language explanations, and (4) a three-dimensional evaluation (accuracy, explainability, efficiency) in a single, reproducible framework. The result is a prototype that demonstrates the feasibility and trade-offs of agentic, explainable healthcare analytics and provides a foundation for future extension (e.g., more datasets, deep learning, deployment studies).

---

*[INSERT FIGURE: Summary table or conceptual map of XAI methods (SHAP, LIME) and their use in our pipeline.]*
