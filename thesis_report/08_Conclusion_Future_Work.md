# Chapter 8: Conclusion and Future Work

## 8.1 Conclusion

This thesis set out to investigate whether a multi-agent AI system with dedicated roles for data processing, predictive modelling, and explanation could deliver accurate and interpretable healthcare analytics compared to a traditional single-pipeline baseline. We designed and implemented a four-agent pipeline (Data, Model, Explainability, Evaluation) and a six-model comparative suite (Logistic Regression, Random Forest, XGBoost, SVM, Gradient Boosting, KNN), integrated SHAP and LIME for explainability, and evaluated the system along three dimensions: predictive accuracy, explainability quality, and collaboration efficiency.

Our findings indicate that **(1)** the multi-agent pipeline achieves predictive performance comparable to the baseline, with the best model (e.g., KNN or Gradient Boosting) reaching competitive ROC-AUC on the UCI Diabetes and Heart Disease datasets; **(2)** explainability is successfully integrated through SHAP summary and waterfall plots, LIME feature weights, and natural-language explanations, providing auditors and clinicians with multiple views of model behaviour; and **(3)** the multi-agent design incurs a measurable overhead in execution time due to coordination and the added explainability step, which is acceptable for a research prototype and can be optimised in future work.

We conclude that agentic, explainable healthcare analytics is feasible and that the proposed architecture offers a clear separation of concerns, reproducibility, and a path toward more transparent decision support in healthcare.

## 8.2 Future Work

- **Extended evaluation:** Multiple runs with different random seeds; statistical tests (e.g., paired t-test) for multi-agent vs. baseline; additional datasets (e.g., MIMIC, eICU) where legally and ethically possible.  
- **Deep learning:** Integrate neural networks (e.g., MLP, 1D-CNN for tabular data) with appropriate explainability (e.g., Integrated Gradients, SHAP for deep models).  
- **User studies:** Conduct studies with clinicians to assess the perceived usefulness and readability of SHAP/LIME and natural-language explanations.  
- **Deployment:** Consider API deployment, access control, and audit logging for use in clinical or research environments.  
- **Automated evaluation agent:** Extend the Evaluation Agent to run A/B tests or continuous monitoring and to produce automated reports for stakeholders.

---

*[Optional: Short paragraph on the contribution to the field and closing remark.]*
