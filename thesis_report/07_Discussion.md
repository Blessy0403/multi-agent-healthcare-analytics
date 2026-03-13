# Chapter 7: Discussion

## 7.1 Interpretation of Results

### 7.1.1 Predictive Performance

[Interpret your results: Which model(s) performed best on Diabetes/Heart Disease and why? How does multi-agent accuracy compare to the baseline? If they are similar, this supports the claim that the multi-agent design does not sacrifice accuracy. If there are differences, discuss possible causes (e.g., random seed, data split, tuning).]

### 7.1.2 Explainability

[Discuss the usefulness of SHAP and LIME in your setting: Do the top features align with clinical intuition? Are natural-language explanations clear? Mention limitations—e.g., post-hoc explanations do not guarantee causal interpretation.]

### 7.1.3 Efficiency and Overhead

[Discuss the overhead of the multi-agent pipeline (e.g., ~4× longer than baseline in the example run). Is this acceptable for a research prototype? What would be needed to reduce it (e.g., fewer models, smaller grids, parallel agents)?]

## 7.2 Limitations

- **Datasets:** Public UCI data; not necessarily representative of real clinical populations or regulatory requirements.  
- **Models:** Classical ML only; no deep learning or time-series models.  
- **Explainability:** SHAP/LIME are approximations; stability and fidelity can vary.  
- **Single run vs. repeated runs:** For a rigorous thesis, consider reporting mean and standard deviation over multiple runs with different seeds.  
- **User study:** No formal user study with clinicians; readability and usefulness of explanations are not empirically validated.

## 7.3 Threats to Validity

- **Internal:** Hyperparameter grids and split ratios may favour certain models; document choices and consider sensitivity analysis.  
- **External:** Generalisation to other hospitals or countries is limited without further data.  
- **Construct:** ROC-AUC and F1 are standard but may not capture all aspects of clinical utility (e.g., cost of false negatives).

## 7.4 Implications for Practice and Research

- **Practice:** The pipeline demonstrates that an explainable, multi-agent analytics workflow is feasible and can produce both accurate predictions and interpretable outputs; deployment would require integration, validation, and regulatory steps.  
- **Research:** The framework can be extended with more agents (e.g., dedicated evaluation agent that also runs A/B tests), more datasets, or deep learning models with appropriate explainability methods.

---

*[Optional: Include a short subsection on ethical considerations—fairness, privacy, accountability—and how explainability and logging support them.]*
