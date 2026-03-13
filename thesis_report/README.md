# Thesis Report Content — 100-Page Structure

This folder contains the **full thesis report content** for *Multi-Agent AI Collaboration for Explainable Healthcare Analytics*. The material is organised so that, with your own results, figures, and minor expansions, you can reach approximately **100 pages** when formatted (e.g., in Word or LaTeX).

## Files and Purpose

| File | Content | Approx. pages (when formatted) |
|------|---------|-------------------------------|
| **00_Title_Abstract_Acknowledgements.md** | Title, abstract, acknowledgements, TOC | 2–3 |
| **01_Introduction.md** | Background, problem, RQs, contributions, scope | 6–8 |
| **02_Literature_Review.md** | XAI, multi-agent, healthcare ML, evaluation | 8–12 |
| **03_Methodology.md** | Architecture, agents, models, evaluation dimensions | 10–14 |
| **04_System_Design_Implementation.md** | Stack, run structure, agents, dashboard | 8–10 |
| **05_Experimental_Setup.md** | Datasets, hardware, run protocol, metrics | 4–6 |
| **06_Results.md** | Accuracy, explainability, efficiency + figure/table placeholders | 15–25 |
| **07_Discussion.md** | Interpretation, limitations, threats, implications | 6–8 |
| **08_Conclusion_Future_Work.md** | Conclusion, future work | 3–4 |
| **09_References.md** | Bibliography | 2–3 |
| **10_Appendices.md** | Config, directory structure, samples, how to run | 5–8 |
| **FIGURES_AND_CHARTS_GUIDE.md** | Where to get every chart/table for the thesis | — |

**Total:** Roughly 70–100+ pages once you:
- Replace all `—` in tables with your run’s numbers.
- Insert the figures listed in **FIGURES_AND_CHARTS_GUIDE.md** (screenshots from the dashboard or PNGs from `outputs/runs/<run_id>/figures/`).
- Add captions and references in your preferred editor.
- Slightly expand or shorten sections to match your university’s requirements.

## How to Use

1. **Copy into Word/LaTeX:** Paste each chapter into your thesis template; apply your styles and numbering.  
2. **Fill results:** After running `python main.py`, open the dashboard, select your run, and copy metrics from the Modeling and Benchmarking pages into **Table 6.1** and **Table 6.2**. Use the Efficiency page and `reports/collaboration_evaluation.json` for **Table 6.3**.  
3. **Add figures:** Follow **FIGURES_AND_CHARTS_GUIDE.md** to capture or export each chart; insert them where the text says *[INSERT FIGURE …]*.  
4. **Expand if needed:** Add more dataset runs, extra tables (e.g., per-seed results), or longer discussion to reach 100 pages.  
5. **References:** Add any extra papers you cite to **09_References.md** and format according to your university’s guidelines.

## Quick Links to Dashboard for Figures

- **Home:** Pipeline flow, quick stats.  
- **Overview:** Best models, KPIs, comparison table.  
- **Data:** Schema, distributions, class pie.  
- **Modeling:** Leaderboard, confusion matrix, ROC curve.  
- **Explainability:** SHAP summary/bar/waterfall, LIME, NL text.  
- **Benchmarking:** Comparison table and bar chart.  
- **Efficiency:** Agent timeline, efficiency metrics.  
- **Prediction Explainer:** Prediction card, SHAP, NL.

Run the dashboard with:  
`streamlit run dashboard/app.py`  
then select your run and dataset to generate and capture all figures and numbers.
