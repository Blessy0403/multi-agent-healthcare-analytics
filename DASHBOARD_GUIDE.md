# Streamlit Dashboard Guide

## Quick Start

1. **Run the pipeline first:**
   ```bash
   python main.py
   ```
   This generates outputs in `outputs/runs/<run_id>/`

2. **Launch the dashboard:**
   ```bash
   streamlit run dashboard/app.py
   ```
   Or use the Makefile:
   ```bash
   make dashboard
   ```

## Dashboard Pages

### 1. Overview
- Run summary and metadata
- Best models per dataset
- Key performance indicators (KPIs)
- Baseline vs Multi-Agent comparison

### 2. Data
- Dataset profile and schema
- Missing values visualization
- Feature distributions
- Class imbalance analysis

### 3. Modeling
- Model leaderboard
- Detailed metrics per model
- Confusion matrices
- ROC curves

### 4. Explainability
- SHAP summary plots
- Feature importance bars
- Individual prediction waterfall plots
- LIME explanations
- Natural language explanations

### 5. Benchmarking
- Baseline vs Multi-Agent comparison tables
- Metric-by-metric analysis
- Statistical differences

### 6. Efficiency
- Agent execution timeline
- Handover analysis
- Error tracking
- Collaboration overhead metrics

### 7. Prediction Explainer
- Interactive prediction interface
- Select from test set or manual input
- Real-time SHAP and LIME explanations
- Export explanations as JSON

## Sidebar Controls

- **Run Selector**: Choose which pipeline run to view
- **Dataset Selector**: Switch between datasets (heart_disease, diabetes)
- **Model Selector**: Select which model to analyze

## Output Structure

The dashboard reads from:
```
outputs/runs/<run_id>/
├── data/              # Processed datasets
├── models/            # Trained models (.pkl)
├── explainability/    # SHAP/LIME data
├── figures/           # Visualization plots
├── reports/           # Metrics and reports
└── logs/              # Execution logs
```

## Troubleshooting

**No runs found:**
- Make sure you've run `python main.py` at least once
- Check that `outputs/runs/` directory exists

**Missing data:**
- Ensure the pipeline completed successfully
- Check the run logs in `outputs/runs/<run_id>/logs/`

**Dashboard not loading:**
- Verify all dependencies are installed: `pip install -r requirements.txt`
- Check Streamlit version: `streamlit --version`
