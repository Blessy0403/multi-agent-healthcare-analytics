# Quick Start Guide

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run the pipeline:**
```bash
python main.py
```

## Expected Execution Time

- **Data Agent**: ~10-30 seconds (download + processing)
- **Model Agent**: ~2-5 minutes (hyperparameter tuning)
- **Explainability Agent**: ~1-3 minutes (SHAP + LIME)
- **Baseline**: ~2-5 minutes
- **Evaluation**: ~10 seconds

**Total**: ~5-15 minutes depending on hardware

## Troubleshooting

### ModuleNotFoundError

If you see `ModuleNotFoundError`, install dependencies:
```bash
pip install -r requirements.txt
```

### Dataset Download Issues

If dataset download fails:
- Check internet connection
- UCI repository may be temporarily unavailable
- Try again later or use cached data if available

### Memory Issues

If you encounter memory errors:
- Reduce `shap_sample_size` in `utils/config.py`
- Reduce hyperparameter grid sizes
- Use smaller dataset (diabetes has fewer samples)

## Output Verification

After execution, verify outputs:

```bash
# Check data processing
ls -lh data/processed/

# Check models
ls -lh outputs/models/

# Check explanations
ls -lh outputs/explanations/
ls -lh outputs/plots/

# Check results
ls -lh outputs/results/
```

## Next Steps

1. Review `outputs/results/comparison_report.csv` for accuracy comparison
2. Review `outputs/results/explainability_evaluation.json` for explainability metrics
3. Review `outputs/results/collaboration_evaluation.json` for collaboration efficiency
4. Check `outputs/plots/` for SHAP visualizations
5. Read `outputs/explanations/natural_language_explanations.txt` for human-readable explanations
