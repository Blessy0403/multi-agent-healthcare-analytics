"""
Canonical feature schema for cross-dataset evaluation.

Maps dataset-specific column names to a common set of canonical names so that
Framingham, Heart Disease, and Diabetes (Pima) can be compared with overlap > 1 feature.
"""

from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np


CANONICAL_FEATURES = [
    "age",
    "sex",
    "systolic_bp",
    "diastolic_bp",
    "cholesterol",
    "glucose",
    "bmi",
    "heart_rate",
    "smoker",
]

# Per-dataset: original column name (lowercase) -> canonical name.
DATASET_RENAME_MAP: Dict[str, Dict[str, str]] = {
    "framingham": {
        "age": "age",
        "male": "sex",
        "sex": "sex",
        "sysbp": "systolic_bp",
        "sysbp_": "systolic_bp",
        "diabp": "diastolic_bp",
        "diabp_": "diastolic_bp",
        "totchol": "cholesterol",
        "glucose": "glucose",
        "bmi": "bmi",
        "heartrate": "heart_rate",
        "heartrate_": "heart_rate",
        "currentsmoker": "smoker",
        "currentsmoker_": "smoker",
    },
    "heart_disease": {
        "age": "age",
        "sex": "sex",
        "trestbps": "systolic_bp",
        "chol": "cholesterol",
        "thalach": "heart_rate",
        "fbs": "glucose",
    },
    "diabetes": {
        "age": "age",
        "blood_pressure": "diastolic_bp",
        "bmi": "bmi",
        "glucose": "glucose",
    },
}


def _normalize_sex(series: pd.Series) -> pd.Series:
    """Ensure sex is numeric 0/1. Handle string categories safely."""
    out = series.copy()
    if pd.api.types.is_numeric_dtype(out):
        out = out.fillna(0).astype(float)
        out = (out > 0.5).astype(int)
        return out
    s = out.astype(str).str.lower().str.strip()
    out = np.where(s.isin(["1", "male", "m", "true", "yes"]), 1, 0)
    return pd.Series(out, index=series.index, dtype=int)


def canonicalize_features(
    df: pd.DataFrame,
    dataset_name: str,
    target_column: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Rename columns to canonical names, keep only canonical columns that exist,
    drop rows with NaNs in used canonical columns, ensure sex is 0/1 and numeric.

    Args:
        df: DataFrame with dataset-specific columns (may include target).
        dataset_name: Key for DATASET_RENAME_MAP (e.g. "framingham", "heart_disease", "diabetes").
        target_column: If present, exclude from feature canonicalization and preserve in output.

    Returns:
        (df_canon, used_cols, missing_cols)
        - df_canon: DataFrame with only canonical feature columns (and target if provided), numeric, no NaN in used cols.
        - used_cols: List of canonical column names present and used.
        - missing_cols: List of canonical feature names that were not in the dataset.
    """
    dataset_name = (dataset_name or "").strip().lower()
    cols_lower = {c.lower(): c for c in df.columns}
    if not dataset_name or dataset_name not in DATASET_RENAME_MAP:
        rename_map = {}
    else:
        rev = DATASET_RENAME_MAP[dataset_name]
        rename_map = {}
        for orig, canon in rev.items():
            lo = orig.lower()
            if canon in CANONICAL_FEATURES and lo in cols_lower:
                rename_map[cols_lower[lo]] = canon
        for c in CANONICAL_FEATURES:
            if c in df.columns:
                rename_map[c] = c
            for orig, canon in rev.items():
                if canon == c and orig.lower() in cols_lower and cols_lower[orig.lower()] not in rename_map:
                    rename_map[cols_lower[orig.lower()]] = c

    out = df.copy()
    if rename_map:
        out = out.rename(columns=rename_map)
    used_cols = [c for c in CANONICAL_FEATURES if c in out.columns]
    missing_cols = [c for c in CANONICAL_FEATURES if c not in out.columns]

    if not used_cols:
        return pd.DataFrame(), [], CANONICAL_FEATURES

    # Keep only used canonical feature columns + target if present
    keep = [c for c in used_cols if c in out.columns]
    if target_column and target_column in out.columns:
        keep = keep + [target_column]
    out = out[[c for c in keep if c in out.columns]].copy()

    # Ensure sex is 0/1
    if "sex" in out.columns:
        out["sex"] = _normalize_sex(out["sex"])

    # Coerce to numeric; errors -> NaN
    for col in used_cols:
        if col not in out.columns:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Drop rows with NaN only in used canonical columns (preserve target if present)
    feature_cols = [c for c in used_cols if c in out.columns]
    out = out.dropna(subset=feature_cols)

    return out, used_cols, missing_cols
