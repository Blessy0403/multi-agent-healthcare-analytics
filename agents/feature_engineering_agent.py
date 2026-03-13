"""
Feature Engineering Agent: Builds derived features from DataAgent output.

- Missingness indicators for columns with missing values.
- Pairwise interactions among top N numeric columns by variance (N=6).
- Non-linear transforms (log1p(abs(x)), x^2) for up to 8 numeric columns.
- One-hot encoding for categoricals using train-fitted categories.
- All transforms use TRAIN-only statistics (no leakage); outputs are purely numeric.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.json_utils import json_safe
from utils.logging import AgentLogger


# Caps for deterministic, safe feature expansion
MAX_INTERACTION_COLS = 6
MAX_NONLINEAR_COLS = 8


class FeatureEngineeringAgent:
    """
    Agent that adds engineered features to train/val/test from DataAgent.
    Uses only train statistics; preserves target; ensures aligned columns and no NaNs.
    """

    def __init__(self, config: Optional[Any] = None):
        self.config = config
        self.logger = AgentLogger("feature_engineering_agent")

    def run(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestrator entry: run feature engineering using artifacts["data"].

        Args:
            artifacts: Must contain "data" with train_df, val_df, test_df, target_column.

        Returns:
            artifacts["features"]: train_df, val_df, test_df, target_column, metadata, file_paths.
        """
        self.logger.info("FeatureEngineeringAgent run started")
        result = self.process(artifacts)
        self.logger.info("FeatureEngineeringAgent run completed")
        return result

    def process(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute feature engineering: load data from artifacts["data"], engineer, save, return.
        """
        with self.logger.execution_timer():
            data = artifacts.get("data") or {}
            train_df = data.get("train_df")
            val_df = data.get("val_df")
            test_df = data.get("test_df")
            target_col = (
                data.get("target_column")
                or (data.get("metadata") or {}).get("target_column")
            )

            if train_df is None or val_df is None or not target_col:
                raise ValueError(
                    "FeatureEngineeringAgent requires artifacts['data'] with train_df, val_df, target_column."
                )
            if target_col not in train_df.columns:
                raise ValueError(f"Target column '{target_col}' not in train_df.")

            run_dir = artifacts.get("run_dir")
            if run_dir is None and getattr(self.config, "run_dir", None) is not None:
                run_dir = self.config.run_dir
            run_dir = Path(run_dir) if run_dir is not None else Path(".")
            reports_dir = run_dir / "reports"
            data_dir = run_dir / "data"
            reports_dir.mkdir(parents=True, exist_ok=True)
            data_dir.mkdir(parents=True, exist_ok=True)

            # Optional test_df (some pipelines may not have it)
            if test_df is None:
                test_df = val_df.iloc[0:0].copy()

            train_eng, val_eng, test_eng, meta = self._engineer(
                train_df, val_df, test_df, target_col
            )

            # Sanity: identical feature columns (excluding target); align order
            feat_cols = [c for c in train_eng.columns if c != target_col]
            cols_order = feat_cols + [target_col] if target_col in train_eng.columns else feat_cols

            for name, df in [("val", val_eng), ("test", test_eng)]:
                df_feat = [c for c in df.columns if c != target_col]
                assert set(df_feat) == set(feat_cols), (
                    f"Feature columns mismatch for {name}: "
                    f"extra={set(df_feat) - set(feat_cols)}, missing={set(feat_cols) - set(df_feat)}"
                )
            # Reorder all to same column order (add missing with 0 for safety)
            for c in cols_order:
                if c not in val_eng.columns:
                    val_eng[c] = 0
                if c not in test_eng.columns:
                    test_eng[c] = 0
            train_eng = train_eng.reindex(columns=cols_order)
            val_eng = val_eng.reindex(columns=cols_order)
            test_eng = test_eng.reindex(columns=cols_order)

            # NaNs: impute with train median and log
            for col in feat_cols:
                if col not in train_eng.columns:
                    continue
                train_med = train_eng[col].median()
                for label, df in [("train", train_eng), ("val", val_eng), ("test", test_eng)]:
                    if col not in df.columns:
                        continue
                    nans = df[col].isna().sum()
                    if nans > 0:
                        if label == "train":
                            fill = train_med if pd.notna(train_med) else 0.0
                        else:
                            fill = train_med if pd.notna(train_med) else 0.0
                        df[col] = df[col].fillna(fill)
                        meta["nan_imputation"] = meta.get("nan_imputation", [])
                        meta["nan_imputation"].append(
                            {"column": col, "split": label, "filled_count": int(nans), "value": float(fill)}
                        )
                        self.logger.info(
                            f"Imputed {nans} NaNs in '{col}' ({label}) with train median: {fill}"
                        )

            # Final assert: no NaNs in feature columns
            for label, df in [("train", train_eng), ("val", val_eng), ("test", test_eng)]:
                feat_only = [c for c in feat_cols if c in df.columns]
                if feat_only:
                    remaining = df[feat_only].isna().sum().sum()
                    assert remaining == 0, (
                        f"After imputation, {label} still has {remaining} NaNs in features"
                    )

            # Save outputs
            report_path = reports_dir / "feature_engineering_report.json"
            train_path = data_dir / "engineered_train.csv"
            val_path = data_dir / "engineered_val.csv"
            test_path = data_dir / "engineered_test.csv"

            with open(report_path, "w") as f:
                json.dump(meta, f, indent=2, default=json_safe)

            train_eng.to_csv(train_path, index=False)
            val_eng.to_csv(val_path, index=False)
            test_eng.to_csv(test_path, index=False)

            self.logger.log_artifact("report", str(report_path), {"type": "feature_engineering_report"})
            self.logger.log_artifact("dataset", str(train_path), {"split": "train", "engineered": True})
            self.logger.log_artifact("dataset", str(val_path), {"split": "val", "engineered": True})
            self.logger.log_artifact("dataset", str(test_path), {"split": "test", "engineered": True})

            result = {
                "train_df": train_eng,
                "val_df": val_eng,
                "test_df": test_eng,
                "metadata": meta,
                "file_paths": {
                    "train": train_path,
                    "val": val_path,
                    "test": test_path,
                    "metadata": report_path,
                },
                "target_column": target_col,
            }
            return result

    def _engineer(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Add missingness indicators, interactions (top 6 numeric by variance),
        non-linear transforms (top 8 numeric), and one-hot for categoricals.
        All fitted on train only; apply consistently to val/test.
        """
        meta: Dict[str, Any] = {
            "missingness_indicators": [],
            "interaction_pairs": [],
            "nonlinear_columns": [],
            "onehot_columns": [],
        }

        def feature_cols(df: pd.DataFrame) -> List[str]:
            return [c for c in df.columns if c != target_col]

        # 1) One-hot encode categoricals using train categories
        train = train_df.copy()
        val = val_df.copy()
        test = test_df.copy()

        cat_cols = list(
            train.select_dtypes(include=["object", "category"]).columns
        )
        cat_cols = [c for c in cat_cols if c != target_col]

        if cat_cols:
            for col in cat_cols:
                train[col] = train[col].astype(str)
                val[col] = val[col].astype(str) if col in val.columns else ""
                test[col] = test[col].astype(str) if col in test.columns else ""
                categories = sorted(train[col].dropna().unique().tolist())
                for cat in categories:
                    new_col = f"{col}__{cat}"
                    train[new_col] = (train[col] == cat).astype(int)
                    val[new_col] = (val[col] == cat).astype(int) if col in val.columns else 0
                    test[new_col] = (test[col] == cat).astype(int) if col in test.columns else 0
                    meta["onehot_columns"].append(new_col)
                train = train.drop(columns=[col])
                if col in val.columns:
                    val = val.drop(columns=[col])
                if col in test.columns:
                    test = test.drop(columns=[col])

        # 2) Missingness indicators (columns that had missing in train)
        numeric_cols = list(
            train.select_dtypes(include=[np.number]).columns
        )
        numeric_cols = [c for c in numeric_cols if c != target_col]

        missing_cols = [c for c in numeric_cols if train[c].isna().any()]
        for col in missing_cols:
            ind_name = f"{col}__missing"
            train[ind_name] = train[col].isna().astype(int)
            val[ind_name] = val[col].isna().astype(int) if col in val.columns else 0
            test[ind_name] = test[col].isna().astype(int) if col in test.columns else 0
            meta["missingness_indicators"].append(ind_name)

        # Fill missing in numeric for variance computation (in-place for train only for stats)
        train_fill = train.copy()
        for c in numeric_cols:
            if c in train_fill.columns and train_fill[c].isna().any():
                train_fill[c] = train_fill[c].fillna(train_fill[c].median())

        # 3) Top N numeric by variance (train) -> pairwise interactions (N=6)
        var_series = train_fill[numeric_cols].var()
        var_series = var_series.replace(0, np.nan).dropna()
        top_n = min(MAX_INTERACTION_COLS, len(var_series))
        if top_n > 0:
            top_cols = var_series.nlargest(top_n).index.tolist()
            for i, a in enumerate(top_cols):
                for b in top_cols[i + 1 :]:
                    if a not in train.columns or b not in train.columns:
                        continue
                    name = f"{a}_x_{b}"
                    train[name] = train[a] * train[b]
                    val[name] = val[a] * val[b] if a in val.columns and b in val.columns else 0
                    test[name] = test[a] * test[b] if a in test.columns and b in test.columns else 0
                    meta["interaction_pairs"].append(name)

        # 4) Non-linear transforms for up to 8 numeric columns (train variance order)
        n_nonlinear = min(MAX_NONLINEAR_COLS, len(numeric_cols))
        if n_nonlinear > 0:
            cols_for_nl = var_series.nlargest(n_nonlinear).index.tolist()
            meta["nonlinear_columns"] = cols_for_nl
            for col in cols_for_nl:
                if col not in train.columns:
                    continue
                # log1p(abs(x))
                ln_name = f"{col}__log1p_abs"
                train[ln_name] = np.log1p(np.abs(train[col].fillna(0)))
                val[ln_name] = np.log1p(np.abs(val[col].fillna(0))) if col in val.columns else 0
                test[ln_name] = np.log1p(np.abs(test[col].fillna(0))) if col in test.columns else 0
                # x^2
                sq_name = f"{col}__sq"
                train[sq_name] = train[col].fillna(0) ** 2
                val[sq_name] = (val[col].fillna(0) ** 2) if col in val.columns else 0
                test[sq_name] = (test[col].fillna(0) ** 2) if col in test.columns else 0

        # Align val/test to train columns (fill missing with 0)
        train_cols = list(train.columns)
        val = val.reindex(columns=train_cols, fill_value=0)
        test = test.reindex(columns=train_cols, fill_value=0)
        # Ensure target is last (for consistency)
        if target_col in train_cols:
            order = [c for c in train_cols if c != target_col] + [target_col]
            train = train.reindex(columns=order)
            val = val.reindex(columns=order)
            test = test.reindex(columns=order)

        return train, val, test, meta
