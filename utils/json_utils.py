"""Safe JSON serialization for numpy, pandas, and other non-JSON-native types."""

import numpy as np
import pandas as pd


def json_safe(obj):
    # numpy scalars
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    # numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # pandas types
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, (pd.Series, pd.Index)):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")

    # sets/tuples
    if isinstance(obj, (set, tuple)):
        return list(obj)

    # fallback
    return str(obj)
