from __future__ import annotations

from typing import Dict

import joblib
import numpy as np
import pandas as pd


def residuals_by_group(model_path: str, df: pd.DataFrame, target: str = "charges", group: str = "sex") -> pd.DataFrame:
    pipe = joblib.load(model_path)
    y_true = df[target].values
    X = df.drop(columns=[target])
    preds = pipe.predict(X)
    resid = y_true - preds
    out = pd.DataFrame({group: df[group].values, "residual": resid, "abs_err": np.abs(resid)})
    return out.groupby(group).agg(count=("residual", "count"), mean_resid=("residual", "mean"), mae=("abs_err", "mean")).reset_index()

