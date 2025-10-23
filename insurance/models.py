from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split

from .data import load_csv, split_features_target
from .eval import regression_metrics
from .preprocess import build_pipeline
TorchRegressor = None  # Lazy import; set when needed


def train_models(
    data_csv: str,
    outdir: str = "artifacts",
    target: str = "charges",
    test_size: float = 0.2,
    random_state: int = 42,
    models: List[str] = ["linear", "ridge", "rf", "xgb", "nn"],
) -> Dict:
    df = load_csv(data_csv)
    X, y = split_features_target(df, target=target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    registry: Dict[str, object] = {}
    metrics_records: List[Dict] = []

    def fit_eval(name: str, est) -> Tuple[object, Dict[str, float]]:
        pipe = build_pipeline(est)
        model = pipe.fit(X_train, y_train)
        preds = model.predict(X_test)
        m = regression_metrics(y_test, preds)
        return model, m

    for name in models:
        n = name.lower()
        if n == "linear":
            est = LinearRegression()
        elif n == "ridge":
            est = Ridge(random_state=random_state)
        elif n == "rf":
            est = RandomForestRegressor(random_state=random_state, n_estimators=400)
        elif n == "xgb":
            # Lazy import to avoid hard dependency if unused
            try:
                from xgboost import XGBRegressor  # type: ignore
            except Exception as e:
                raise ImportError("xgboost not installed; install to use 'xgb' model") from e
            est = XGBRegressor(
                n_estimators=500,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.8,
                random_state=random_state,
            )
        elif n == "nn":
            # Lazy import PyTorch wrapper
            global TorchRegressor
            if TorchRegressor is None:
                try:
                    from .torch_regressor import TorchRegressor as _TR  # type: ignore
                except Exception as e:
                    raise ImportError("PyTorch not installed; install to use 'nn' model") from e
                TorchRegressor = _TR
            est = TorchRegressor(epochs=120, hidden=128, dropout=0.05, random_state=random_state)
        else:
            continue

        model, m = fit_eval(n, est)
        registry[n] = model
        metrics_records.append({"model": n, **m})

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    # Save all models
    for name, model in registry.items():
        joblib.dump(model, out / f"model_{name}.joblib")

    metrics_df = pd.DataFrame(metrics_records).sort_values("rmse")
    metrics_df.to_csv(out / "model_metrics.csv", index=False)

    best_row = metrics_df.iloc[0]
    best_name = str(best_row["model"]).lower()
    joblib.dump(registry[best_name], out / "model_best.joblib")

    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_model": best_name,
                "metrics": metrics_df.to_dict(orient="records"),
                "target": target,
            },
            f,
            indent=2,
        )

    return {"best_model": best_name, "metrics": metrics_records}
