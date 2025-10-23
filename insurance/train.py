from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from .data import load_csv, split_features_target
from .preprocess import build_pipeline


def _build_estimator(name: str):
    n = name.lower()
    if n == "linear":
        return LinearRegression()
    if n == "ridge":
        return Ridge(random_state=42)
    if n == "rf":
        return RandomForestRegressor(random_state=42, n_estimators=300)
    raise ValueError(f"Unknown model '{name}'. Use one of: linear, ridge, rf")


def _param_distributions(name: str) -> Dict:
    n = name.lower()
    if n == "ridge":
        return {"model__alpha": np.logspace(-3, 3, 50)}
    if n == "rf":
        return {
            "model__n_estimators": [200, 300, 400, 600],
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", None, 0.7],
        }
    return {}


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}


def main():
    p = argparse.ArgumentParser(description="Train insurance cost model")
    p.add_argument("--data", required=True, help="Path to training CSV")
    p.add_argument("--target", default="charges", help="Target column (default=charges)")
    p.add_argument("--model", default="rf", help="Model: linear|ridge|rf (default=rf)")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--cv", type=int, default=5, help="Cross-validation folds")
    p.add_argument("--n-iters", type=int, default=30, help="Randomized search iterations")
    p.add_argument("--outdir", default="artifacts", help="Output directory for model & metrics")
    args = p.parse_args()

    df = load_csv(args.data)
    X, y = split_features_target(df, target=args.target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    est = _build_estimator(args.model)
    pipe = build_pipeline(est)

    params = _param_distributions(args.model)
    if params:
        search = RandomizedSearchCV(
            pipe,
            param_distributions=params,
            n_iter=args.n_iters,
            cv=args.cv,
            n_jobs=-1,
            random_state=args.random_state,
            refit=True,
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        best_params = search.best_params_
    else:
        model = pipe.fit(X_train, y_train)
        best_params = {}

    preds = model.predict(X_test)
    metrics = _evaluate(y_test, preds)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, outdir / "model.joblib")

    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "best_params": best_params}, f, indent=2)

    # Persist column info (helps later validation)
    with open(outdir / "columns.json", "w", encoding="utf-8") as f:
        json.dump({"feature_columns": list(X.columns), "target": args.target}, f, indent=2)

    print(json.dumps({"metrics": metrics, "best_params": best_params}, indent=2))


if __name__ == "__main__":
    main()
