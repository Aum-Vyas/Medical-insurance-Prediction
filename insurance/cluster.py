from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def fit_kmeans_on_preprocessed(model_path: str, X: pd.DataFrame, n_clusters: int = 4, outdir: str = "artifacts") -> Dict:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    pipe = joblib.load(model_path)
    pre = pipe.named_steps["pre"]
    # Use transform on the already-fitted preprocessor from the saved pipeline
    Xp = pre.transform(X)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(Xp)
    joblib.dump(km, out / "kmeans.joblib")
    pd.DataFrame({"cluster": labels}).to_csv(out / "cluster_labels.csv", index=False)
    return {"n_clusters": n_clusters, "counts": pd.Series(labels).value_counts().to_dict()}


def summarize_charges_by_cluster(labels: np.ndarray, y: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"cluster": labels, "charges": y.values})
    summary = df.groupby("cluster").agg(count=("charges", "count"), avg_charges=("charges", "mean")).reset_index()
    return summary
