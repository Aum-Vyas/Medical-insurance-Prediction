from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay


def shap_summary_for_tree(model_path: str, X_sample: pd.DataFrame, outdir: str):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    model = joblib.load(model_path)
    try:
        pre = model.named_steps["pre"]
        Xp = pre.transform(X_sample)
        est = model.named_steps["model"]
        # Lazy imports
        import shap  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
        expl = shap.TreeExplainer(est)
        shap_values = expl.shap_values(Xp)
        # SHAP summary plot
        plt.figure(figsize=(8, 5))
        shap.summary_plot(shap_values, Xp, show=False)
        plt.tight_layout()
        plt.savefig(out / "shap_summary.png", dpi=150)
        plt.close()
    except Exception as e:
        with open(out / "shap_error.txt", "w", encoding="utf-8") as f:
            f.write(str(e))


def partial_dependence_plots(model_path: str, X_sample: pd.DataFrame, features: List[str], outdir: str):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    model = joblib.load(model_path)
    import matplotlib.pyplot as plt  # type: ignore
    fig, ax = plt.subplots(1, len(features), figsize=(4 * len(features), 3))
    if len(features) == 1:
        ax = [ax]
    try:
        # Cast selected numeric features to float to avoid FutureWarning
        Xp = X_sample.copy()
        for f in features:
            if f in Xp.columns:
                try:
                    Xp[f] = Xp[f].astype(float)
                except Exception:
                    pass
        PartialDependenceDisplay.from_estimator(model, Xp, features=features, ax=ax)
        plt.tight_layout()
        plt.savefig(out / "pdp.png", dpi=150)
    finally:
        plt.close()
