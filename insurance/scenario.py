from __future__ import annotations

from typing import Dict, List

import joblib
import numpy as np
import pandas as pd


def apply_scenarios(record: pd.Series) -> Dict[str, pd.Series]:
    base = record.copy()
    out = {"baseline": base}
    # BMI -10%
    s1 = base.copy(); s1["bmi"] = max(10.0, float(base["bmi"]) * 0.9); out["bmi_-10pct"] = s1
    # Smoking cessation
    s2 = base.copy(); s2["smoker"] = "no"; out["smoking_cessation"] = s2
    # Children +1 (capped at +3)
    s3 = base.copy(); s3["children"] = int(base["children"]) + 1; out["children_plus1"] = s3
    # Age +1 year
    s4 = base.copy(); s4["age"] = int(base["age"]) + 1; out["age_plus1"] = s4
    return out


def evaluate_scenarios(pipe_path: str, df: pd.DataFrame, margin: float = 0.0) -> pd.DataFrame:
    pipe = joblib.load(pipe_path)
    rows: List[Dict] = []
    for idx, row in df.iterrows():
        scenarios = apply_scenarios(row)
        preds = {}
        for name, rec in scenarios.items():
            p = float(pipe.predict(pd.DataFrame([rec]))[0])
            preds[name] = p
        base = preds["baseline"]
        for name, val in preds.items():
            rows.append({
                "row": idx,
                "scenario": name,
                "pred_cost": val,
                "delta": val - base,
                "premium": val * (1.0 + margin),
            })
    return pd.DataFrame(rows)

