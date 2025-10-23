from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from .scenario import apply_scenarios


class Features(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str
    margin: Optional[float] = 0.2


def create_app(model_path: str = "artifacts/model_best.joblib", kmeans_path: str = "artifacts/kmeans.joblib") -> FastAPI:
    app = FastAPI(title="Insurance Pricing API")
    model_file = Path(model_path)
    kmeans_file = Path(kmeans_path)
    pipe = joblib.load(model_file) if model_file.exists() else None
    kmeans = joblib.load(kmeans_file) if kmeans_file.exists() else None

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": pipe is not None}

    @app.get("/")
    def root():
        # Friendly redirect to interactive docs
        return RedirectResponse(url="/docs")

    @app.post("/predict")
    def predict(feat: Features, include_scenarios: bool = False):
        if pipe is None:
            return {"error": "Model not loaded"}
        df = pd.DataFrame([{k: getattr(feat, k) for k in ["age", "sex", "bmi", "children", "smoker", "region"]}])
        pred = float(pipe.predict(df)[0])
        margin = feat.margin if feat.margin is not None else 0.2
        premium = pred * (1.0 + margin)

        cluster = None
        if kmeans is not None:
            Xp = pipe.named_steps["pre"].transform(df)
            cluster = int(kmeans.predict(Xp)[0])

        resp = {
            "predicted_cost": pred,
            "premium": premium,
            "margin": margin,
            "cluster": cluster,
        }
        if include_scenarios:
            sc = apply_scenarios(df.iloc[0])
            scen_rows = []
            for name, rec in sc.items():
                val = float(pipe.predict(pd.DataFrame([rec]))[0])
                scen_rows.append({
                    "scenario": name,
                    "pred_cost": val,
                    "premium": val * (1.0 + margin),
                    "delta": val - pred,
                })
            resp["scenarios"] = scen_rows
        return resp

    return app


app = create_app()
