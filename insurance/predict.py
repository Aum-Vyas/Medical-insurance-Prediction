from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: c.strip().lower() for c in df.columns})


def _ensure_columns(df: pd.DataFrame, required: List[str]) -> pd.DataFrame:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    return df[required].copy()


def _load_input(csv: Optional[str], json_path: Optional[str]) -> pd.DataFrame:
    if csv:
        p = Path(csv)
        if not p.exists():
            cwd = Path.cwd()
            raise FileNotFoundError(f"CSV not found: {p}. Current working directory: {cwd}. Provide a valid --csv path or run from the folder containing the file.")
        df = pd.read_csv(p)
    elif json_path:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data)
    else:
        raise ValueError("Provide --csv or --json input")
    return _normalize_df(df)


def main():
    p = argparse.ArgumentParser(description="Predict insurance charges")
    p.add_argument("--model", required=True, help="Path to trained joblib model")
    p.add_argument("--csv", help="CSV with records to score")
    p.add_argument("--json", help="JSON file (single or list of records)")
    p.add_argument("--out", help="Save predictions to CSV path")
    p.add_argument("--price-margin", type=float, default=0.0, help="Optional margin like 0.15 for +15%")
    args = p.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    pipe = joblib.load(model_path)

    # Infer required columns from pipeline preprocessor if possible
    try:
        pre = pipe.named_steps["pre"]
        feature_columns = []
        for _, _, cols in pre.transformers:
            feature_columns.extend(list(cols))
    except Exception:
        # Fallback to canonical order
        feature_columns = ["age", "bmi", "children", "sex", "smoker", "region"]

    df = _load_input(args.csv, args.json)
    df = _ensure_columns(df, feature_columns)
    preds = pipe.predict(df)
    if args.price_margin and args.price_margin != 0:
        priced = preds * (1.0 + args.price_margin)
    else:
        priced = preds

    out_df = df.copy()
    out_df["prediction"] = preds
    out_df["price"] = priced

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.out, index=False)
        print(json.dumps({"saved": args.out, "rows": len(out_df)}, indent=2))
    else:
        print(out_df.to_csv(index=False))


if __name__ == "__main__":
    main()
