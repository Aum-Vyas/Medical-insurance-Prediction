from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


DEFAULT_REQUIRED = ["age", "sex", "bmi", "children", "smoker", "region"]


_ALIASES: Dict[str, str] = {
    "bmi": "bmi",
    "bmi_": "bmi",
    "b.m.i": "bmi",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=mapping)
    # alias remap
    for col in list(df.columns):
        key = col.replace(" ", "").replace("-", "").replace("_", "")
        if col not in _ALIASES and key in _ALIASES:
            df = df.rename(columns={col: _ALIASES[key]})
    return df


def load_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")
    df = pd.read_csv(p)
    if df.empty:
        raise ValueError("CSV is empty")
    df = _normalize_columns(df)
    return df


def split_features_target(
    df: pd.DataFrame,
    target: str = "charges",
    required: Tuple[str, ...] = tuple(DEFAULT_REQUIRED),
) -> Tuple[pd.DataFrame, pd.Series]:
    t = target.strip().lower()
    if t not in df.columns:
        raise KeyError(f"Target column '{target}' not found. Available: {list(df.columns)}")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    X = df[list(required)].copy()
    y = df[t].copy()
    return X, y
