from __future__ import annotations

from typing import List, Tuple

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    numeric: List[str] = ["age", "bmi", "children"],
    categorical: List[str] = ["sex", "smoker", "region"],
) -> ColumnTransformer:
    # Dense outputs for broad estimator compatibility
    num = Pipeline(steps=[("scaler", StandardScaler(with_mean=True))])
    # Handle OneHotEncoder API change (sparse_output vs sparse)
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # sklearn >=1.2
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)  # sklearn <1.2
    cat = Pipeline(steps=[("ohe", ohe)])
    pre = ColumnTransformer(
        transformers=[
            ("num", num, numeric),
            ("cat", cat, categorical),
        ]
    )
    return pre


def build_pipeline(estimator) -> Pipeline:
    pre = build_preprocessor()
    pipe = Pipeline(steps=[("pre", pre), ("model", estimator)])
    return pipe
