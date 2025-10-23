from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from .data import load_csv, split_features_target
from .models import train_models
from .cluster import fit_kmeans_on_preprocessed, summarize_charges_by_cluster
from .explain import shap_summary_for_tree, partial_dependence_plots
from .scenario import evaluate_scenarios
from .fairness import residuals_by_group


def main():
    p = argparse.ArgumentParser(description="End-to-end insurance analytics pipeline")
    p.add_argument("--data", required=True)
    p.add_argument("--outdir", default="artifacts")
    p.add_argument("--margin", type=float, default=0.2, help="Pricing margin (e.g., 0.2 for +20%)")
    p.add_argument("--clusters", type=int, default=4)
    args = p.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    # Train and compare models
    results = train_models(args.data, outdir=str(out))
    best_model_path = out / "model_best.joblib"

    # Data splits for downstream analyses
    df = load_csv(args.data)
    X, y = split_features_target(df)

    # Explainability
    shap_dir = out / "explain"
    shap_dir.mkdir(exist_ok=True)
    # SHAP for best model if tree-based
    try:
        shap_summary_for_tree(str(best_model_path), X.sample(min(1000, len(X)), random_state=42), str(shap_dir))
    except Exception:
        pass
    # PDP for key features
    partial_dependence_plots(str(best_model_path), X, ["age", "bmi", "children"], str(shap_dir))

    # Clustering on preprocessed features
    clus_info = fit_kmeans_on_preprocessed(str(best_model_path), X, n_clusters=args.clusters, outdir=str(out))
    km = joblib.load(out / "kmeans.joblib")
    pipe = joblib.load(best_model_path)
    # Predictions used for pricing and per-cluster premium summary
    preds = pipe.predict(X)
    Xp = pipe.named_steps["pre"].transform(X)
    labels = km.predict(Xp)
    pd.DataFrame({"cluster": labels}).to_csv(out / "cluster_labels.csv", index=False)
    cluster_summary = summarize_charges_by_cluster(labels, y)
    # Add premium summary per cluster
    tmp_pricing = pd.DataFrame({"cluster": labels, "premium": preds * (1.0 + args.margin)})
    prem_summary = tmp_pricing.groupby("cluster").agg(avg_premium=("premium", "mean")).reset_index()
    cluster_summary = cluster_summary.merge(prem_summary, on="cluster", how="left")
    cluster_summary.to_csv(out / "cluster_summary.csv", index=False)

    # Pricing: prediction + margin
    pricing_df = X.copy()
    pricing_df["predicted_cost"] = preds
    pricing_df["premium"] = preds * (1.0 + args.margin)
    pricing_df["cluster"] = labels
    pricing_df.to_csv(out / "pricing.csv", index=False)

    # Scenarios on a sample
    scen = evaluate_scenarios(str(best_model_path), X.sample(min(50, len(X)), random_state=42), margin=args.margin)
    scen.to_csv(out / "scenario_results.csv", index=False)

    # Fairness / bias
    fb_sex = residuals_by_group(str(best_model_path), df, group="sex")
    fb_region = residuals_by_group(str(best_model_path), df, group="region")
    fb_sex.to_csv(out / "fairness_sex.csv", index=False)
    fb_region.to_csv(out / "fairness_region.csv", index=False)
    fairness_report = {
        "sex": fb_sex.to_dict(orient="records"),
        "region": fb_region.to_dict(orient="records"),
    }
    with open(out / "fairness_report.json", "w", encoding="utf-8") as f:
        json.dump(fairness_report, f, indent=2)

    with open(out / "run_complete.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_model": results["best_model"],
            "clusters": clus_info,
            "margin": args.margin,
        }, f, indent=2)

    print("Pipeline complete. Outputs in:", str(out.resolve()))


if __name__ == "__main__":
    main()
