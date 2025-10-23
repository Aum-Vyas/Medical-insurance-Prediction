from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import plotly.express as px

# Note: scenario helpers are not required for the dashboard; imports removed


ART_DEFAULT = "artifacts"


@st.cache_data(show_spinner=False)
def load_table(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        try:
            if path.suffix == ".csv":
                return pd.read_csv(path)
            if path.suffix == ".json":
                return pd.DataFrame(json.load(open(path, "r", encoding="utf-8")))
        except Exception:
            return None
    return None


@st.cache_resource(show_spinner=False)
def load_pipe(path: Path):
    if path.exists():
        return joblib.load(path)
    return None


def section_header(title: str, subtitle: Optional[str] = None):
    st.markdown(f"<h2 style='margin-bottom:0.25rem'>{title}</h2>", unsafe_allow_html=True)
    if subtitle:
        st.caption(subtitle)


def overview(art: Path):
    section_header("Overview", "Model metrics and artifacts summary")
    metrics_csv = art / "model_metrics.csv"
    summary_json = art / "summary.json"
    metrics = load_table(metrics_csv)
    if metrics is not None and not metrics.empty:
        st.dataframe(metrics, use_container_width=True, hide_index=True)
        fig = px.bar(metrics.sort_values("rmse"), x="model", y=["rmse", "mae"], barmode="group", title="RMSE/MAE by model")
        st.plotly_chart(fig, use_container_width=True)
    if summary_json.exists():
        info = json.load(open(summary_json, "r", encoding="utf-8"))
        st.code(json.dumps(info, indent=2))


def explainability(art: Path):
    section_header("Explainability", "SHAP summary and PDPs")
    shap_img = art / "explain" / "shap_summary.png"
    pdp_img = art / "explain" / "pdp.png"
    cols = st.columns(2)
    with cols[0]:
        if shap_img.exists():
            st.image(str(shap_img), caption="SHAP Summary", use_column_width=True)
        else:
            st.info("SHAP plot not available (requires a tree-based best model and shap installed)")
    with cols[1]:
        if pdp_img.exists():
            st.image(str(pdp_img), caption="Partial Dependence (age, bmi, children)", use_column_width=True)
        else:
            st.info("PDP plot not generated yet")


def clustering(art: Path):
    section_header("Clustering", "Cluster sizes and average charges/premiums")
    cluster_summary = load_table(art / "cluster_summary.csv")
    if cluster_summary is None or cluster_summary.empty:
        st.info("Run the pipeline to generate clustering outputs.")
        return
    st.dataframe(cluster_summary, use_container_width=True, hide_index=True)
    if set(["cluster", "count"]).issubset(cluster_summary.columns):
        fig = px.bar(cluster_summary, x="cluster", y="count", title="Cluster Sizes", color="cluster")
        st.plotly_chart(fig, use_container_width=True)
    if set(["cluster", "avg_charges"]).issubset(cluster_summary.columns):
        fig2 = px.bar(cluster_summary, x="cluster", y="avg_charges", title="Average Charges by Cluster", color="cluster")
        st.plotly_chart(fig2, use_container_width=True)
    if "avg_premium" in cluster_summary.columns:
        fig3 = px.bar(cluster_summary, x="cluster", y="avg_premium", title="Average Premium by Cluster", color="cluster")
        st.plotly_chart(fig3, use_container_width=True)


def pricing_explorer(art: Path):
    section_header("Pricing Explorer", "Predicted cost + premium per record")
    pricing = load_table(art / "pricing.csv")
    if pricing is None or pricing.empty:
        st.info("Pricing file not found. Run the pipeline first.")
        return
    # Filters
    cols = st.columns(4)
    with cols[0]:
        region = st.selectbox("Region", options=["All"] + sorted(pricing["region"].dropna().unique().tolist()))
    with cols[1]:
        sex = st.selectbox("Sex", options=["All"] + sorted(pricing["sex"].dropna().unique().tolist()))
    with cols[2]:
        smoker = st.selectbox("Smoker", options=["All"] + sorted(pricing["smoker"].dropna().unique().tolist()))
    with cols[3]:
        cluster = st.selectbox("Cluster", options=["All"] + sorted(pricing["cluster"].dropna().unique().tolist())) if "cluster" in pricing.columns else "All"

    df = pricing.copy()
    if region != "All":
        df = df[df["region"] == region]
    if sex != "All":
        df = df[df["sex"] == sex]
    if smoker != "All":
        df = df[df["smoker"] == smoker]
    if cluster != "All" and "cluster" in df.columns:
        df = df[df["cluster"] == cluster]

    st.dataframe(df.head(500), use_container_width=True)
    if set(["predicted_cost", "premium"]).issubset(df.columns):
        fig = px.histogram(df, x="predicted_cost", nbins=30, title="Predicted Cost Distribution")
        st.plotly_chart(fig, use_container_width=True)
        fig2 = px.histogram(df, x="premium", nbins=30, title="Premium Distribution", color_discrete_sequence=["#2b8a3e"])
        st.plotly_chart(fig2, use_container_width=True)


def scenario_simulator(art: Path):
    section_header("Scenario Simulator", "What-if analysis per customer")
    pipe = load_pipe(art / "model_best.joblib")
    if pipe is None:
        st.info("Best model not found. Run the pipeline first.")
        return

    st.caption("Set baseline features then tweak scenario controls")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", min_value=18, max_value=100, value=45)
        children = st.number_input("Children", min_value=0, max_value=10, value=2)
    with c2:
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=28.0, step=0.1)
        region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])
    with c3:
        sex = st.selectbox("Sex", ["male", "female"]) 
        smoker = st.selectbox("Smoker", ["no", "yes"]) 

    margin = st.slider("Margin (+%)", min_value=0, max_value=50, value=20, step=5)
    base_df = pd.DataFrame([{"age": age, "sex": sex, "bmi": bmi, "children": children, "smoker": smoker, "region": region}])

    base_pred = float(pipe.predict(base_df)[0])
    base_premium = base_pred * (1 + margin / 100.0)

    st.metric("Baseline Predicted Cost", f"${base_pred:,.0f}")
    st.metric("Baseline Premium", f"${base_premium:,.0f}")

    st.subheader("Adjustments")
    dcol1, dcol2, dcol3 = st.columns(3)
    with dcol1:
        bmi_delta = st.slider("BMI change (%)", -30, 30, 0, step=5)
    with dcol2:
        smoker_flip = st.checkbox("Set Non-Smoker", value=(smoker == "no"))
    with dcol3:
        add_child = st.checkbox("Children +1", value=False)

    mod = base_df.iloc[0].copy()
    if bmi_delta != 0:
        mod["bmi"] = float(mod["bmi"]) * (1 + bmi_delta / 100.0)
    if smoker_flip:
        mod["smoker"] = "no"
    if add_child:
        mod["children"] = int(mod["children"]) + 1
    mod_df = pd.DataFrame([mod])
    mod_pred = float(pipe.predict(mod_df)[0])
    mod_premium = mod_pred * (1 + margin / 100.0)

    cols = st.columns(2)
    cols[0].metric("Scenario Predicted Cost", f"${mod_pred:,.0f}", delta=f"{mod_pred - base_pred:,.0f}")
    cols[1].metric("Scenario Premium", f"${mod_premium:,.0f}", delta=f"{mod_premium - base_premium:,.0f}")


def fairness_section(art: Path):
    section_header("Fairness", "Residuals/MAE by demographic groups")
    sex_df = load_table(art / "fairness_sex.csv")
    reg_df = load_table(art / "fairness_region.csv")
    c1, c2 = st.columns(2)
    with c1:
        if sex_df is not None and not sex_df.empty:
            st.dataframe(sex_df, use_container_width=True)
            if set(["sex", "mae"]).issubset(sex_df.columns):
                fig = px.bar(sex_df, x="sex", y="mae", title="MAE by sex")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Fairness by sex not available")
    with c2:
        if reg_df is not None and not reg_df.empty:
            st.dataframe(reg_df, use_container_width=True)
            if set(["region", "mae"]).issubset(reg_df.columns):
                fig = px.bar(reg_df, x="region", y="mae", title="MAE by region")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Fairness by region not available")


def main():
    st.set_page_config(page_title="Insurance Pricing Dashboard", page_icon="💹", layout="wide")
    sns.set_theme(style="whitegrid")

    st.markdown(
        """
        <style>
        .block-container {padding-top: 2rem;}
        h1, h2, h3 { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Insurance Cost Prediction & Pricing")
    st.caption("Model metrics • Explainability • Clustering • Pricing • Scenarios • Fairness")

    with st.sidebar:
        st.header("Settings")
        artifacts_dir = st.text_input("Artifacts directory", ART_DEFAULT)
        art = Path(artifacts_dir)
        if not art.exists():
            st.warning("Artifacts directory not found. Run the pipeline to generate outputs.")
        show_sections = st.multiselect(
            "Sections",
            ["Overview", "Explainability", "Clustering", "Pricing", "Scenario Simulator", "Fairness"],
            default=["Overview", "Explainability", "Clustering", "Pricing", "Scenario Simulator", "Fairness"],
        )

    art = Path(artifacts_dir)
    if "Overview" in show_sections:
        overview(art)
    if "Explainability" in show_sections:
        explainability(art)
    if "Clustering" in show_sections:
        clustering(art)
    if "Pricing" in show_sections:
        pricing_explorer(art)
    if "Scenario Simulator" in show_sections:
        scenario_simulator(art)
    if "Fairness" in show_sections:
        fairness_section(art)


if __name__ == "__main__":
    main()
