"""
Page 04 — CLV Prediction
CLV distribution · Profitability ranking · CLV × Cluster matrix · Feature importance
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from utils.styles import inject_styles, page_header, metric_card, plotly_dark_layout, CLUSTER_COLORS
from core.model_cache import get_models

st.set_page_config(
    page_title="CLV Prediction | SegmentAI",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_styles()

models    = get_models()
df        = models["df"]
clv_model = models["clv_model"]

page_header(
    "Customer Lifetime Value Prediction",
    "GradientBoosting model ranks every customer by predicted future value.",
    "💰",
)

# ── Guard ─────────────────────────────────────────────────────────────────────
if "CLV_Score" not in df.columns:
    st.error("CLV scores not found — check model_cache.")
    st.stop()

# ── KPI Row ───────────────────────────────────────────────────────────────────
avg_clv   = df["CLV_Score"].mean()
max_clv   = df["CLV_Score"].max()
prem_pct  = (df["CLV_Tier"] == "Premium").mean() * 100 if "CLV_Tier" in df.columns else 0
cv_r2     = clv_model.cv_score if hasattr(clv_model, "cv_score") else 0

cols = st.columns(4)
for col, (lbl, val, icon) in zip(cols, [
    ("Avg CLV Score",    f"{avg_clv:.0f}/1000",   "📊"),
    ("Max CLV Score",    f"{max_clv:.0f}",         "🚀"),
    ("Premium Customers",f"{prem_pct:.1f}%",       "👑"),
    ("Model CV R²",      f"{cv_r2:.3f}",           "🎯"),
]):
    with col:
        st.markdown(metric_card(lbl, val, icon), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── CLV Distribution ──────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📈 CLV Score Distribution")
    fig_hist = px.histogram(
        df, x="CLV_Score", nbins=50,
        color_discrete_sequence=["#7c3aed"],
        labels={"CLV_Score": "CLV Score (0–1000)"},
    )
    fig_hist = plotly_dark_layout(fig_hist, height=330)
    fig_hist.update_traces(opacity=0.8)
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    st.markdown("### 🎂 CLV Tier Breakdown")
    if "CLV_Tier" in df.columns:
        tier_counts = df["CLV_Tier"].value_counts()
        tier_colors = {"Low": "#ef4444", "Medium": "#f59e0b", "High": "#10b981", "Premium": "#7c3aed"}
        fig_tier = px.pie(
            values=tier_counts.values,
            names=tier_counts.index,
            color=tier_counts.index,
            color_discrete_map=tier_colors,
            hole=0.55,
        )
        fig_tier = plotly_dark_layout(fig_tier, height=330)
        fig_tier.update_traces(
            textposition="inside",
            textinfo="percent+label",
            insidetextfont=dict(color="white"),
        )
        st.plotly_chart(fig_tier, use_container_width=True)
    else:
        st.info("CLV Tier not available.")

st.markdown("---")

# ── CLV × Cluster Matrix ──────────────────────────────────────────────────────
st.markdown("### 🧩 CLV Score by Customer Segment")
if "Cluster" in df.columns:
    fig_violin = px.violin(
        df, x="Cluster", y="CLV_Score",
        color=df["Cluster"].astype(str),
        color_discrete_sequence=CLUSTER_COLORS,
        box=True, points="outliers",
        labels={"CLV_Score": "CLV Score", "Cluster": "Cluster"},
    )
    fig_violin = plotly_dark_layout(fig_violin, height=400)
    st.plotly_chart(fig_violin, use_container_width=True)

st.markdown("---")

# ── Profitability Ranking ────────────────────────────────────────────────────
st.markdown("### 🏆 Top 30 Most Valuable Customers")
rank_cols = [c for c in ["CLV_Rank", "CLV_Score", "CLV_Tier", "Monetary", "Frequency",
                          "Recency", "Income", "Cluster", "RFM_Segment"] if c in df.columns]
top30 = df.nsmallest(30, "CLV_Rank")[rank_cols].reset_index(drop=True)
top30.index += 1
st.dataframe(
    top30.style
    .background_gradient(subset=["CLV_Score"], cmap="Purples")
    .background_gradient(subset=["Monetary"] if "Monetary" in top30.columns else [], cmap="Greens"),
    use_container_width=True,
)

st.markdown("---")

# ── CLV Feature Importance ────────────────────────────────────────────────────
st.markdown("### 🔍 Which Features Drive CLV?")
fi_df = clv_model.feature_importance()
if not fi_df.empty:
    fig_fi = px.bar(
        fi_df.head(15), x="Importance", y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Purples",
    )
    fig_fi = plotly_dark_layout(fig_fi, "Top Feature Importances (GradientBoosting)", height=400)
    fig_fi.update_layout(coloraxis_showscale=False)
    fig_fi.update_traces(marker_line_width=0)
    st.plotly_chart(fig_fi, use_container_width=True)
else:
    st.info("Feature importance not available.")

# ── Revenue Potential ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 💵 Revenue Potential by Segment")
if "Cluster" in df.columns and "CLV_Score" in df.columns:
    revenue_df = df.groupby("Cluster").agg(
        Customers=("CLV_Score", "count"),
        AvgCLV=("CLV_Score", "mean"),
        TotalCLV=("CLV_Score", "sum"),
    ).round(1).reset_index()
    revenue_df["RevenueShare%"] = (revenue_df["TotalCLV"] / revenue_df["TotalCLV"].sum() * 100).round(1)

    fig_rev = px.bar(
        revenue_df, x="Cluster", y="TotalCLV",
        color=revenue_df["Cluster"].astype(str),
        color_discrete_sequence=CLUSTER_COLORS,
        text="RevenueShare%",
        labels={"TotalCLV": "Total CLV Score"},
    )
    fig_rev = plotly_dark_layout(fig_rev, "Total CLV Score by Segment (% Revenue Share)", height=350)
    fig_rev.update_traces(texttemplate="%{text}%", textposition="outside", textfont=dict(color="white"))
    st.plotly_chart(fig_rev, use_container_width=True)
