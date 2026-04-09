"""
Page 01 — Platform Overview
KPI cards · 3D PCA cluster scatter · Segment distribution · Cluster profiles
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from utils.styles import inject_styles, page_header, metric_card, plotly_dark_layout, CLUSTER_COLORS
from core.model_cache import get_models

st.set_page_config(
    page_title="Overview | SegmentAI",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_styles()

# ── Load models ───────────────────────────────────────────────────────────────
models = get_models()
df     = models["df"]
engine = models["engine"]

# ── Header ────────────────────────────────────────────────────────────────────
page_header(
    "Platform Overview",
    "Real-time AI-powered customer intelligence — all segments at a glance.",
    "🏠",
)

# ── KPI Metrics ───────────────────────────────────────────────────────────────
n_customers  = len(df)
n_segments   = df["Cluster"].nunique()
best_algo    = models["best_algo"]
avg_clv      = df["CLV_Score"].mean() if "CLV_Score" in df.columns else 0
n_anomalies  = int(df["IsAnomaly"].sum()) if "IsAnomaly" in df.columns else 0
sil_score    = models["best_metrics"].get("silhouette", 0)

cols = st.columns(5)
cards = [
    ("Total Customers",    f"{n_customers:,}",      "👥", ""),
    ("Segments Found",     str(n_segments),          "🎯", f"via {best_algo}"),
    ("Best Algorithm",     best_algo,                "🤖", f"Silhouette: {sil_score:.3f}"),
    ("Avg CLV Score",      f"{avg_clv:.0f}",         "💰", "out of 1,000"),
    ("Anomalies Detected", str(n_anomalies),         "⚠️",  f"{n_anomalies/n_customers*100:.1f}% of total"),
]
for col, (label, value, icon, delta) in zip(cols, cards):
    with col:
        st.markdown(metric_card(label, value, icon, delta), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── 3D Scatter + Pie ──────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("### 🔮 3D Customer Cluster Space")
    plot_df = df.copy()
    if "PCA1" not in plot_df.columns or "PCA2" not in plot_df.columns or "PCA3" not in plot_df.columns:
        st.warning("PCA coordinates not available.")
    else:
        hover_cols = [c for c in ["Income", "TotalSpend", "CLV_Score", "RFM_Segment"] if c in plot_df.columns]
        fig3d = px.scatter_3d(
            plot_df.sample(min(1500, len(plot_df))),
            x="PCA1", y="PCA2", z="PCA3",
            color=plot_df["Cluster"].astype(str).iloc[:min(1500, len(plot_df))],
            color_discrete_sequence=CLUSTER_COLORS,
            hover_data=hover_cols,
            opacity=0.75,
            size_max=6,
        )
        fig3d = plotly_dark_layout(fig3d, height=450)
        fig3d.update_traces(marker=dict(size=3))
        fig3d.update_layout(
            scene=dict(
                bgcolor="rgba(0,0,0,0)",
                xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.05)"),
                zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.05)"),
            )
        )
        st.plotly_chart(fig3d, use_container_width=True)

with col2:
    st.markdown("### 📊 Segment Distribution")
    cluster_counts = df["Cluster"].value_counts().sort_index()
    fig_pie = px.pie(
        values=cluster_counts.values,
        names=[f"Cluster {c}" for c in cluster_counts.index],
        color_discrete_sequence=CLUSTER_COLORS,
        hole=0.5,
    )
    fig_pie = plotly_dark_layout(fig_pie, height=300)
    fig_pie.update_traces(
        textposition="inside",
        textinfo="percent+label",
        insidetextfont=dict(color="white", size=11),
        marker=dict(line=dict(color="#0a0a0f", width=2)),
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # Mini metrics per cluster
    st.markdown("**Cluster Sizes**")
    for c_id, count in cluster_counts.items():
        pct = count / n_customers * 100
        color = CLUSTER_COLORS[int(c_id) % len(CLUSTER_COLORS)]
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:10px;margin:4px 0">'
            f'<div style="width:12px;height:12px;background:{color};border-radius:50%"></div>'
            f'<span style="color:#f1f5f9;font-weight:600">Cluster {c_id}</span>'
            f'<span style="color:#94a3b8;margin-left:auto">{count:,} ({pct:.1f}%)</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ── Cluster Profiles Table ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🗂️ Cluster Profile Comparison")

key_cols = [c for c in ["Cluster", "Income", "TotalSpend", "Recency", "Frequency",
                          "CLV_Score", "Age", "CampaignAcceptRate", "RFM_Segment"]
            if c in df.columns]
if "Cluster" in df.columns and len(key_cols) > 1:
    profile_df = df[key_cols].copy()
    numeric_cols = profile_df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "Cluster"]
    profile_agg  = profile_df.groupby("Cluster")[numeric_cols].mean().round(1)

    if "RFM_Segment" in profile_df.columns:
        rfm_mode = profile_df.groupby("Cluster")["RFM_Segment"].agg(
            lambda x: x.mode()[0] if len(x) else "Unknown"
        )
        profile_agg["RFM_Segment"] = rfm_mode

    st.dataframe(
        profile_agg.style.background_gradient(subset=numeric_cols, cmap="magma"),
        use_container_width=True,
    )

# ── Algorithm Performance Banner ──────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🏆 AutoML Best Result")
m = models["best_metrics"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Algorithm", best_algo)
c2.metric("Silhouette ↑", f"{m.get('silhouette', 0):.4f}")
c3.metric("Davies-Bouldin ↓", f"{m.get('davies_bouldin', 0):.4f}")
c4.metric("Composite Score", f"{m.get('composite', 0):.4f}")
