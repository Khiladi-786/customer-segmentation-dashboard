"""
Page 02 — Algorithm Comparison
Side-by-side metrics · Elbow curve · 2D scatter per algorithm · Dendrogram
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
import numpy as np

from utils.styles import inject_styles, page_header, plotly_dark_layout, CLUSTER_COLORS
from core.model_cache import get_models
from utils.metrics import metrics_dataframe

st.set_page_config(
    page_title="Algorithms | SegmentAI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_styles()

models = get_models()
engine = models["engine"]
X      = models["X"]
df     = models["df"]

page_header(
    "Algorithm Comparison",
    "AutoML evaluated 4 algorithms across the full hyperparameter grid — here's how they compare.",
    "🤖",
)

# ── Metrics Table ─────────────────────────────────────────────────────────────
st.markdown("### 📊 Performance Metrics (All Algorithms)")

all_results = models["all_results"]
metrics_rows = []
for r in all_results:
    metrics_rows.append({
        "algorithm":         r["algorithm"],
        "silhouette":        r["metrics"]["silhouette"],
        "davies_bouldin":    r["metrics"]["davies_bouldin"],
        "calinski_harabasz": r["metrics"]["calinski_harabasz"],
        "composite":         r["metrics"]["composite"],
        "n_clusters":        r.get("n_clusters", 0),
    })
metrics_df = metrics_dataframe(metrics_rows)
metrics_df.insert(0, "🏆", ["⭐" if row["Algorithm"] == models["best_algo"] else "" for _, row in metrics_df.iterrows()])

st.dataframe(
    metrics_df.style
    .background_gradient(subset=["Composite Score ↑"], cmap="Purples")
    .background_gradient(subset=["Silhouette ↑"], cmap="Blues")
    .format({
        "Silhouette ↑":        "{:.4f}",
        "Davies-Bouldin ↓":    "{:.4f}",
        "Calinski-Harabasz ↑": "{:.1f}",
        "Composite Score ↑":   "{:.4f}",
    }),
    use_container_width=True,
)

st.markdown("> **Best algorithm** (⭐) selected by composite score:  "
            f"**{models['best_algo']}** with composite = "
            f"**{models['best_metrics'].get('composite', 0):.4f}**")

st.markdown("---")

# ── Metric Bars ───────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🎯 Silhouette Score (higher = better)")
    if metrics_rows:
        df_sil = pd.DataFrame({
            "Algorithm": [r["algorithm"] for r in metrics_rows],
            "Silhouette": [r["silhouette"] for r in metrics_rows],
        }).sort_values("Silhouette", ascending=True)
        fig_sil = px.bar(
            df_sil, x="Silhouette", y="Algorithm", orientation="h",
            color="Silhouette", color_continuous_scale="Purples",
        )
        fig_sil = plotly_dark_layout(fig_sil, height=300)
        fig_sil.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_sil, use_container_width=True)

with col2:
    st.markdown("### 📉 Davies-Bouldin Index (lower = better)")
    if metrics_rows:
        df_db = pd.DataFrame({
            "Algorithm": [r["algorithm"] for r in metrics_rows],
            "DaviesBouldin": [r["davies_bouldin"] for r in metrics_rows],
        }).sort_values("DaviesBouldin", ascending=False)
        fig_db = px.bar(
            df_db, x="DaviesBouldin", y="Algorithm", orientation="h",
            color="DaviesBouldin", color_continuous_scale="Reds",
        )
        fig_db = plotly_dark_layout(fig_db, height=300)
        fig_db.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_db, use_container_width=True)

st.markdown("---")

# ── Elbow Curve ───────────────────────────────────────────────────────────────
st.markdown("### 🦾 K-Means Elbow Curve")
elbow = models["elbow"]
if elbow:
    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(
        x=elbow["k"], y=elbow["inertia"],
        mode="lines+markers",
        line=dict(color="#7c3aed", width=3),
        marker=dict(color="#06b6d4", size=8, symbol="circle"),
        name="Inertia",
        fill="tozeroy",
        fillcolor="rgba(124, 58, 237, 0.1)",
    ))
    fig_elbow = plotly_dark_layout(fig_elbow, "Inertia vs. Number of Clusters (K)", height=350)
    fig_elbow.update_layout(xaxis_title="K", yaxis_title="Inertia")
    st.plotly_chart(fig_elbow, use_container_width=True)
    st.caption("The 'elbow' point indicates the optimal K — where adding more clusters gives diminishing returns.")

st.markdown("---")

# ── 2D Scatter per Algorithm ──────────────────────────────────────────────────
st.markdown("### 🗺️ 2D PCA Projection per Algorithm")

algo_tabs = st.tabs([r["algorithm"] for r in all_results])
for tab, result in zip(algo_tabs, all_results):
    with tab:
        labels = result["labels"]
        coords_2d = engine.project_2d(X)
        n = min(len(coords_2d), len(labels))
        scatter_df = pd.DataFrame({
            "PC1":     coords_2d[:n, 0],
            "PC2":     coords_2d[:n, 1],
            "Cluster": labels[:n].astype(str),
        })
        fig_scatter = px.scatter(
            scatter_df.sample(min(1000, n)),
            x="PC1", y="PC2",
            color="Cluster",
            color_discrete_sequence=CLUSTER_COLORS,
            opacity=0.7,
            title=f"{result['algorithm']} — {result.get('n_clusters', '?')} clusters | "
                  f"Silhouette={result['metrics']['silhouette']:.3f}",
        )
        fig_scatter = plotly_dark_layout(fig_scatter, height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)
        m = result["metrics"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Silhouette",        f"{m['silhouette']:.4f}")
        c2.metric("Davies-Bouldin",    f"{m['davies_bouldin']:.4f}")
        c3.metric("Composite",         f"{m['composite']:.4f}")
