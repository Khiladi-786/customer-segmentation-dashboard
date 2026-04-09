"""
Page 03 — RFM Analysis
RFM score distributions · Segment heatmap · Radar charts · Top customers
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from utils.styles import inject_styles, page_header, plotly_dark_layout, CLUSTER_COLORS
from core.model_cache import get_models

st.set_page_config(
    page_title="RFM Analysis | SegmentAI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_styles()

models = get_models()
df     = models["df"]

page_header(
    "RFM Analysis",
    "Recency · Frequency · Monetary deep-dive — understand what drives each customer segment.",
    "📊",
)

# ── Check RFM Columns ─────────────────────────────────────────────────────────
rfm_cols = ["R_Score", "F_Score", "M_Score", "RFM_Score", "Monetary", "Frequency", "Recency"]
available = [c for c in rfm_cols if c in df.columns]
if not available:
    st.error("RFM features not found — check feature engineering pipeline.")
    st.stop()

# ── Segment Sizes ─────────────────────────────────────────────────────────────
if "RFM_Segment" in df.columns:
    st.markdown("### 🏷️ RFM Segment Distribution")
    seg_counts = df["RFM_Segment"].value_counts()
    fig_seg = px.bar(
        x=seg_counts.index, y=seg_counts.values,
        labels={"x": "RFM Segment", "y": "Customer Count"},
        color=seg_counts.index,
        color_discrete_sequence=CLUSTER_COLORS,
        text=seg_counts.values,
    )
    fig_seg = plotly_dark_layout(fig_seg, height=300)
    fig_seg.update_traces(textposition="outside", textfont=dict(color="white"))
    st.plotly_chart(fig_seg, use_container_width=True)

st.markdown("---")

# ── R / F / M Distributions ───────────────────────────────────────────────────
st.markdown("### 📈 Distribution of R, F, M Scores per Cluster")

if "Cluster" in df.columns:
    tab_r, tab_f, tab_m = st.tabs(["Recency Score", "Frequency Score", "Monetary Score"])

    for tab, col, label in zip(
        [tab_r, tab_f, tab_m],
        ["R_Score", "F_Score", "M_Score"],
        ["Recency (R)", "Frequency (F)", "Monetary (M)"],
    ):
        with tab:
            if col in df.columns:
                fig_box = px.box(
                    df.dropna(subset=[col]),
                    x="Cluster", y=col,
                    color=df["Cluster"].astype(str).loc[df[col].notna()],
                    color_discrete_sequence=CLUSTER_COLORS,
                    labels={"x": "Cluster", "y": label},
                    notched=True,
                )
                fig_box = plotly_dark_layout(fig_box, f"{label} Score Distribution by Cluster", height=380)
                st.plotly_chart(fig_box, use_container_width=True)
            else:
                st.info(f"{col} not available.")

st.markdown("---")

# ── RFM Heatmap ───────────────────────────────────────────────────────────────
st.markdown("### 🌡️ RFM Cluster Heatmap")

if "Cluster" in df.columns:
    heat_cols = [c for c in ["R_Score", "F_Score", "M_Score", "RFM_Score",
                               "Monetary", "Frequency", "Recency"] if c in df.columns]
    heat_df = df.groupby("Cluster")[heat_cols].mean().round(2)

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    heat_norm = pd.DataFrame(
        scaler.fit_transform(heat_df),
        index=heat_df.index,
        columns=heat_df.columns,
    )

    fig_heat = go.Figure(data=go.Heatmap(
        z=heat_norm.values,
        x=heat_norm.columns.tolist(),
        y=[f"Cluster {c}" for c in heat_norm.index],
        colorscale="Viridis",
        hoverongaps=False,
        text=heat_df.values.round(1),
        texttemplate="%{text}",
        textfont=dict(color="white", size=11),
    ))
    fig_heat = plotly_dark_layout(fig_heat, "Normalized RFM Features per Cluster", height=350)
    st.plotly_chart(fig_heat, use_container_width=True)

st.markdown("---")

# ── Radar Charts ──────────────────────────────────────────────────────────────
st.markdown("### 🕸️ RFM Radar Chart — Cluster Profiles")

if "Cluster" in df.columns:
    radar_cols = [c for c in ["R_Score", "F_Score", "M_Score"] if c in df.columns]
    if radar_cols:
        radar_df = df.groupby("Cluster")[radar_cols].mean()

        from sklearn.preprocessing import MinMaxScaler
        sclr = MinMaxScaler()
        radar_norm = pd.DataFrame(
            sclr.fit_transform(radar_df),
            index=radar_df.index,
            columns=radar_df.columns,
        )

        def hex_to_rgba(hex_color: str, alpha: float = 0.2) -> str:
            """Convert #RRGGBB hex to rgba(r,g,b,a) for Plotly compatibility."""
            h = hex_color.lstrip("#")
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return f"rgba({r},{g},{b},{alpha})"

        fig_radar = go.Figure()
        for i, (cluster_id, row) in enumerate(radar_norm.iterrows()):
            values = row.tolist() + [row.iloc[0]]  # Close the loop
            color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=radar_cols + [radar_cols[0]],
                fill="toself",
                name=f"Cluster {cluster_id}",
                line=dict(color=color, width=2),
                fillcolor=hex_to_rgba(color, 0.2),
            ))
        fig_radar = plotly_dark_layout(fig_radar, height=450)
        fig_radar.update_layout(polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,0.1)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        ))
        st.plotly_chart(fig_radar, use_container_width=True)

st.markdown("---")

# ── Top Customers by RFM ──────────────────────────────────────────────────────
st.markdown("### 🏆 Top 20 Customers by RFM Score")

if "RFM_Score" in df.columns:
    top_cols = [c for c in ["RFM_Score", "Monetary", "Frequency", "Recency",
                              "Income", "Cluster", "RFM_Segment", "CLV_Score"] if c in df.columns]
    top20 = df.nlargest(20, "RFM_Score")[top_cols].reset_index(drop=True)
    top20.index += 1
    st.dataframe(top20.style.background_gradient(subset=["RFM_Score"], cmap="Purples"),
                 use_container_width=True)
