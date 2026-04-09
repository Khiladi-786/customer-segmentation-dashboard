"""
Page 08 — Anomaly Detection
Isolation Forest results · Risk distribution · 3D visualization · Anomaly table
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from utils.styles import inject_styles, page_header, metric_card, plotly_dark_layout, CLUSTER_COLORS
from core.model_cache import get_models

st.set_page_config(
    page_title="Anomaly Detection | SegmentAI",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_styles()

models   = get_models()
df       = models["df"]
detector = models["detector"]

page_header(
    "Anomaly Detection",
    "Isolation Forest identifies unusual customers whose behavior deviates significantly from their segment.",
    "⚠️",
)

# ── Guard ─────────────────────────────────────────────────────────────────────
if "IsAnomaly" not in df.columns:
    st.error("Anomaly detection not run — check model_cache.")
    st.stop()

# ── KPIs ──────────────────────────────────────────────────────────────────────
n_total   = len(df)
n_anom    = int(df["IsAnomaly"].sum())
n_normal  = n_total - n_anom
anom_pct  = n_anom / n_total * 100
avg_risk  = df[df["IsAnomaly"]]["AnomalyRisk"].mean() if n_anom else 0

cols = st.columns(4)
for col, (lbl, val, icon, delta) in zip(cols, [
    ("Total Customers",   f"{n_total:,}",         "👥", ""),
    ("Anomalies Found",   f"{n_anom}",             "⚠️",  f"{anom_pct:.1f}% of total"),
    ("Normal Customers",  f"{n_normal:,}",         "✅", f"{100-anom_pct:.1f}% of total"),
    ("Avg Anomaly Risk",  f"{avg_risk:.1f}/100",   "🔥", "higher = more unusual"),
]):
    with col:
        st.markdown(metric_card(lbl, val, icon, delta), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Risk Distribution ─────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 Anomaly Risk Score Distribution")
    fig_hist = px.histogram(
        df, x="AnomalyRisk", nbins=40,
        color=df["IsAnomaly"].map({True: "⚠️ Anomaly", False: "✅ Normal"}),
        color_discrete_map={"⚠️ Anomaly": "#ef4444", "✅ Normal": "#10b981"},
        barmode="overlay",
        labels={"AnomalyRisk": "Anomaly Risk Score (0–100)"},
    )
    fig_hist = plotly_dark_layout(fig_hist, height=320)
    fig_hist.update_traces(opacity=0.75)
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    st.markdown("### 🥧 Normal vs. Anomaly")
    fig_donut = px.pie(
        values=[n_normal, n_anom],
        names=["✅ Normal", "⚠️ Anomaly"],
        color_discrete_sequence=["#10b981", "#ef4444"],
        hole=0.6,
    )
    fig_donut = plotly_dark_layout(fig_donut, height=320)
    fig_donut.update_traces(
        textinfo="percent+label",
        insidetextfont=dict(color="white"),
    )
    st.plotly_chart(fig_donut, use_container_width=True)

st.markdown("---")

# ── 3D Scatter with Anomalies Highlighted ────────────────────────────────────
st.markdown("### 🔮 3D Cluster Space — Anomalies Highlighted")
if "PCA1" in df.columns:
    normal_df  = df[~df["IsAnomaly"]].sample(min(800, (~df["IsAnomaly"]).sum()))
    anomaly_df = df[df["IsAnomaly"]]

    fig3d = go.Figure()

    # Normal points per cluster
    for c in sorted(normal_df["Cluster"].unique()):
        cdf = normal_df[normal_df["Cluster"] == c]
        fig3d.add_trace(go.Scatter3d(
            x=cdf["PCA1"], y=cdf["PCA2"], z=cdf.get("PCA3", pd.Series([0]*len(cdf))),
            mode="markers",
            name=f"Cluster {c}",
            marker=dict(size=2.5, color=CLUSTER_COLORS[int(c) % len(CLUSTER_COLORS)], opacity=0.4),
        ))

    # Anomaly points
    if len(anomaly_df) > 0:
        fig3d.add_trace(go.Scatter3d(
            x=anomaly_df["PCA1"],
            y=anomaly_df["PCA2"],
            z=anomaly_df.get("PCA3", pd.Series([0]*len(anomaly_df))),
            mode="markers",
            name="⚠️ Anomaly",
            marker=dict(
                size=7,
                color="#ef4444",
                symbol="diamond",
                opacity=0.9,
                line=dict(color="white", width=1),
            ),
        ))

    fig3d = plotly_dark_layout(fig3d, height=500)
    fig3d.update_layout(
        scene=dict(
            bgcolor="rgba(0,0,0,0)",
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.05)"),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.05)"),
        )
    )
    st.plotly_chart(fig3d, use_container_width=True)

st.markdown("---")

# ── Anomaly Profile Comparison ────────────────────────────────────────────────
st.markdown("### 📈 Anomalous vs. Normal Customer Profiles")
compare_cols = [c for c in ["Income", "TotalSpend", "Recency", "Frequency",
                              "CLV_Score", "Age", "CampaignAcceptRate"] if c in df.columns]
if compare_cols:
    anom_means   = df[df["IsAnomaly"]][compare_cols].mean()
    normal_means = df[~df["IsAnomaly"]][compare_cols].mean()

    from sklearn.preprocessing import MinMaxScaler
    all_means = pd.DataFrame({"Anomaly": anom_means, "Normal": normal_means})
    scaler    = MinMaxScaler()
    normed    = pd.DataFrame(
        scaler.fit_transform(all_means.T).T,
        index=all_means.index, columns=all_means.columns
    )

    def hex_to_rgba(hex_color: str, alpha: float = 0.2) -> str:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    fig_radar = go.Figure()
    for col_name, color in [("Anomaly", "#ef4444"), ("Normal", "#10b981")]:
        vals = normed[col_name].tolist() + [normed[col_name].iloc[0]]
        cats = compare_cols + [compare_cols[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=cats, fill="toself", name=col_name,
            line=dict(color=color, width=2),
            fillcolor=hex_to_rgba(color, 0.2),
        ))
    fig_radar = plotly_dark_layout(fig_radar, height=380)
    fig_radar.update_layout(polar=dict(
        bgcolor="rgba(0,0,0,0)",
        radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,0.1)"),
        angularaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
    ))
    st.plotly_chart(fig_radar, use_container_width=True)

st.markdown("---")

# ── Anomaly Table ─────────────────────────────────────────────────────────────
st.markdown("### 🗃️ Detected Anomalies")
risk_threshold = st.slider("Filter by Anomaly Risk ≥", 0, 100, 50)
anom_display = df[df["IsAnomaly"] & (df["AnomalyRisk"] >= risk_threshold)].copy()
anom_cols = [c for c in ["AnomalyRisk", "Cluster", "Income", "TotalSpend",
                           "Recency", "CLV_Score", "RFM_Segment"] if c in anom_display.columns]
if len(anom_display) > 0:
    st.dataframe(
        anom_display[anom_cols].sort_values("AnomalyRisk", ascending=False).reset_index(drop=True)
        .style.background_gradient(subset=["AnomalyRisk"], cmap="Reds"),
        use_container_width=True,
    )
    st.caption(f"Showing {len(anom_display)} anomalies with risk ≥ {risk_threshold}")
else:
    st.info(f"No anomalies with risk ≥ {risk_threshold}. Lower the threshold.")
