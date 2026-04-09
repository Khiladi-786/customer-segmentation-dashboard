"""
Page 07 — What-If Simulator
Modify customer attributes with sliders → real-time cluster prediction and comparison.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from utils.styles import inject_styles, page_header, metric_card, plotly_dark_layout, CLUSTER_COLORS
from core.model_cache import get_models

st.set_page_config(
    page_title="What-If Simulator | SegmentAI",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_styles()

models   = get_models()
df       = models["df"]
engine   = models["engine"]
pipeline = models["pipeline"]
clv_model = models["clv_model"]

page_header(
    "What-If Simulator",
    "Adjust customer attributes and instantly see which segment they'd belong to — powered by the live AI model.",
    "🔮",
)

# ── Build column defaults ─────────────────────────────────────────────────────
def col_range(col, default):
    if col in df.columns:
        return float(df[col].min()), float(df[col].max()), float(df[col].median())
    return 0.0, 100.0, default

# ── Sidebar Controls ──────────────────────────────────────────────────────────
st.sidebar.markdown("## 🎛️ Customer Attributes")

r_min, r_max, r_def = col_range("Recency",  30.0)
income_min, income_max, income_def = col_range("Income", 50000.0)
mnt_wine_min, mnt_wine_max, mnt_wine_def = col_range("MntWines", 300.0)
mnt_meat_min, mnt_meat_max, mnt_meat_def = col_range("MntMeatProducts", 200.0)
freq_min, freq_max, freq_def = col_range("Frequency", 10.0)
age_min, age_max, age_def = col_range("Age", 40.0)

recency = st.sidebar.slider("📅 Recency (days)", int(r_min), int(r_max), int(r_def))
income  = st.sidebar.slider("💵 Annual Income ($)", int(income_min), int(income_max), int(income_def), step=1000)
mnt_wines  = st.sidebar.slider("🍷 Wine Spend ($)", int(mnt_wine_min), int(mnt_wine_max), int(mnt_wine_def))
mnt_meat   = st.sidebar.slider("🥩 Meat Spend ($)", int(mnt_meat_min), int(mnt_meat_max), int(mnt_meat_def))
num_web    = st.sidebar.slider("🌐 Web Purchases", 0, 20, 5)
num_store  = st.sidebar.slider("🏪 Store Purchases", 0, 20, 7)
num_catalog = st.sidebar.slider("📦 Catalog Purchases", 0, 15, 3)
kidhome    = st.sidebar.slider("👶 Children at Home", 0, 3, 0)
teenhome   = st.sidebar.slider("🧒 Teens at Home", 0, 3, 0)
year_birth = st.sidebar.slider("🎂 Birth Year", 1940, 2000, 1975)
num_web_visits = st.sidebar.slider("🖥️ Web Visits/Month", 0, 20, 5)
num_deals  = st.sidebar.slider("🏷️ Deal Purchases", 0, 15, 2)

# ── Build Customer Record ─────────────────────────────────────────────────────
customer_record = {
    "Year_Birth":          year_birth,
    "Education":           "Graduation",
    "Marital_Status":      "Single",
    "Income":              float(income),
    "Kidhome":             kidhome,
    "Teenhome":            teenhome,
    "Recency":             recency,
    "MntWines":            float(mnt_wines),
    "MntFruits":           50.0,
    "MntMeatProducts":     float(mnt_meat),
    "MntFishProducts":     50.0,
    "MntSweetProducts":    30.0,
    "MntGoldProds":        50.0,
    "NumDealsPurchases":   num_deals,
    "NumWebPurchases":     num_web,
    "NumCatalogPurchases": num_catalog,
    "NumStorePurchases":   num_store,
    "NumWebVisitsMonth":   num_web_visits,
    "AcceptedCmp1": 0, "AcceptedCmp2": 0, "AcceptedCmp3": 0,
    "AcceptedCmp4": 0, "AcceptedCmp5": 0,
    "Complain": 0, "Z_CostContact": 3, "Z_Revenue": 11, "Response": 0,
}

# ── Predict ───────────────────────────────────────────────────────────────────
try:
    X_single     = pipeline.preprocess_single(customer_record)
    pred_cluster = int(engine.predict(X_single)[0])
    cluster_color = CLUSTER_COLORS[pred_cluster % len(CLUSTER_COLORS)]

    # CLV score — use predict_single_clv for correct scaling
    clv_val = 0.0
    clv_tier = "Unknown"
    if clv_model and clv_model.is_fitted:
        clv_val, clv_tier = clv_model.predict_single_clv(customer_record)

    # RFM estimate
    total_spend = mnt_wines + mnt_meat + 50 + 50 + 30 + 50
    frequency   = num_web + num_catalog + num_store
    r_sc = 5 if recency < 30 else 3 if recency < 60 else 1
    f_sc = 5 if frequency >= 15 else 3 if frequency >= 8 else 1
    m_sc = 5 if total_spend >= 1000 else 3 if total_spend >= 400 else 1
    rfm_score = r_sc + f_sc + m_sc
    rfm_seg = ("Champions" if rfm_score >= 13 else "Loyal Customers" if rfm_score >= 10
               else "Potential Loyalists" if rfm_score >= 7 else "At Risk" if rfm_score >= 5
               else "Lost / Inactive")

    # ── Prediction Result ────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,{cluster_color}22,rgba(6,182,212,0.1));
                    border:2px solid {cluster_color}88;border-radius:20px;
                    padding:32px;text-align:center;margin-bottom:28px">
            <div style="font-size:3.5rem;margin-bottom:8px">🎯</div>
            <div style="font-size:1.1rem;color:#94a3b8;text-transform:uppercase;
                        letter-spacing:0.1em;font-weight:600;margin-bottom:6px">
                Predicted Segment
            </div>
            <div style="font-size:3rem;font-weight:900;
                        background:linear-gradient(135deg,{cluster_color},{cluster_color}aa);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                        margin-bottom:10px">
                CLUSTER {pred_cluster}
            </div>
            <div style="display:flex;justify-content:center;gap:16px;flex-wrap:wrap;margin-top:12px">
                <span style="background:{cluster_color}33;color:{cluster_color};
                             padding:6px 16px;border-radius:20px;font-weight:600;font-size:0.9rem">
                    RFM: {rfm_seg}
                </span>
                <span style="background:rgba(6,182,212,0.2);color:#67e8f9;
                             padding:6px 16px;border-radius:20px;font-weight:600;font-size:0.9rem">
                    CLV: {clv_val:.0f}/1000
                </span>
                <span style="background:rgba(16,185,129,0.2);color:#6ee7b7;
                             padding:6px 16px;border-radius:20px;font-weight:600;font-size:0.9rem">
                    Tier: {clv_tier}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Side-by-side Comparison ──────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 How You Compare to Your Cluster")
        if pred_cluster in df["Cluster"].values:
            cluster_df = df[df["Cluster"] == pred_cluster]
            compare_cols = {
                "Income":   income,
                "Recency":  recency,
                "MntWines": mnt_wines,
                "MntMeatProducts": mnt_meat,
                "Frequency": frequency,
            }
            compare_rows = []
            for feat, user_val in compare_cols.items():
                if feat in cluster_df.columns:
                    avg_val = cluster_df[feat].mean()
                    compare_rows.append({
                        "Feature":       feat,
                        "Your Value":    user_val,
                        "Cluster Avg":   round(avg_val, 1),
                        "Difference %":  round((user_val - avg_val) / (abs(avg_val) + 1) * 100, 1),
                    })
            if compare_rows:
                cmp_df = pd.DataFrame(compare_rows)
                fig_cmp = go.Figure()
                fig_cmp.add_trace(go.Bar(
                    name="Your Value",
                    x=cmp_df["Feature"],
                    y=cmp_df["Your Value"],
                    marker_color="#7c3aed",
                ))
                fig_cmp.add_trace(go.Bar(
                    name=f"Cluster {pred_cluster} Avg",
                    x=cmp_df["Feature"],
                    y=cmp_df["Cluster Avg"],
                    marker_color="#06b6d4",
                ))
                fig_cmp = plotly_dark_layout(fig_cmp, height=350)
                fig_cmp.update_layout(barmode="group")
                st.plotly_chart(fig_cmp, use_container_width=True)

    with col2:
        st.markdown("### 🗺️ Your Position in Cluster Space")
        # Show 2D scatter with this customer highlighted
        plot_df = df[["PCA1", "PCA2", "Cluster"]].copy() if "PCA1" in df.columns else None
        if plot_df is not None:
            try:
                from sklearn.decomposition import PCA as _PCA
                pca_2d = engine.pca_2d
                cust_coords = pca_2d.transform(X_single)
                fig_pos = px.scatter(
                    plot_df.sample(min(800, len(plot_df))),
                    x="PCA1", y="PCA2",
                    color=plot_df["Cluster"].astype(str).iloc[:min(800, len(plot_df))],
                    color_discrete_sequence=CLUSTER_COLORS,
                    opacity=0.4,
                )
                fig_pos.add_trace(go.Scatter(
                    x=[cust_coords[0, 0]],
                    y=[cust_coords[0, 1]],
                    mode="markers",
                    marker=dict(size=20, color=cluster_color, symbol="star",
                                line=dict(color="white", width=2)),
                    name="⭐ You",
                ))
                fig_pos = plotly_dark_layout(fig_pos, "Your Position (⭐) in Customer Space", height=350)
                st.plotly_chart(fig_pos, use_container_width=True)
            except Exception as e:
                st.info(f"Position plot unavailable: {e}")

except Exception as e:
    st.error(f"Prediction error: {e}")
    st.info("Adjust the sliders on the sidebar and try again.")
