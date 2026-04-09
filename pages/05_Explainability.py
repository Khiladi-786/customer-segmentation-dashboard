"""
Page 05 — Explainability (SHAP)
Global feature importance · Per-cluster impact heatmap · Customer-level waterfall
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from utils.styles import inject_styles, page_header, plotly_dark_layout, CLUSTER_COLORS
from core.model_cache import get_models

st.set_page_config(
    page_title="Explainability | SegmentAI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_styles()

models = get_models()
df     = models["df"]
xai    = models["xai"]
fi_df  = models["fi_df"]

page_header(
    "Explainable AI (SHAP)",
    "Understand WHY each customer belongs to their segment — powered by SHAP and Random Forest surrogate.",
    "🧬",
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🌍 Global Importance",
    "🗺️ Cluster Feature Impact",
    "👤 Customer-Level Explanation",
])

# ── Tab 1: Global Importance ─────────────────────────────────────────────────
with tab1:
    st.markdown("### 🌍 Global Feature Importance")
    st.markdown(
        "These features have the greatest influence on cluster assignment across all customers, "
        "measured by mean absolute SHAP values from a surrogate Random Forest."
    )

    global_imp = xai.global_feature_importance() if xai.shap_values is not None else fi_df
    if global_imp.empty:
        global_imp = fi_df

    col1, col2 = st.columns([2, 1])

    with col1:
        if not global_imp.empty:
            top_n = min(20, len(global_imp))
            plot_data = global_imp.head(top_n)
            value_col = "SHAP_Importance" if "SHAP_Importance" in plot_data.columns else "Importance"

            fig_bar = go.Figure(go.Bar(
                x=plot_data[value_col],
                y=plot_data["Feature"],
                orientation="h",
                marker=dict(
                    color=plot_data[value_col],
                    colorscale="Purples",
                    line=dict(width=0),
                ),
            ))
            fig_bar = plotly_dark_layout(fig_bar, "Top Features Driving Cluster Assignment", height=500)
            fig_bar.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.markdown("**Top 10 Features**")
        if not global_imp.empty:
            value_col = "SHAP_Importance" if "SHAP_Importance" in global_imp.columns else "Importance"
            for _, row in global_imp.head(10).iterrows():
                val = row[value_col]
                pct = val / global_imp[value_col].max() * 100 if global_imp[value_col].max() > 0 else 0
                st.markdown(
                    f'<div style="margin:6px 0">'
                    f'<div style="display:flex;justify-content:space-between;font-size:0.85rem">'
                    f'<span style="color:#f1f5f9">{row["Feature"]}</span>'
                    f'<span style="color:#7c3aed;font-weight:600">{val:.4f}</span>'
                    f'</div>'
                    f'<div style="background:rgba(255,255,255,0.05);border-radius:4px;height:6px;margin-top:3px">'
                    f'<div style="background:linear-gradient(90deg,#7c3aed,#06b6d4);border-radius:4px;height:6px;width:{pct:.0f}%"></div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

# ── Tab 2: Cluster Impact Heatmap ────────────────────────────────────────────
with tab2:
    st.markdown("### 🗺️ Feature Impact per Cluster")
    st.markdown(
        "Each cell shows the **mean SHAP value** of that feature for customers in that cluster. "
        "Positive = feature pushes customers *toward* this cluster."
    )

    impact_df = xai.cluster_feature_impact()
    if not impact_df.empty:
        fig_heat = go.Figure(data=go.Heatmap(
            z=impact_df.values,
            x=impact_df.columns.tolist(),
            y=impact_df.index.tolist(),
            colorscale=[
                [0.0,  "#ef4444"],
                [0.5,  "#0a0a0f"],
                [1.0,  "#7c3aed"],
            ],
            zmid=0,
            hoverongaps=False,
            texttemplate="%{z:.3f}",
            textfont=dict(color="white", size=9),
        ))
        fig_heat = plotly_dark_layout(fig_heat, height=450)
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        # Fallback: use RF feature importance per cluster from fi_df
        st.info("Full SHAP heatmap requires SHAP fitting. Showing RF feature importance instead.")
        if "Cluster" in df.columns and not fi_df.empty:
            top_feats = fi_df.head(10)["Feature"].tolist()
            available = [f for f in top_feats if f in df.columns]
            if available:
                cluster_means = df.groupby("Cluster")[available].mean()
                from sklearn.preprocessing import StandardScaler
                normed = pd.DataFrame(
                    StandardScaler().fit_transform(cluster_means),
                    index=cluster_means.index,
                    columns=cluster_means.columns,
                )
                fig_heat = go.Figure(data=go.Heatmap(
                    z=normed.values,
                    x=normed.columns.tolist(),
                    y=[f"Cluster {c}" for c in normed.index],
                    colorscale="Viridis",
                    texttemplate="%{z:.2f}",
                    textfont=dict(color="white", size=10),
                ))
                fig_heat = plotly_dark_layout(fig_heat, "Normalized Feature Means per Cluster", height=400)
                st.plotly_chart(fig_heat, use_container_width=True)

# ── Tab 3: Customer Explanation ───────────────────────────────────────────────
with tab3:
    st.markdown("### 👤 Individual Customer Explanation")
    st.markdown(
        "Select a customer to see **why** the model assigned them to their cluster — "
        "shown as positive (pushes toward cluster) and negative (pushes away) feature contributions."
    )

    sample_size = xai.get_sample_size() if xai else 0
    if sample_size > 0:
        col_a, col_b = st.columns([1, 3])
        with col_a:
            cust_idx = st.slider("Customer Index", 0, max(0, sample_size - 1), 0)
        with col_b:
            cust_data = xai.customer_shap(cust_idx)
            if cust_data:
                shap_vals    = cust_data["shap_values"]
                feat_names   = cust_data["feature_names"]
                cluster      = cust_data["cluster"]
                base_val     = cust_data["base_value"]

                explanation_df = pd.DataFrame({
                    "Feature":    feat_names,
                    "SHAP Value": shap_vals,
                    "Direction":  ["↑ Supports" if v > 0 else "↓ Against" for v in shap_vals],
                }).sort_values("SHAP Value", key=abs, ascending=False).head(15)

                colors_list = [
                    "#10b981" if v > 0 else "#ef4444"
                    for v in explanation_df["SHAP Value"]
                ]

                fig_wf = go.Figure(go.Bar(
                    x=explanation_df["SHAP Value"],
                    y=explanation_df["Feature"],
                    orientation="h",
                    marker=dict(color=colors_list, line=dict(width=0)),
                    text=[f"{v:+.3f}" for v in explanation_df["SHAP Value"]],
                    textposition="outside",
                    textfont=dict(color="white"),
                ))
                fig_wf = plotly_dark_layout(
                    fig_wf,
                    f"Customer #{cust_idx} → Cluster {cluster} | Feature Contributions",
                    height=420,
                )
                fig_wf.add_vline(x=0, line=dict(color="rgba(255,255,255,0.3)", dash="dash"))
                fig_wf.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_wf, use_container_width=True)

                st.info(
                    f"🎯 **Cluster {cluster}** | "
                    f"Base prediction: {base_val:.3f} | "
                    f"Green bars increase cluster probability, Red bars decrease it."
                )
    else:
        st.info("SHAP explainer not fitted — check SHAP library installation.")
