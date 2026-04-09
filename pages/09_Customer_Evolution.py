"""
Page 09 — Customer Evolution Tracking
Sankey cluster migrations · Time series cluster sizes · Individual customer journey
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from utils.styles import inject_styles, page_header, plotly_dark_layout, CLUSTER_COLORS
from core.model_cache import get_models

st.set_page_config(
    page_title="Customer Evolution | SegmentAI",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_styles()

models      = get_models()
df          = models["df"]
evo_df      = models["evolution_df"]
n_clusters  = models["n_clusters"]

page_header(
    "Customer Evolution Tracking",
    "Track how customers move between segments over time — powered by behavior simulation across 3 periods.",
    "🔄",
)

PERIOD_LABELS = {0: "Now (Q1)", 1: "Q2", 2: "Q3", 3: "Q4"}

# ── Overview Stats ────────────────────────────────────────────────────────────
if evo_df is not None and len(evo_df) > 0:
    # Migrations: customers who changed cluster from period 0 to last
    base = evo_df[evo_df["Period"] == 0].set_index("CustomerIdx")["Cluster"]
    last = evo_df[evo_df["Period"] == evo_df["Period"].max()].set_index("CustomerIdx")["Cluster"]
    common = base.index.intersection(last.index)
    migrations = (base[common] != last[common]).sum()
    stable     = (base[common] == last[common]).sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tracked Customers",  f"{len(common):,}")
    c2.metric("Stable Customers",   f"{stable:,}",     delta=f"{stable/max(len(common),1)*100:.0f}%")
    c3.metric("Migrated Clusters",  f"{migrations:,}", delta=f"-{migrations/max(len(common),1)*100:.0f}%")
    c4.metric("Periods Simulated",  str(evo_df["Period"].nunique()))

    st.markdown("---")

    # ── Sankey Diagram ───────────────────────────────────────────────────────
    st.markdown("### 🌊 Sankey — Cluster Migration Flow")

    periods = sorted(evo_df["Period"].unique())
    if len(periods) >= 2:
        # Build Sankey from period 0 → period 1 → period 2 → period 3
        all_nodes  = []
        node_map   = {}
        link_source = []
        link_target = []
        link_value  = []
        link_color  = []

        for p in periods:
            for c in range(n_clusters):
                node_label = f"{PERIOD_LABELS.get(p, f'P{p}')} | C{c}"
                node_map[(p, c)] = len(all_nodes)
                all_nodes.append(node_label)

        for p_from, p_to in zip(periods[:-1], periods[1:]):
            p_from_df = evo_df[evo_df["Period"] == p_from].set_index("CustomerIdx")["Cluster"]
            p_to_df   = evo_df[evo_df["Period"] == p_to].set_index("CustomerIdx")["Cluster"]
            common_ids = p_from_df.index.intersection(p_to_df.index)
            trans = pd.DataFrame({
                "from_c": p_from_df[common_ids].values,
                "to_c":   p_to_df[common_ids].values,
            }).groupby(["from_c", "to_c"]).size().reset_index(name="count")

            for _, row in trans.iterrows():
                fc, tc, cnt = int(row["from_c"]), int(row["to_c"]), int(row["count"])
                link_source.append(node_map[(p_from, fc)])
                link_target.append(node_map[(p_to, tc)])
                link_value.append(cnt)
                link_color.append(CLUSTER_COLORS[fc % len(CLUSTER_COLORS)] + "99")

        node_colors = []
        for (p, c) in sorted(node_map.keys(), key=lambda x: node_map[x]):
            node_colors.append(CLUSTER_COLORS[c % len(CLUSTER_COLORS)] + "cc")

        fig_sankey = go.Figure(go.Sankey(
            node=dict(
                pad=15,
                thickness=25,
                line=dict(color="rgba(255,255,255,0.1)", width=0.5),
                label=all_nodes,
                color=node_colors,
                hovertemplate="%{label}<br>Customers: %{value}<extra></extra>",
            ),
            link=dict(
                source=link_source,
                target=link_target,
                value=link_value,
                color=link_color,
                hovertemplate="From %{source.label}<br>To %{target.label}<br>Count: %{value}<extra></extra>",
            ),
        ))
        fig_sankey = plotly_dark_layout(fig_sankey, "Customer Cluster Migration Flow", height=520)
        st.plotly_chart(fig_sankey, use_container_width=True)

    st.markdown("---")

    # ── Cluster Size Over Time ────────────────────────────────────────────────
    st.markdown("### 📈 Cluster Size Over Time")
    size_df = evo_df.groupby(["Period", "Cluster"]).size().reset_index(name="Count")
    size_df["Period_Label"] = size_df["Period"].map(PERIOD_LABELS)
    size_df["Cluster_str"]  = size_df["Cluster"].astype(str)

    fig_line = px.line(
        size_df, x="Period_Label", y="Count",
        color="Cluster_str",
        color_discrete_sequence=CLUSTER_COLORS,
        markers=True,
        labels={"Count": "Customers", "Period_Label": "Time Period", "Cluster_str": "Cluster"},
    )
    fig_line.update_traces(line=dict(width=3), marker=dict(size=8))
    fig_line = plotly_dark_layout(fig_line, "Cluster Population Change Over Time", height=380)
    st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("---")

    # ── Individual Journey ────────────────────────────────────────────────────
    st.markdown("### 👤 Individual Customer Journey")

    max_idx = evo_df["CustomerIdx"].max()
    cust_idx = st.number_input(
        "Enter Customer Index", min_value=0, max_value=int(max_idx), value=0
    )

    journey = evo_df[evo_df["CustomerIdx"] == cust_idx].sort_values("Period")
    if len(journey) > 0:
        journey["Period_Label"] = journey["Period"].map(PERIOD_LABELS)

        fig_journey = go.Figure()
        fig_journey.add_trace(go.Scatter(
            x=journey["Period_Label"],
            y=journey["Cluster"],
            mode="lines+markers+text",
            line=dict(color="#7c3aed", width=4),
            marker=dict(
                size=16,
                color=[CLUSTER_COLORS[int(c) % len(CLUSTER_COLORS)] for c in journey["Cluster"]],
                line=dict(color="white", width=2),
            ),
            text=[f"C{c}" for c in journey["Cluster"]],
            textposition="top center",
            textfont=dict(color="white", size=13),
            name=f"Customer {cust_idx}",
        ))
        fig_journey = plotly_dark_layout(
            fig_journey, f"Customer #{cust_idx} — Cluster Journey Over Time", height=350
        )
        fig_journey.update_layout(
            yaxis=dict(
                title="Cluster ID",
                tickmode="array",
                tickvals=list(range(n_clusters)),
                ticktext=[f"Cluster {c}" for c in range(n_clusters)],
            ),
        )
        st.plotly_chart(fig_journey, use_container_width=True)

        # Journey summary
        if len(journey) >= 2:
            start_c = int(journey.iloc[0]["Cluster"])
            end_c   = int(journey.iloc[-1]["Cluster"])
            if start_c == end_c:
                st.success(f"✅ Customer #{cust_idx} remained in **Cluster {start_c}** throughout all periods.")
            else:
                st.warning(f"🔄 Customer #{cust_idx} migrated from **Cluster {start_c}** → **Cluster {end_c}**.")
    else:
        st.info(f"No evolution data found for customer index {cust_idx}.")

    st.markdown("---")

    # ── Migration Matrix ──────────────────────────────────────────────────────
    st.markdown("### 🔢 Migration Matrix (Period 0 → Latest)")
    base     = evo_df[evo_df["Period"] == 0].set_index("CustomerIdx")["Cluster"]
    last_p   = evo_df["Period"].max()
    last_per = evo_df[evo_df["Period"] == last_p].set_index("CustomerIdx")["Cluster"]
    common   = base.index.intersection(last_per.index)

    matrix = pd.crosstab(
        base[common].rename("From (Now)").map(lambda x: f"C{x}"),
        last_per[common].rename("To (Latest)").map(lambda x: f"C{x}"),
    )

    fig_mat = px.imshow(
        matrix, text_auto=True,
        color_continuous_scale="Purples",
        labels={"x": "To Cluster (Latest)", "y": "From Cluster (Now)", "color": "Customers"},
    )
    fig_mat = plotly_dark_layout(fig_mat, "Cluster Migration Matrix", height=380)
    st.plotly_chart(fig_mat, use_container_width=True)
    st.caption("Diagonal cells = stable customers. Off-diagonal = migrations.")

else:
    st.warning("Evolution data not found — re-run the training pipeline.")
