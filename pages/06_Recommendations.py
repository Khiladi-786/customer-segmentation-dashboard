"""
Page 06 — AI-Powered Recommendations
LLM-generated campaign cards · Downloadable PDF report
"""

import streamlit as st
import pandas as pd

from utils.styles import inject_styles, page_header, CLUSTER_COLORS
from core.model_cache import get_models
from core.llm_recommender import LLMRecommender

st.set_page_config(
    page_title="Recommendations | SegmentAI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_styles()

models = get_models()
df     = models["df"]

page_header(
    "AI-Powered Marketing Recommendations",
    "Google Gemini LLM generates personalized campaign strategies per segment — or uses curated fallback templates.",
    "🚀",
)

# ── LLM Status Banner ─────────────────────────────────────────────────────────
rec = LLMRecommender()
if rec.use_llm:
    st.success("🤖 **Gemini AI Active** — generating live personalized strategies from real cluster data.")
else:
    st.info(
        "📋 **Template Mode** — Gemini API key not set. Using curated marketing templates. "
        "Add `GEMINI_API_KEY` to `.env` to enable live LLM generation."
    )

st.markdown("---")

# ── Generate Recommendations ─────────────────────────────────────────────────
@st.cache_data(show_spinner="🤖 Generating AI recommendations …")
def load_recommendations(_df):
    recommender = LLMRecommender()
    return recommender.recommend_all(_df, cluster_col="Cluster")

if "Cluster" not in df.columns:
    st.error("Cluster column not found.")
    st.stop()

recs = load_recommendations(df)

# ── Campaign Cards ────────────────────────────────────────────────────────────
st.markdown("### 🎯 Campaign Strategies by Segment")

TIER_COLORS = {
    "High":   ("🟢", "#10b981"),
    "Medium": ("🟡", "#f59e0b"),
    "Low":    ("🔴", "#ef4444"),
}

for i, r in enumerate(recs):
    cluster_id = r.get("cluster_id", i)
    rfm_seg    = r.get("rfm_segment", "Unknown")
    color      = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]
    bp         = r.get("budget_priority", "Medium")
    bp_icon, bp_color = TIER_COLORS.get(bp, ("🟡", "#f59e0b"))

    with st.expander(f"🎯 Cluster {cluster_id} — {r.get('campaign_name', 'Campaign')}  •  RFM: {rfm_seg}", expanded=(i == 0)):
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(
                f"""
                <div style="background:rgba(255,255,255,0.04);border:1px solid {color}44;border-radius:14px;padding:20px;">
                    <h3 style="color:{color};margin:0 0 12px 0">{r.get('campaign_name', '—')}</h3>
                    <p style="color:#cbd5e1;line-height:1.6;margin-bottom:16px">{r.get('strategy', '—')}</p>
                    <div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:16px">
                        <span style="background:rgba(124,58,237,0.2);color:#a78bfa;padding:4px 12px;border-radius:20px;font-size:0.8rem;font-weight:600">
                            📡 {r.get('primary_channel','—')}
                        </span>
                        <span style="background:rgba(6,182,212,0.2);color:#67e8f9;padding:4px 12px;border-radius:20px;font-size:0.8rem;font-weight:600">
                            🎁 {r.get('offer_type','—')}
                        </span>
                        <span style="background:rgba(16,185,129,0.2);color:#6ee7b7;padding:4px 12px;border-radius:20px;font-size:0.8rem;font-weight:600">
                            📈 ROI: {r.get('expected_roi','?')}
                        </span>
                        <span style="background:rgba(245,158,11,0.2);color:#fcd34d;padding:4px 12px;border-radius:20px;font-size:0.8rem;font-weight:600">
                            {bp_icon} Budget: {bp}
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(f"**📋 Action Plan — Cluster {cluster_id}**")
            actions = r.get("actions", [])
            for j, action in enumerate(actions, 1):
                st.markdown(
                    f'<div style="display:flex;align-items:flex-start;gap:10px;margin:8px 0;padding:8px;'
                    f'background:rgba(255,255,255,0.03);border-radius:8px;border-left:3px solid {color}">'
                    f'<span style="color:{color};font-weight:700;min-width:20px">{j}.</span>'
                    f'<span style="color:#cbd5e1;font-size:0.88rem">{action}</span></div>',
                    unsafe_allow_html=True,
                )

st.markdown("---")

# ── Download PDF ──────────────────────────────────────────────────────────────
st.markdown("### 📥 Download Campaign Report")
col_dl1, col_dl2 = st.columns(2)

with col_dl1:
    if st.button("📄 Generate PDF Report", use_container_width=True):
        with st.spinner("Generating PDF …"):
            try:
                from utils.report_generator import generate_pdf_report
                from pathlib import Path
                artifacts = Path(__file__).parent.parent / "artifacts"
                artifacts.mkdir(exist_ok=True)
                path = generate_pdf_report(str(artifacts / "segmentation_report.pdf"))
                with open(path, "rb") as f:
                    st.download_button(
                        "⬇️ Download PDF",
                        f.read(),
                        file_name="segmentation_report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
            except Exception as e:
                st.error(f"PDF generation failed: {e}")

with col_dl2:
    if st.button("📊 Generate Excel Report", use_container_width=True):
        with st.spinner("Generating Excel …"):
            try:
                from utils.report_generator import generate_excel_report
                from pathlib import Path
                artifacts = Path(__file__).parent.parent / "artifacts"
                artifacts.mkdir(exist_ok=True)
                path = generate_excel_report(df, str(artifacts / "segmentation_report.xlsx"))
                with open(path, "rb") as f:
                    st.download_button(
                        "⬇️ Download Excel",
                        f.read(),
                        file_name="segmentation_report.xlsx",
                        mime="application/vnd.ms-excel",
                        use_container_width=True,
                    )
            except Exception as e:
                st.error(f"Excel generation failed: {e}")

# ── Strategy Summary Table ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📊 Strategy Overview Table")
summary_rows = []
for r in recs:
    summary_rows.append({
        "Cluster":          r.get("cluster_id", "?"),
        "RFM Segment":      r.get("rfm_segment", "—"),
        "Campaign":         r.get("campaign_name", "—"),
        "Channel":          r.get("primary_channel", "—"),
        "Offer":            r.get("offer_type", "—"),
        "Budget Priority":  r.get("budget_priority", "—"),
        "Expected ROI":     r.get("expected_roi", "—"),
    })
if summary_rows:
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
