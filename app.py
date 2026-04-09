"""
SegmentAI — Advanced AI-Powered Customer Segmentation Platform
Landing page / entry point.
"""

import streamlit as st

st.set_page_config(
    page_title="SegmentAI — Customer Intelligence Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.styles import inject_styles, CLUSTER_COLORS, ACCENT_PURPLE, ACCENT_CYAN

inject_styles()

# ── Hero Section ──────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div style="text-align:center;padding:60px 40px 40px;
                background:radial-gradient(ellipse at center, {ACCENT_PURPLE}18 0%, transparent 70%);
                border-radius:24px;margin-bottom:40px;">
        <div style="font-size:4rem;margin-bottom:12px">🧠</div>
        <h1 style="font-size:3.2rem;font-weight:900;
                   background:linear-gradient(135deg, {ACCENT_PURPLE}, {ACCENT_CYAN});
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                   margin-bottom:12px;line-height:1.1">
            SegmentAI
        </h1>
        <p style="font-size:1.25rem;color:#94a3b8;max-width:600px;margin:0 auto 24px;line-height:1.6">
            Production-grade AI customer segmentation platform with multi-algorithm AutoML,
            SHAP explainability, CLV prediction, and LLM-powered marketing intelligence.
        </p>
        <div style="display:flex;justify-content:center;gap:12px;flex-wrap:wrap;">
            <span style="background:{ACCENT_PURPLE}33;color:#a78bfa;padding:6px 18px;
                         border-radius:20px;font-weight:600;font-size:0.9rem">
                🤖 4 Algorithms + AutoML
            </span>
            <span style="background:rgba(6,182,212,0.2);color:#67e8f9;padding:6px 18px;
                         border-radius:20px;font-weight:600;font-size:0.9rem">
                🧬 SHAP Explainability
            </span>
            <span style="background:rgba(16,185,129,0.2);color:#6ee7b7;padding:6px 18px;
                         border-radius:20px;font-weight:600;font-size:0.9rem">
                💰 CLV Prediction
            </span>
            <span style="background:rgba(245,158,11,0.2);color:#fcd34d;padding:6px 18px;
                         border-radius:20px;font-weight:600;font-size:0.9rem">
                ⚠️ Anomaly Detection
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Navigation Cards ──────────────────────────────────────────────────────────
st.markdown("## 🗺️ Platform Modules")
st.markdown("<br>", unsafe_allow_html=True)

PAGES = [
    {
        "icon": "🏠", "title": "Overview",
        "desc": "3D cluster visualization, KPI dashboard, and segment profiles at a glance.",
        "page": "01_Overview",
        "color": ACCENT_PURPLE,
    },
    {
        "icon": "🤖", "title": "Algorithm Comparison",
        "desc": "KMeans vs DBSCAN vs Agglomerative vs GMM — AutoML selects the best.",
        "page": "02_Algorithms",
        "color": "#06b6d4",
    },
    {
        "icon": "📊", "title": "RFM Analysis",
        "desc": "Recency, Frequency, Monetary segmentation with heatmaps and radar charts.",
        "page": "03_RFM_Analysis",
        "color": "#ec4899",
    },
    {
        "icon": "💰", "title": "CLV Prediction",
        "desc": "GradientBoosting predicts customer lifetime value and profitability rank.",
        "page": "04_CLV_Prediction",
        "color": "#10b981",
    },
    {
        "icon": "🧬", "title": "Explainability (SHAP)",
        "desc": "Why is each customer in their segment? Global and per-customer SHAP waterfall.",
        "page": "05_Explainability",
        "color": "#f59e0b",
    },
    {
        "icon": "🚀", "title": "AI Recommendations",
        "desc": "Gemini LLM generates personalized campaign strategies per segment.",
        "page": "06_Recommendations",
        "color": "#8b5cf6",
    },
    {
        "icon": "🔮", "title": "What-If Simulator",
        "desc": "Slide any attribute and see which cluster a customer would be assigned to — live.",
        "page": "07_WhatIf_Simulator",
        "color": "#3b82f6",
    },
    {
        "icon": "⚠️", "title": "Anomaly Detection",
        "desc": "Isolation Forest spots unusual customers whose behavior defies their segment.",
        "page": "08_Anomaly_Detection",
        "color": "#ef4444",
    },
    {
        "icon": "🔄", "title": "Customer Evolution",
        "desc": "Sankey diagrams track how customers migrate between segments over time.",
        "page": "09_Customer_Evolution",
        "color": "#14b8a6",
    },
]

rows = [PAGES[i:i+3] for i in range(0, len(PAGES), 3)]
for row in rows:
    cols = st.columns(3)
    for col, page in zip(cols, row):
        with col:
            st.markdown(
                f"""
                <div style="background:rgba(255,255,255,0.04);
                            border:1px solid {page['color']}44;
                            border-radius:14px;padding:24px;
                            transition:all 0.2s;cursor:pointer;
                            margin-bottom:16px;
                            border-top:3px solid {page['color']}">
                    <div style="font-size:2rem;margin-bottom:10px">{page['icon']}</div>
                    <div style="font-weight:700;font-size:1.05rem;
                                color:#f1f5f9;margin-bottom:8px">{page['title']}</div>
                    <div style="color:#94a3b8;font-size:0.87rem;
                                line-height:1.5">{page['desc']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ── Tech Stack ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 🛠️ Technology Stack")

tech = [
    ("Python 3.11+",        "Core language"),
    ("Scikit-learn",        "ML algorithms"),
    ("SHAP",                "Explainability"),
    ("MLflow",              "Experiment tracking"),
    ("Streamlit",           "Dashboard"),
    ("Plotly",              "Interactive charts"),
    ("FastAPI",             "REST API"),
    ("SQLite / PostgreSQL", "Persistence"),
    ("Google Gemini",       "LLM campaigns"),
    ("Docker",              "Containerization"),
]
tech_cols = st.columns(5)
for i, (name, role) in enumerate(tech):
    with tech_cols[i % 5]:
        st.markdown(
            f"""
            <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);
                        border-radius:10px;padding:14px;text-align:center;margin-bottom:10px">
                <div style="font-weight:700;color:#f1f5f9;font-size:0.9rem">{name}</div>
                <div style="color:#64748b;font-size:0.75rem;margin-top:4px">{role}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── Quick Start Info ──────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("⚡ Quick Start Guide"):
    st.markdown("""
    ```bash
    # 1. Clone & enter directory
    cd customer-segmentation-dashboard

    # 2. Set up environment (optional — Gemini LLM)
    cp .env.example .env
    # Edit .env and add your GEMINI_API_KEY

    # 3. Install dependencies
    pip install -r requirements.txt

    # 4. Run the dashboard
    streamlit run app.py

    # 5. (Optional) Start FastAPI backend in a second terminal
    uvicorn backend.main:app --reload --port 8000
    # API docs: http://localhost:8000/docs

    # 6. (Optional) View MLflow experiments
    mlflow ui --port 5000
    # Dashboard: http://localhost:5000

    # 7. (Optional) Docker
    docker-compose up --build
    ```
    """)

st.markdown(
    f"""
    <div style="text-align:center;padding:32px;color:#475569;font-size:0.85rem">
        Built with ❤️ by Nikhil More ·
        <a href="https://github.com/Khiladi-786" style="color:{ACCENT_PURPLE}">GitHub</a> ·
        <a href="https://www.linkedin.com/in/nikhil-moretech" style="color:{ACCENT_CYAN}">LinkedIn</a>
    </div>
    """,
    unsafe_allow_html=True,
)