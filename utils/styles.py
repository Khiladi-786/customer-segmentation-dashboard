"""
Shared Streamlit CSS design system.
Import inject_styles() at the top of every page.
"""

import streamlit as st

ACCENT_PURPLE = "#7c3aed"
ACCENT_CYAN   = "#06b6d4"
ACCENT_PINK   = "#ec4899"
ACCENT_GREEN  = "#10b981"
ACCENT_AMBER  = "#f59e0b"

DARK_BG       = "#0a0a0f"
CARD_BG       = "rgba(255, 255, 255, 0.04)"
BORDER_COLOR  = "rgba(255, 255, 255, 0.08)"
TEXT_PRIMARY  = "#f1f5f9"
TEXT_MUTED    = "#94a3b8"

CLUSTER_COLORS = [
    "#7c3aed", "#06b6d4", "#ec4899",
    "#10b981", "#f59e0b", "#ef4444",
    "#8b5cf6", "#3b82f6", "#14b8a6",
]


def inject_styles() -> None:
    """Inject the full dark-theme CSS into the current Streamlit page."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        /* ── Base ── */
        html, body, [class*="css"], .stApp {{
            font-family: 'Inter', -apple-system, sans-serif !important;
            background-color: {DARK_BG} !important;
            color: {TEXT_PRIMARY} !important;
        }}

        /* ── Sidebar ── */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #0d0d18 0%, #0a0a0f 100%) !important;
            border-right: 1px solid {BORDER_COLOR} !important;
        }}
        section[data-testid="stSidebar"] .stSelectbox label,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span {{
            color: {TEXT_PRIMARY} !important;
        }}

        /* ── Header ── */
        .main-header {{
            background: linear-gradient(135deg, {ACCENT_PURPLE}22 0%, {ACCENT_CYAN}22 100%);
            border: 1px solid {BORDER_COLOR};
            border-radius: 16px;
            padding: 28px 32px;
            margin-bottom: 28px;
        }}
        .main-header h1 {{
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, {ACCENT_PURPLE}, {ACCENT_CYAN});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 0 0 6px 0;
        }}
        .main-header p {{
            color: {TEXT_MUTED};
            font-size: 0.95rem;
            margin: 0;
        }}

        /* ── Metric Cards ── */
        .metric-card {{
            background: {CARD_BG};
            border: 1px solid {BORDER_COLOR};
            border-radius: 14px;
            padding: 20px 24px;
            text-align: center;
            transition: transform 0.2s, border-color 0.2s;
            backdrop-filter: blur(8px);
        }}
        .metric-card:hover {{
            transform: translateY(-2px);
            border-color: {ACCENT_PURPLE}66;
        }}
        .metric-value {{
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, {ACCENT_PURPLE}, {ACCENT_CYAN});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            line-height: 1;
            margin-bottom: 4px;
        }}
        .metric-label {{
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: {TEXT_MUTED};
            font-weight: 600;
        }}

        /* ── Section Cards ── */
        .section-card {{
            background: {CARD_BG};
            border: 1px solid {BORDER_COLOR};
            border-radius: 14px;
            padding: 24px;
            margin-bottom: 20px;
            backdrop-filter: blur(8px);
        }}
        .section-title {{
            font-size: 1.1rem;
            font-weight: 700;
            color: {TEXT_PRIMARY};
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        /* ── Cluster Badges ── */
        .cluster-badge {{
            display: inline-block;
            padding: 3px 12px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
        }}

        /* ── Streamlit widgets ── */
        .stButton > button {{
            background: linear-gradient(135deg, {ACCENT_PURPLE}, {ACCENT_CYAN}) !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 10px 24px !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
            transition: opacity 0.2s !important;
        }}
        .stButton > button:hover {{
            opacity: 0.85 !important;
        }}
        .stSelectbox > div > div {{
            background: rgba(255,255,255,0.05) !important;
            border: 1px solid {BORDER_COLOR} !important;
            color: {TEXT_PRIMARY} !important;
            border-radius: 10px !important;
        }}
        .stSlider > div {{
            color: {TEXT_PRIMARY} !important;
        }}
        .stSlider [data-baseweb="slider"] {{
            background: {ACCENT_PURPLE}33 !important;
        }}
        .stDataFrame, .stTable {{
            background: {CARD_BG} !important;
        }}
        div[data-testid="stMetric"] {{
            background: {CARD_BG} !important;
            border: 1px solid {BORDER_COLOR} !important;
            border-radius: 12px !important;
            padding: 16px !important;
        }}
        div[data-testid="stMetric"] label {{
            color: {TEXT_MUTED} !important;
        }}
        div[data-testid="stMetric"] div {{
            color: {TEXT_PRIMARY} !important;
        }}
        .stTabs [data-baseweb="tab-list"] {{
            background: transparent !important;
            gap: 4px;
        }}
        .stTabs [data-baseweb="tab"] {{
            background: {CARD_BG} !important;
            border: 1px solid {BORDER_COLOR} !important;
            border-radius: 8px !important;
            color: {TEXT_MUTED} !important;
            padding: 8px 20px !important;
        }}
        .stTabs [aria-selected="true"] {{
            background: {ACCENT_PURPLE}33 !important;
            border-color: {ACCENT_PURPLE} !important;
            color: {TEXT_PRIMARY} !important;
        }}
        h1, h2, h3 {{
            color: {TEXT_PRIMARY} !important;
        }}
        p, li, span {{
            color: {TEXT_MUTED};
        }}
        .stAlert {{
            border-radius: 10px !important;
        }}
        /* Hide Streamlit branding */
        #MainMenu {{ visibility: hidden; }}
        footer {{ visibility: hidden; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, icon: str = "📊", delta: str = "") -> str:
    """Return HTML for a glassmorphism metric card."""
    delta_html = f'<div style="font-size:0.75rem;color:#10b981;margin-top:4px">{delta}</div>' if delta else ""
    return f"""
    <div class="metric-card">
        <div style="font-size:1.5rem;margin-bottom:6px">{icon}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {delta_html}
    </div>
    """


def page_header(title: str, subtitle: str, icon: str = "🧠") -> None:
    """Render the standard glassmorphism page header."""
    st.markdown(
        f"""
        <div class="main-header">
            <h1>{icon} {title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_card(title: str, icon: str = "▶") -> None:
    """Render a section card header."""
    st.markdown(
        f'<div class="section-title">{icon} {title}</div>',
        unsafe_allow_html=True,
    )


def plotly_dark_layout(fig, title: str = "", height: int = 500):
    """Apply the dark theme to a Plotly figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(color="#f1f5f9", size=16)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        font=dict(color="#94a3b8", family="Inter"),
        height=height,
        legend=dict(
            bgcolor="rgba(255,255,255,0.05)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
        ),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig
