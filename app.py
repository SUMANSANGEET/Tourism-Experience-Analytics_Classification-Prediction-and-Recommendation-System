import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
import base64
BASE_DIR = Path(__file__).resolve().parent
BACKGROUND_IMAGE = BASE_DIR / "assets" / "tourism_analytics_background.png"

def get_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

from utils import load_all
import advanced as A
import filters as F
from page_views import (
    executive_dashboard, interactive_analytics, traveler_360,
    predict_visit_mode, recommend_attractions, trip_planner,
    model_lab, business_insights,
)

st.set_page_config(
    page_title="Tourism Experience Analytics",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# THEME — "Java–Bali field notes": deep volcanic teal, temple-brick clay,
# sun-bleached sand. Grounded in the dataset itself (Bali beaches & temples,
# Yogyakarta palaces, Malang volcanoes).
# ----------------------------------------------------------------------
if not BACKGROUND_IMAGE.exists():
    raise FileNotFoundError(
        f"Background image not found: {BACKGROUND_IMAGE}"
    )

BACKGROUND_B64 = get_image_base64(BACKGROUND_IMAGE)
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Work+Sans:wght@400;500;600;700&display=swap');

:root {
    --ink:      #16302E;
    --teal:     #163A36;
    --teal-2:   #1F5C55;
    --clay:     #C97A3D;
    --clay-2:   #A85A2A;
    --gold:     #D9A441;
    --sand:     #F7F4EE;
    --sand-2:   #EFE8DA;
    --line:     #DDD3BE;
}

html, body, [class*="css"]  {
    font-family: 'Work Sans', sans-serif;
    color: var(--ink);
}

h1, h2, h3, h4 {
    font-family: 'Fraunces', serif;
    color: var(--teal);
    letter-spacing: -0.01em;
}

.stApp {
    background: var(--sand);
}

section[data-testid="stSidebar"] {
    background: var(--teal);
}
section[data-testid="stSidebar"] * {
    color: #F0EEE3 !important;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    color: var(--gold) !important;
}

/* Hero banner */
.hero {
    background-image:
        linear-gradient(
            90deg,
            rgba(10, 42, 39, 0.88) 0%,
            rgba(10, 42, 39, 0.68) 45%,
            rgba(10, 42, 39, 0.25) 100%
        ),
        url("assets/tourism_analytics_background.png");

    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;

    min-height: 360px;
    border-radius: 14px;
    padding: 3.5rem 3rem;
    margin-bottom: 1.6rem;

    color: #F7F4EE;
    position: relative;
    overflow: hidden;

    border: 1px solid #0F2A27;

    display: flex;
    flex-direction: column;
    justify-content: center;
}
.hero h1 {
    color: #FAF7EF;
    font-size: 2.3rem;
    margin-bottom: 0.35rem;
    font-weight: 600;
}
.hero p {
    color: #D8E4DF;
    font-size: 1.02rem;
    max-width: 640px;
    margin-bottom: 0;
}
.hero .tag {
    display: inline-block;
    background: rgba(217, 164, 65, 0.18);
    border: 1px solid rgba(217, 164, 65, 0.55);
    color: #F1CC85;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    font-size: 0.78rem;
    letter-spacing: 0.02em;
    margin-bottom: 0.9rem;
}

/* KPI cards */
.kpi-card {
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-left: 4px solid var(--clay);
    border-radius: 10px;
    padding: 1.0rem 1.2rem;
    height: 100%;
}
.kpi-card .kpi-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #6B7A76;
    margin-bottom: 0.25rem;
}
.kpi-card .kpi-value {
    font-family: 'Fraunces', serif;
    font-size: 1.9rem;
    color: var(--teal);
    font-weight: 600;
    line-height: 1.1;
}
.kpi-card .kpi-sub {
    font-size: 0.78rem;
    color: #8A9490;
    margin-top: 0.2rem;
}

/* Section headers */
.section-label {
    color: var(--clay-2);
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: -0.4rem;
    font-weight: 600;
}

/* Recommendation cards */
.rec-card {
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 0.95rem 1.1rem;
    margin-bottom: 0.6rem;
}
.rec-card .rec-title {
    font-family: 'Fraunces', serif;
    font-size: 1.08rem;
    color: var(--teal);
    font-weight: 600;
}
.rec-card .rec-meta {
    font-size: 0.82rem;
    color: #7A8783;
}
.rec-badge {
    display: inline-block;
    background: var(--sand-2);
    color: var(--clay-2);
    border-radius: 999px;
    padding: 0.15rem 0.6rem;
    font-size: 0.75rem;
    font-weight: 600;
    float: right;
}

/* Prediction result banner */
.predict-result {
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-left: 5px solid var(--teal-2);
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    margin: 0.6rem 0 1.0rem 0;
}
.predict-result .big {
    font-family: 'Fraunces', serif;
    font-size: 1.6rem;
    color: var(--teal);
    font-weight: 600;
}

div[data-testid="stMetricValue"] {
    color: var(--teal);
    font-family: 'Fraunces', serif;
}

.stTabs [data-baseweb="tab"] {
    font-weight: 600;
    color: var(--ink);
}
.stTabs [aria-selected="true"] {
    color: var(--clay-2) !important;
}

button[kind="primary"], .stButton>button {
    background: var(--clay);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
}
.stButton>button:hover {
    background: var(--clay-2);
    color: white;
}

footer {visibility: hidden;}

.insight-box {
    background-color: #F7F4EE; border-left: 4px solid var(--teal-2);
    padding: 0.85rem 1.05rem; border-radius: 8px; margin-bottom: 0.7rem;
    color: var(--ink);
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

PLOTLY_TEMPLATE = "plotly_white"
COLOR_SEQ = ["#163A36", "#C97A3D", "#1F5C55", "#D9A441", "#A85A2A", "#5E8C85", "#E0B368"]
px.defaults.color_discrete_sequence = COLOR_SEQ
px.defaults.template = PLOTLY_TEMPLATE


data = load_all()
lookup = data["lookup"]
master = lookup["master_sample"]

# ----------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------------------------
st.sidebar.markdown("## 🌋 Tourism Experience Analytics")
st.sidebar.caption("Classification · Prediction · Recommendation · Intelligence")
NAV_PAGES = [
    "🏝️ Executive Dashboard",
    "📊 Interactive Analytics",
    "👤 Traveler 360°",
    "🧭 Predict Visit Mode",
    "🎯 Recommend Attractions",
    "🗺️ AI Trip Planner",
    "🧪 Model Performance",
    "💼 Business Insights",
]
page = st.sidebar.radio("Navigate", NAV_PAGES, label_visibility="collapsed")

# Global filters apply to the data-exploration pages only (prediction/recommendation/
# trip-planning intentionally use the full trained population, not a filtered slice).
FILTERED_PAGES = {"🏝️ Executive Dashboard", "📊 Interactive Analytics"}
filters = F.render_filter_panel(master)
filters_active = F.active_filter_count(filters) > 0
filtered_master = F.apply_filters(master, filters) if page in FILTERED_PAGES else master

st.sidebar.markdown("---")
st.sidebar.caption(
    "Built on a real multi-table tourism dataset — 52,930 transactions, "
    "33,530 users, 30 rated attractions across Indonesia."
)
st.sidebar.caption("P Suman Sangeet · INNOVEXIS Data Science & Gen AI Internship")

# ========================================================================
# ROUTING
# ========================================================================
if page == "🏝️ Executive Dashboard":
    executive_dashboard.render(data, filtered_master, filters_active)

elif page == "📊 Interactive Analytics":
    interactive_analytics.render(data, filtered_master, filters_active)

elif page == "👤 Traveler 360°":
    traveler_360.render(data)

elif page == "🧭 Predict Visit Mode":
    predict_visit_mode.render(data)

elif page == "🎯 Recommend Attractions":
    recommend_attractions.render(data)

elif page == "🗺️ AI Trip Planner":
    trip_planner.render(data)

elif page == "🧪 Model Performance":
    model_lab.render(data)

elif page == "💼 Business Insights":
    business_insights.render(data)
