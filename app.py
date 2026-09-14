import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
import base64

# ADD THIS
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent
BACKGROUND_IMAGE = BASE_DIR / "assets" / "tourism_analytics_background.png"

def get_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

from utils import (
    get_returning_user_ids,
    get_user_history,
    load_all,
    predict_rating,
    predict_visit_mode,
    recommend_collaborative,
    recommend_for_new_visitor,
    recommend_similar_to,
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
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

PLOTLY_TEMPLATE = "plotly_white"
COLOR_SEQ = ["#163A36", "#C97A3D", "#1F5C55", "#D9A441", "#A85A2A", "#5E8C85", "#E0B368"]
px.defaults.color_discrete_sequence = COLOR_SEQ
px.defaults.template = PLOTLY_TEMPLATE

data = load_all()
lookup = data["lookup"]

# ----------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------------------------
st.sidebar.markdown("## 🌋 Tourism Tourism Experience Analytics: Classification, Prediction, and Recommendation System")
st.sidebar.caption("Classification · Prediction · Recommendation")
page = st.sidebar.radio(
    "Navigate",
    [
        "🏝️ Overview",
        "📊 Visual Insights",
        "🧭 Predict Visit Mode",
        "🎯 Recommend Attractions",
        "🧪 Model Performance",
    ],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Built on a real multi-table tourism dataset — 52,930 transactions, "
    "33,530 users, 30 rated attractions across Indonesia."
)
st.sidebar.caption("P Suman Sangeet · INNOVEXIS Data Science & Gen AI Internship")

master = lookup["master_sample"]

# ========================================================================
# PAGE 1 — OVERVIEW
# ========================================================================
if page == "🏝️ Overview":
    st.markdown(
        """
        <div class="hero">
            <div class="tag">Tourism Intelligence Platform</div>
            <h1>Tourism Experience Analytics</h1>
            <p>Turn raw visit history into personalized predictions and attraction
            recommendations — a single interactive system spanning classification,
            regression, and hybrid recommendation over 30 rated attractions across
            Bali, Yogyakarta, and Malang.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        (c1, "Transactions", f"{len(master):,}", "user-attraction visits"),
        (c2, "Unique Travelers", f"{master['UserId'].nunique():,}", "distinct users"),
        (c3, "Rated Attractions", f"{lookup['attraction_names'].shape[0]}", "with visit history"),
        (c4, "Avg. Rating", f"{lookup['global_avg_rating']:.2f} / 5", "across all visits"),
        (c5, "Countries Reached", f"{master['Country'].nunique()}", "traveler origins"),
    ]
    for col, label, value, sub in kpis:
        col.markdown(
            f"""<div class="kpi-card">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{value}</div>
                    <div class="kpi-sub">{sub}</div>
                </div>""",
            unsafe_allow_html=True,
        )

    st.write("")
    left, right = st.columns([1.3, 1])
    with left:
        st.markdown('<div class="section-label">What this app does</div>', unsafe_allow_html=True)
        st.markdown(
            """
- **Predict a traveler's visit mode** *(Business · Couples · Family · Friends · Solo)*
  from their location, travel dates, and the kind of attraction they're considering.
- **Recommend attractions** two ways — item-based collaborative filtering for
  returning travelers with rating history, and a content + popularity blend for
  new visitors with no history yet.
- **Explore the data** through recruiter-friendly visuals: popular attractions,
  top regions, visit-mode mix, seasonality, and behavioural user segments.
- **Compare model performance** across every regression and classification model
  trained, including feature importance and a confusion matrix.
            """
        )
    with right:
        st.markdown('<div class="section-label">How it\'s built</div>', unsafe_allow_html=True)
        st.markdown(
            """
            **Data:** 8-table relational tourism dataset (transactions, users,
            geography, attraction metadata) — cleaned, joined, and leakage-safely
            split before any target-derived features are computed.

            **Models:** Random Forest / XGBoost / Gradient Boosting, benchmarked
            and the best performer auto-selected for both prediction tasks.

            **Recommender:** hybrid — collaborative filtering (item-item cosine
            similarity) with a content-based + popularity fallback for cold-start
            visitors.
            """
        )

    st.write("")
    st.markdown('<div class="section-label">Snapshot</div>', unsafe_allow_html=True)
    mode_counts = master["VisitModeLabel"].value_counts().reset_index()
    mode_counts.columns = ["VisitMode", "Count"]
    top_attr = lookup["popularity"].head(6)
    sc1, sc2 = st.columns(2)
    with sc1:
        fig = px.pie(mode_counts, names="VisitMode", values="Count", hole=0.5,
                     title="Visit Mode Mix")
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(height=360, margin=dict(t=50, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)
    with sc2:
        fig = px.bar(top_attr, x="avg_rating", y="Attraction", orientation="h",
                     color="num_ratings", color_continuous_scale=["#EFE8DA", "#C97A3D", "#163A36"],
                     title="Top-Rated Attractions (color = review volume)")
        fig.update_layout(height=360, yaxis=dict(categoryorder="total ascending"),
                           margin=dict(t=50, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

# ========================================================================
# PAGE 2 — VISUAL INSIGHTS
# ========================================================================
elif page == "📊 Visual Insights":
    st.markdown("## 📊 Visual Insights")
    st.caption("Recruiter-friendly, interactive views into traveler behaviour and attraction demand.")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🏆 Popular Attractions", "🌍 Regions & Geography", "🧑‍🤝‍🧑 User Segments", "📈 Trends & Ratings"]
    )

    with tab1:
        col1, col2 = st.columns([1.4, 1])
        pop = lookup["popularity"]
        with col1:
            fig = px.bar(
                pop.sort_values("num_ratings", ascending=True).tail(15),
                x="num_ratings", y="Attraction", orientation="h",
                color="avg_rating", color_continuous_scale=["#EFE8DA", "#C97A3D", "#163A36"],
                title="Most-Visited Attractions (color = avg rating)",
                labels={"num_ratings": "Number of Visits", "avg_rating": "Avg Rating"},
            )
            fig.update_layout(height=520, margin=dict(t=50))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            cat_counts = master["AttractionCategory"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Visits"]
            fig = px.bar(cat_counts, x="Visits", y="Category", orientation="h",
                         color="Visits", color_continuous_scale=["#EFE8DA", "#1F5C55", "#163A36"],
                         title="Visits by Attraction Category")
            fig.update_layout(height=520, yaxis=dict(categoryorder="total ascending"),
                               margin=dict(t=50), showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            cont = master["Continent"].value_counts().reset_index()
            cont.columns = ["Continent", "Visits"]
            fig = px.bar(cont, x="Continent", y="Visits", color="Continent",
                         text="Visits", title="Traveler Volume by Continent")
            fig.update_traces(textposition="outside")
            fig.update_layout(height=420, showlegend=False, margin=dict(t=50))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            region_counts = master["Region"].value_counts().reset_index().head(10)
            region_counts.columns = ["Region", "Visits"]
            fig = px.bar(region_counts, x="Visits", y="Region", orientation="h",
                         color="Visits", color_continuous_scale=["#EFE8DA", "#C97A3D", "#A85A2A"],
                         title="Top 10 Origin Regions")
            fig.update_layout(height=420, yaxis=dict(categoryorder="total ascending"),
                               margin=dict(t=50), showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        cross = pd.crosstab(master["Region"], master["VisitModeLabel"])
        cross = cross.loc[cross.sum(axis=1).sort_values(ascending=False).head(10).index]
        fig = px.imshow(cross, text_auto=True, aspect="auto",
                         color_continuous_scale=["#F7F4EE", "#C97A3D", "#163A36"],
                         title="Visit-Mode Concentration Across Top 10 Regions")
        fig.update_layout(height=460, margin=dict(t=50))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        ub = lookup["user_behavior"]
        st.markdown(
            "Travelers are grouped by **visit frequency, average rating given, and category "
            "diversity** using KMeans clustering — a behavioural lens that complements geography."
        )
        seg_summary = (
            ub.groupby("SegmentName")
            .agg(Users=("UserId", "count"), AvgVisits=("num_visits", "mean"),
                 AvgRating=("avg_rating", "mean"), AvgCategoryDiversity=("num_categories", "mean"))
            .reset_index()
            .sort_values("Users", ascending=False)
        )
        col1, col2 = st.columns([1, 1.2])
        with col1:
            fig = px.pie(seg_summary, names="SegmentName", values="Users", hole=0.5,
                         title="Traveler Segments")
            fig.update_traces(textinfo="percent+label")
            fig.update_layout(height=420, margin=dict(t=50))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.scatter(
                ub.sample(min(4000, len(ub)), random_state=42),
                x="num_visits", y="avg_rating", color="SegmentName",
                size="num_categories", size_max=14, opacity=0.65,
                title="Segment Map — Visits vs. Avg Rating (bubble = category diversity)",
                labels={"num_visits": "Number of Visits", "avg_rating": "Average Rating Given"},
            )
            fig.update_layout(height=420, margin=dict(t=50))
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            seg_summary.round(2).rename(columns={
                "SegmentName": "Segment", "AvgVisits": "Avg. Visits",
                "AvgRating": "Avg. Rating", "AvgCategoryDiversity": "Avg. Category Diversity",
            }),
            use_container_width=True, hide_index=True,
        )

    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            trend = master.groupby(["VisitYear", "VisitMonth"]).size().reset_index(name="Visits")
            trend["Period"] = pd.to_datetime(
                trend["VisitYear"].astype(str) + "-" + trend["VisitMonth"].astype(str) + "-01"
            )
            trend = trend.sort_values("Period")
            fig = px.line(trend, x="Period", y="Visits", markers=True,
                         title="Monthly Visit Volume Over Time", color_discrete_sequence=["#163A36"])
            fig.update_layout(height=420, margin=dict(t=50))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.histogram(master, x="Rating", nbins=5, color="VisitModeLabel", barmode="group",
                               title="Rating Distribution by Visit Mode")
            fig.update_layout(height=420, margin=dict(t=50), bargap=0.15)
            st.plotly_chart(fig, use_container_width=True)

# ========================================================================
# PAGE 3 — PREDICT VISIT MODE
# ========================================================================
elif page == "🧭 Predict Visit Mode":
    st.markdown("## 🧭 Predict Visit Mode")
    st.caption(
        "Tell us about the trip — we'll predict whether this looks like a Business, "
        "Couples, Family, Friends, or Solo visit, using the best-performing classifier "
        f"(**{lookup['best_cls_name']}**)."
    )

    with st.form("predict_mode_form"):
        c1, c2 = st.columns(2)
        with c1:
            continent = st.selectbox("Continent", lookup["continent_list"], index=2)
            region_opts = lookup["region_by_continent"].get(continent, [])
            region = st.selectbox("Region", region_opts if region_opts else ["—"])
            country_opts = lookup["country_by_region"].get(region, [])
            country = st.selectbox("Country", country_opts if country_opts else ["—"])
        with c2:
            year = st.number_input("Visit Year", min_value=2018, max_value=2027, value=2025)
            month = st.slider("Visit Month", 1, 12, 6)
            category = st.selectbox("Attraction Category of Interest", lookup["category_list"])

        attraction_names = lookup["attraction_names"]
        attraction_category = lookup["attraction_category"]
        cat_attractions = [aid for aid, c in attraction_category.items() if c == category]
        attraction_choice = st.selectbox(
            "Specific attraction (optional — improves accuracy)",
            ["Not sure yet"] + [attraction_names[a] for a in cat_attractions],
        )
        submitted = st.form_submit_button("Predict Visit Mode", use_container_width=True)

    if submitted:
        attr_avg_train = lookup["attr_avg_train"]
        if attraction_choice != "Not sure yet":
            aid = [a for a in cat_attractions if attraction_names[a] == attraction_choice][0]
            attr_avg = attr_avg_train.get(aid, lookup["global_avg_rating"])
        else:
            same_cat_ids = attraction_category[attraction_category == category].index
            vals = [attr_avg_train.get(a) for a in same_cat_ids if a in attr_avg_train.index]
            attr_avg = float(np.mean(vals)) if vals else lookup["global_avg_rating"]

        pred_label, proba = predict_visit_mode(
            data, continent, region, country, category, int(year), int(month),
            attr_avg, lookup["global_avg_rating"],
        )
        est_rating = predict_rating(
            data, continent, region, country, category, int(year), int(month), attr_avg
        )

        st.markdown(
            f"""<div class="predict-result">
                    <div class="kpi-label">PREDICTED VISIT MODE</div>
                    <div class="big">🧳 {pred_label}</div>
                </div>""",
            unsafe_allow_html=True,
        )
        rc1, rc2 = st.columns([1.3, 1])
        with rc1:
            fig = px.bar(
                x=proba.values, y=proba.index, orientation="h",
                color=proba.values, color_continuous_scale=["#EFE8DA", "#C97A3D", "#163A36"],
                labels={"x": "Probability", "y": "Visit Mode"},
                title="Prediction Confidence Across All Visit Modes",
            )
            fig.update_layout(height=340, margin=dict(t=50), coloraxis_showscale=False,
                               yaxis=dict(categoryorder="total ascending"))
            st.plotly_chart(fig, use_container_width=True)
        with rc2:
            st.metric("Estimated Attraction Rating", f"{est_rating:.2f} / 5")
            st.metric("Model Confidence", f"{proba.iloc[0]*100:.1f}%")
            st.caption(
                "Rating estimate comes from the regression model trained on the same "
                "leakage-safe features (year, month, geography, category, attraction history)."
            )

# ========================================================================
# PAGE 4 — RECOMMEND ATTRACTIONS
# ========================================================================
elif page == "🎯 Recommend Attractions":
    st.markdown("## 🎯 Recommend Attractions")
    st.caption("Two recommendation paths — pick the one that matches your traveler.")

    mode = st.radio(
        "Who is this recommendation for?",
        ["🔁 Returning traveler (has visit history)", "🆕 New visitor (no history yet)"],
        horizontal=True,
    )

    if mode.startswith("🔁"):
        st.markdown('<div class="section-label">Returning traveler — collaborative filtering</div>',
                    unsafe_allow_html=True)
        user_ids = get_returning_user_ids(data)
        col1, col2 = st.columns([1, 2])
        with col1:
            user_id = st.selectbox(
                "Select a User ID (from travelers with rating history)",
                user_ids, index=0,
            )
            n = st.slider("How many recommendations?", 3, 10, 5)
        with col2:
            hist = get_user_history(data, user_id)
            st.markdown(f"**Visit history for User {user_id}** ({len(hist)} recorded visits)")
            if not hist.empty:
                st.dataframe(
                    hist[["Attraction", "AttractionCategory", "VisitModeLabel", "Rating", "VisitYear"]]
                    .rename(columns={"AttractionCategory": "Category", "VisitModeLabel": "Visit Mode"}),
                    use_container_width=True, hide_index=True, height=180,
                )

        recs = recommend_collaborative(data, user_id, n=n)
        st.write("")
        st.markdown("#### Recommended for this traveler")
        if recs.empty:
            st.info("No collaborative-filtering signal available for this user yet — try the New Visitor path instead.")
        else:
            cols = st.columns(min(3, len(recs)))
            for i, row in recs.iterrows():
                with cols[i % len(cols)]:
                    st.markdown(
                        f"""<div class="rec-card">
                                <span class="rec-badge">{row['MatchScore']:.0f}% match</span>
                                <div class="rec-title">{row['Attraction']}</div>
                                <div class="rec-meta">{row['Category']}</div>
                            </div>""",
                        unsafe_allow_html=True,
                    )

    else:
        st.markdown('<div class="section-label">New visitor — content + popularity blend</div>',
                    unsafe_allow_html=True)
        new_visitor_mode = st.radio(
            "How should we find recommendations?",
            ["By preferred category", "By an attraction you already have in mind"],
            horizontal=True, key="new_visitor_mode",
        )

        if new_visitor_mode == "By preferred category":
            col1, col2, col3 = st.columns(3)
            with col1:
                category = st.selectbox("Preferred category", ["Any"] + lookup["category_list"])
            with col2:
                pref_mode = st.selectbox("Likely visit mode", ["Any"] + lookup["visit_mode_list"])
            with col3:
                n = st.slider("How many recommendations?", 3, 10, 5, key="new_n")

            recs = recommend_for_new_visitor(data, category, n=n)
            st.write("")
            st.markdown("#### Recommended attractions")
            if recs.empty:
                st.info("No attractions matched — try a broader category.")
            else:
                cols = st.columns(min(3, len(recs)))
                for i, row in recs.iterrows():
                    with cols[i % len(cols)]:
                        st.markdown(
                            f"""<div class="rec-card">
                                    <span class="rec-badge">⭐ {row['avg_rating']:.2f}</span>
                                    <div class="rec-title">{row['Attraction']}</div>
                                    <div class="rec-meta">{row['AttractionCategory']} · {row['num_ratings']:,} visits</div>
                                </div>""",
                            unsafe_allow_html=True,
                        )
                st.caption(
                    "Ranked by category match, review volume, and average rating — a "
                    "content + popularity blend that avoids the cold-start problem "
                    "collaborative filtering has for brand-new travelers."
                )
        else:
            full_names = lookup["full_attraction_names"]
            name_to_id = {name: aid for aid, name in full_names.items()}
            col1, col2 = st.columns([2, 1])
            with col1:
                chosen_name = st.selectbox(
                    "Pick an attraction you already have in mind",
                    sorted(name_to_id.keys()),
                    index=0,
                )
            with col2:
                n = st.slider("How many recommendations?", 3, 10, 5, key="similar_n")

            aid = name_to_id[chosen_name]
            recs = recommend_similar_to(data, aid, n=n)
            st.write("")
            st.markdown(f"#### Attractions similar to *{chosen_name}*")
            if recs.empty:
                st.info("No similar attractions found in the catalog for this pick.")
            else:
                cols = st.columns(min(3, len(recs)))
                for i, row in recs.iterrows():
                    with cols[i % len(cols)]:
                        st.markdown(
                            f"""<div class="rec-card">
                                    <span class="rec-badge">{row['Similarity']:.0f}% similar</span>
                                    <div class="rec-title">{row['Attraction']}</div>
                                    <div class="rec-meta">{row['Category']}</div>
                                </div>""",
                            unsafe_allow_html=True,
                        )
                st.caption(
                    "Content-based similarity computed across the full 1,698-attraction "
                    "catalog (category + average rating profile) — works even for "
                    "attractions with little or no rating history."
                )

# ========================================================================
# PAGE 5 — MODEL PERFORMANCE
# ========================================================================
elif page == "🧪 Model Performance":
    st.markdown("## 🧪 Model Performance")
    st.caption("Every candidate model, benchmarked head-to-head, with the winner auto-selected for the live app.")

    tab1, tab2 = st.tabs(["📈 Regression — Rating Prediction", "🧭 Classification — Visit Mode"])

    with tab1:
        reg_df = lookup["reg_results_df"]
        st.dataframe(reg_df.style.format({"R2": "{:.3f}", "RMSE": "{:.3f}", "MAE": "{:.3f}"}),
                     use_container_width=True, hide_index=True)
        fig = px.bar(
            reg_df.melt(id_vars="Model", value_vars=["R2", "RMSE", "MAE"]),
            x="Model", y="value", color="variable", barmode="group",
            title="Regression Model Comparison",
        )
        fig.update_layout(height=420, margin=dict(t=50))
        st.plotly_chart(fig, use_container_width=True)
        st.success(f"**Best regressor: {lookup['best_reg_name']}** — auto-selected for live rating estimates.")
        st.caption(
            "R² in the 0.10–0.15 range is expected for review-style rating data: ratings "
            "cluster tightly around 4–5, leaving limited variance for any model to explain. "
            "Attraction reputation (AttractionAvgRating) is consistently the dominant signal."
        )

    with tab2:
        cls_df = lookup["cls_results_df"]
        st.dataframe(
            cls_df.style.format({"Accuracy": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"}),
            use_container_width=True, hide_index=True,
        )
        fig = px.bar(
            cls_df.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1"]),
            x="Model", y="value", color="variable", barmode="group",
            title="Classification Model Comparison",
        )
        fig.update_layout(height=420, margin=dict(t=50), yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
        st.success(f"**Best classifier: {lookup['best_cls_name']}** — auto-selected for live visit-mode predictions.")
        st.caption(
            "The dataset is imbalanced — Couples and Family visits vastly outnumber Business "
            "trips — so weighted F1 is the fairer scorecard here, and class-balancing during "
            "training helps the model pay attention to minority segments."
        )

    st.markdown("---")
    st.markdown('<div class="section-label">Feature importance</div>', unsafe_allow_html=True)
    fi1, fi2 = st.columns(2)
    reg_model = data["reg_model"]
    cls_model = data["cls_model"]
    reg_features = data["feature_lists"]["reg_features"]
    cls_features = data["feature_lists"]["cls_features"]
    with fi1:
        if hasattr(reg_model, "feature_importances_"):
            imp = pd.Series(reg_model.feature_importances_, index=reg_features).sort_values(ascending=False)
            fig = px.bar(imp, orientation="h", title=f"Feature Importance — {lookup['best_reg_name']} (Rating)")
            fig.update_layout(height=380, yaxis=dict(categoryorder="total ascending"), showlegend=False,
                               margin=dict(t=50))
            st.plotly_chart(fig, use_container_width=True)
    with fi2:
        if hasattr(cls_model, "feature_importances_"):
            imp = pd.Series(cls_model.feature_importances_, index=cls_features).sort_values(ascending=False)
            fig = px.bar(imp, orientation="h", title=f"Feature Importance — {lookup['best_cls_name']} (Visit Mode)")
            fig.update_layout(height=380, yaxis=dict(categoryorder="total ascending"), showlegend=False,
                               margin=dict(t=50))
            st.plotly_chart(fig, use_container_width=True)

elif page == "Model Performance":

    # Your existing Model Performance title/content
    st.title("Model Performance")

    # Your existing benchmark tables/metrics
    # ...
    # Regression Benchmark
    # Classification Benchmark
    # ...

    # ==========================================
    # FEATURE IMPORTANCE
    # ==========================================

    st.markdown("---")

    st.markdown(
        '<div class="section-label">Feature importance</div>',
        unsafe_allow_html=True
    )

    reg_model = data["reg_model"]
    cls_model = data["cls_model"]

    reg_features = data["feature_lists"]["reg_features"]
    cls_features = data["feature_lists"]["cls_features"]

    fi1, fi2 = st.columns(2)

    with fi1:
        if hasattr(reg_model, "feature_importances_"):
            imp = pd.Series(
                reg_model.feature_importances_,
                index=reg_features
            ).sort_values(ascending=False)

            fig = px.bar(
                imp,
                orientation="h",
                title=f"Feature Importance — {lookup['best_reg_name']} (Rating)"
            )

            fig.update_layout(
                height=380,
                yaxis=dict(categoryorder="total ascending"),
                showlegend=False,
                margin=dict(t=50)
            )

            st.plotly_chart(fig, use_container_width=True)

    with fi2:
        if hasattr(cls_model, "feature_importances_"):
            imp = pd.Series(
                cls_model.feature_importances_,
                index=cls_features
            ).sort_values(ascending=False)

            fig = px.bar(
                imp,
                orientation="h",
                title=f"Feature Importance — {lookup['best_cls_name']} (Visit Mode)"
            )

            fig.update_layout(
                height=380,
                yaxis=dict(categoryorder="total ascending"),
                showlegend=False,
                margin=dict(t=50)
            )

            st.plotly_chart(fig, use_container_width=True)