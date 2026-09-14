import pandas as pd
import plotly.express as px
import streamlit as st

import advanced as A
from utils import get_returning_user_ids, recommend_collaborative


def render(data):
    st.markdown("## 👤 Traveler 360°")
    st.caption("A full behavioural profile for any returning traveler — history, ratings, segment, and preferences.")

    user_ids = get_returning_user_ids(data, limit=2000)
    user_id = st.selectbox("Select a Traveler ID", user_ids, index=0)

    profile = A.traveler_profile(data, user_id)
    if profile is None:
        st.info("No recorded history for this traveler.")
        return

    st.markdown(f"### Traveler {user_id} · *{profile['segment']}*")
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, label, value in zip(
        [c1, c2, c3, c4, c5],
        ["Total Visits", "Avg Rating", "Favourite Category", "Favourite Mode", "Category Diversity"],
        [profile["total_visits"], f"{profile['avg_rating']:.2f} / 5", profile["fav_category"],
         profile["fav_mode"], profile["category_diversity"]],
    ):
        col.markdown(f"""<div class="kpi-card"><div class="kpi-label">{label}</div>
                     <div class="kpi-value" style="font-size:1.2rem;">{value}</div></div>""",
                     unsafe_allow_html=True)

    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(profile["mode_dist"], names=profile["mode_dist"].index, values=profile["mode_dist"].values,
                     hole=0.5, title="Visit-Mode Distribution")
        fig.update_layout(height=340, margin=dict(t=50))
        st.plotly_chart(fig, width="stretch")
    with col2:
        fig = px.bar(profile["category_dist"], title="Favourite Categories",
                     labels={"index": "", "value": "Visits"})
        fig.update_layout(height=340, margin=dict(t=50), showlegend=False)
        st.plotly_chart(fig, width="stretch")

    st.markdown('<div class="section-label">Rating Behaviour Over Time</div>', unsafe_allow_html=True)
    trend = profile["rating_trend"].copy()
    if not trend.empty:
        trend["Period"] = pd.to_datetime(trend["VisitYear"].astype(str) + "-" + trend["VisitMonth"].astype(str) + "-01")
        fig = px.line(trend, x="Period", y="Rating", markers=True, title="Rating Given Over Time")
        fig.update_layout(height=320, margin=dict(t=50), yaxis_range=[0.5, 5.5])
        st.plotly_chart(fig, width="stretch")

    st.markdown('<div class="section-label">Visit History</div>', unsafe_allow_html=True)
    st.dataframe(
        profile["history"][["Attraction", "AttractionCategory", "VisitModeLabel", "Rating", "VisitYear", "VisitMonth"]]
        .rename(columns={"AttractionCategory": "Category", "VisitModeLabel": "Visit Mode"}),
        width="stretch", hide_index=True, height=220,
    )

    st.markdown('<div class="section-label">🎯 Personalized Recommendations</div>', unsafe_allow_html=True)
    recs = recommend_collaborative(data, user_id, n=5)
    if recs.empty:
        st.info("Not enough rating signal yet for personalized recommendations.")
    else:
        cols = st.columns(min(5, len(recs)))
        for i, row in recs.iterrows():
            with cols[i % len(cols)]:
                st.markdown(f"""<div class="rec-card"><span class="rec-badge">{row['MatchScore']:.0f}%</span>
                             <div class="rec-title">{row['Attraction']}</div>
                             <div class="rec-meta">{row['Category']}</div></div>""", unsafe_allow_html=True)

    st.download_button("⬇️ Download this traveler's history (CSV)", A.to_csv_bytes(profile["history"]),
                        file_name=f"traveler_{user_id}_history.csv", mime="text/csv")
