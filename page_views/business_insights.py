import pandas as pd
import streamlit as st

import advanced as A


def render(data):
    lookup = data["lookup"]
    master = lookup["master_sample"]
    ub = lookup.get("user_behavior")

    st.markdown("## 💼 Business Insights")
    st.caption("Data → ML → Insight → Action — every line below is calculated live from the current dataset.")

    seg_summary = None
    if ub is not None:
        seg_summary = (
            ub.groupby("SegmentName")
            .agg(Users=("UserId", "count"), AvgVisits=("num_visits", "mean"), AvgRating=("avg_rating", "mean"))
            .reset_index().rename(columns={"SegmentName": "Segment"})
        )

    st.markdown("### 🧠 Automated Insights")
    insights = A.generate_insights(master, seg_summary)
    for icon, text in insights:
        st.markdown(f"""<div class="insight-box">{icon} {text}</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🎯 Segment-Level Strategy")
    if seg_summary is not None:
        seg_summary = seg_summary.copy()
        seg_summary["Recommended Strategy"] = seg_summary["Segment"].apply(A.segment_strategy)
        st.dataframe(seg_summary.round(2), width="stretch", hide_index=True)

    st.markdown("---")
    st.markdown("### 💼 Business Recommendations")

    recs = [
        ("📣 Marketing", "Focus campaigns on the highest-volume visitor segment and their favourite "
                          "attraction categories, where advertising spend converts most reliably."),
        ("🏛️ Attraction Management", "Prioritize attractions with high traffic but below-average ratings — "
                                       "these are large audiences whose experience can be improved fastest."),
        ("🎯 Personalization", "Use behavioural segments (visit frequency, rating pattern, category diversity) "
                                "to tailor which recommendation mode — collaborative vs. content-based — a "
                                "traveler sees first."),
        ("😊 Customer Experience", "Investigate attractions with high visit counts but rating scores trailing "
                                    "the dataset average; these are the biggest single lever on overall satisfaction."),
        ("🗓️ Tourism Planning", "Allocate staffing and inventory around the seasonal demand peak identified "
                                 "in the Trends tab, rather than spreading resources evenly across the year."),
    ]
    for title, text in recs:
        st.markdown(f"""<div class="insight-box"><b>{title}</b><br>{text}</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📥 Export")
    col1, col2 = st.columns(2)
    with col1:
        report_html = A.build_executive_report_html(master, lookup, insights, seg_summary)
        st.download_button("📥 Download Executive Report (HTML)", report_html,
                            file_name="tourism_executive_report.html", mime="text/html", width="stretch")
    with col2:
        pop = lookup["popularity"]
        st.download_button("⬇️ Download attraction ranking (CSV)", A.to_csv_bytes(pop),
                            file_name="attraction_ranking.csv", mime="text/csv", width="stretch")
