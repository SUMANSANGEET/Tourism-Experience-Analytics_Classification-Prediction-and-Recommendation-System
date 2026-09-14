import pandas as pd
import plotly.express as px
import streamlit as st

import advanced as A


def render(data, filtered_master, filters_active):
    lookup = data["lookup"]
    m = filtered_master

    st.markdown(
        """
        <div class="hero">
            <div class="tag">Tourism Intelligence Platform</div>
            <h1>Executive Dashboard</h1>
            <p>Live KPIs and automatically-derived business insights — every number below
            recalculates from whatever filters are active in the sidebar.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if m.empty:
        st.warning("No transactions match the current filters — widen your selection.")
        return

    if filters_active:
        st.info(f"Showing **{len(m):,}** transactions after filters "
                f"(of {len(lookup['master_sample']):,} total).")

    c1, c2, c3, c4, c5 = st.columns(5)
    top_attr_row = m.groupby("Attraction").size().sort_values(ascending=False)
    top_attraction = top_attr_row.index[0] if not top_attr_row.empty else "—"
    top_mode = m["VisitModeLabel"].mode().iat[0] if not m["VisitModeLabel"].mode().empty else "—"
    returning_share = None
    ub = lookup.get("user_behavior")
    if ub is not None:
        returning_users = set(ub.loc[ub["num_visits"] > 1, "UserId"])
        returning_share = m["UserId"].isin(returning_users).mean() * 100

    kpis = [
        (c1, "Transactions", f"{len(m):,}", "user-attraction visits"),
        (c2, "Unique Travelers", f"{m['UserId'].nunique():,}", "distinct users"),
        (c3, "Avg. Rating", f"{m['Rating'].mean():.2f} / 5", "across selection"),
        (c4, "Top Attraction", top_attraction[:22] + ("…" if len(top_attraction) > 22 else ""), "by visit volume"),
        (c5, "Dominant Mode", top_mode, "most common visit mode"),
    ]
    for col, label, value, sub in kpis:
        col.markdown(f"""<div class="kpi-card"><div class="kpi-label">{label}</div>
                     <div class="kpi-value" style="font-size:1.3rem;">{value}</div>
                     <div class="kpi-sub">{sub}</div></div>""", unsafe_allow_html=True)

    if returning_share is not None:
        st.caption(f"↩️ **{returning_share:.0f}%** of these transactions come from returning travelers "
                   f"(2+ recorded visits).")

    st.write("")
    st.markdown('<div class="section-label">Automated Business Insights</div>', unsafe_allow_html=True)
    seg_summary = None
    if ub is not None:
        seg_summary = (
            ub.groupby("SegmentName")
            .agg(Users=("UserId", "count"), AvgVisits=("num_visits", "mean"), AvgRating=("avg_rating", "mean"))
            .reset_index().rename(columns={"SegmentName": "Segment"})
        )
    insights = A.generate_insights(m, seg_summary)
    cols = st.columns(2)
    for i, (icon, text) in enumerate(insights):
        with cols[i % 2]:
            st.markdown(f"""<div class="insight-box">{icon} {text}</div>""", unsafe_allow_html=True)

    st.write("")
    left, right = st.columns(2)
    with left:
        mode_counts = m["VisitModeLabel"].value_counts().reset_index()
        mode_counts.columns = ["VisitMode", "Count"]
        fig = px.pie(mode_counts, names="VisitMode", values="Count", hole=0.5, title="Visit Mode Mix")
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(height=360, margin=dict(t=50, b=10, l=10, r=10))
        st.plotly_chart(fig, width="stretch")
    with right:
        top_attr = (
            m.groupby(["Attraction", "AttractionCategory"])
            .agg(avg_rating=("Rating", "mean"), num_ratings=("Rating", "count"))
            .reset_index().sort_values("num_ratings", ascending=False).head(8)
        )
        fig = px.bar(top_attr.sort_values("avg_rating"), x="avg_rating", y="Attraction", orientation="h",
                     color="num_ratings", color_continuous_scale=["#EFE8DA", "#C97A3D", "#163A36"],
                     title="Top Attractions in Current Selection (color = visit volume)")
        fig.update_layout(height=360, margin=dict(t=50, b=10, l=10, r=10))
        st.plotly_chart(fig, width="stretch")

    st.write("")
    st.markdown('<div class="section-label">Download</div>', unsafe_allow_html=True)
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button("⬇️ Download filtered transactions (CSV)", A.to_csv_bytes(m),
                            file_name="filtered_transactions.csv", mime="text/csv", width="stretch")
    with dl2:
        report_html = A.build_executive_report_html(m, lookup, insights, seg_summary)
        st.download_button("📥 Download Executive Report (HTML)", report_html,
                            file_name="tourism_executive_report.html", mime="text/html", width="stretch")
