import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import advanced as A

COUNTRY_NAME_FIXES = {"Russia": "Russia", "South Korea": "South Korea"}  # placeholder for future overrides


def render(data, filtered_master, filters_active):
    lookup = data["lookup"]
    m = filtered_master
    st.markdown("## 📊 Interactive Analytics")
    st.caption("Attraction demand, geography, trends, and customer behaviour — all reactive to the sidebar filters.")

    if m.empty:
        st.warning("No transactions match the current filters — widen your selection.")
        return

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🏆 Attraction Analytics", "🌍 Geography", "📈 Trends & Ratings", "🧑‍🤝‍🧑 Customer Behaviour"]
    )

    # ------------------------------------------------------------------
    with tab1:
        col1, col2 = st.columns([1.4, 1])
        pop = (
            m.groupby(["Attraction", "AttractionCategory"])
            .agg(num_ratings=("Rating", "count"), avg_rating=("Rating", "mean"))
            .reset_index()
        )
        with col1:
            fig = px.bar(
                pop.sort_values("num_ratings", ascending=True).tail(15),
                x="num_ratings", y="Attraction", orientation="h",
                color="avg_rating", color_continuous_scale=["#EFE8DA", "#C97A3D", "#163A36"],
                title="Most-Visited Attractions (color = avg rating)",
            )
            fig.update_layout(height=480, margin=dict(t=50))
            st.plotly_chart(fig, width="stretch")
        with col2:
            cat_counts = m["AttractionCategory"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Visits"]
            fig = px.bar(cat_counts, x="Visits", y="Category", orientation="h",
                         color="Visits", color_continuous_scale=["#EFE8DA", "#1F5C55", "#163A36"],
                         title="Visits by Category")
            fig.update_layout(height=480, yaxis=dict(categoryorder="total ascending"),
                               margin=dict(t=50), showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, width="stretch")

        st.markdown("---")
        st.markdown('<div class="section-label">Why is this attraction popular?</div>', unsafe_allow_html=True)
        chosen = st.selectbox("Pick an attraction to break down", sorted(pop["Attraction"].unique()))
        bd = A.attraction_breakdown(m, chosen)
        if bd:
            b1, b2, b3 = st.columns(3)
            with b1:
                fig = px.bar(bd["by_mode"], title="By Visit Mode", labels={"index": "", "value": "Visits"})
                fig.update_layout(height=300, showlegend=False, margin=dict(t=50))
                st.plotly_chart(fig, width="stretch")
            with b2:
                fig = px.bar(bd["by_country"], title="By Origin Country", labels={"index": "", "value": "Visits"})
                fig.update_layout(height=300, showlegend=False, margin=dict(t=50))
                st.plotly_chart(fig, width="stretch")
            with b3:
                month_df = bd["by_month"].rename("Visits").reset_index()
                month_df["MonthName"] = month_df["VisitMonth"].apply(lambda x: pd.Timestamp(2000, int(x), 1).strftime("%b"))
                fig = px.bar(month_df, x="MonthName", y="Visits", title="By Month")
                fig.update_layout(height=300, margin=dict(t=50))
                st.plotly_chart(fig, width="stretch")
            st.caption(f"**{chosen}** — {bd['n']:,} visits recorded, {bd['avg_rating']:.2f}/5 average rating. "
                       f"Top origin: **{bd['by_country'].index[0] if len(bd['by_country']) else '—'}** · "
                       f"Top mode: **{bd['by_mode'].index[0] if len(bd['by_mode']) else '—'}**.")

    # ------------------------------------------------------------------
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            cont = m["Continent"].value_counts().reset_index()
            cont.columns = ["Continent", "Visits"]
            fig = px.bar(cont, x="Continent", y="Visits", color="Continent", text="Visits",
                         title="Traveler Volume by Continent")
            fig.update_traces(textposition="outside")
            fig.update_layout(height=380, showlegend=False, margin=dict(t=50))
            st.plotly_chart(fig, width="stretch")
        with col2:
            region_counts = m["Region"].value_counts().reset_index().head(10)
            region_counts.columns = ["Region", "Visits"]
            fig = px.bar(region_counts, x="Visits", y="Region", orientation="h", color="Visits",
                         color_continuous_scale=["#EFE8DA", "#C97A3D", "#A85A2A"], title="Top 10 Origin Regions")
            fig.update_layout(height=380, yaxis=dict(categoryorder="total ascending"),
                               margin=dict(t=50), showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, width="stretch")

        st.markdown('<div class="section-label">Origin Country → Attraction Region</div>', unsafe_allow_html=True)
        geo = A.geo_aggregate(m, "Country")
        fig = px.choropleth(
            geo, locations="Country", locationmode="country names", color="visits",
            hover_name="Country", color_continuous_scale=["#EFE8DA", "#C97A3D", "#163A36"],
            hover_data={"avg_rating": ":.2f", "dominant_mode": True, "visits": ":,"},
            title="Visitor Origin Map — click a country to filter the table below",
        )
        fig.update_layout(height=460, margin=dict(t=50, b=0), geo=dict(showframe=False, showcoastlines=False))
        event = st.plotly_chart(fig, width="stretch", on_select="rerun", key="geo_map_select")

        clicked_country = None
        if event and event.get("selection", {}).get("points"):
            clicked_country = event["selection"]["points"][0].get("location")

        if clicked_country:
            st.success(f"Filtered to **{clicked_country}** — click empty map space or change filters to reset.")
            country_rows = m[m["Country"] == clicked_country]
        else:
            country_rows = geo

        st.dataframe(
            (country_rows if clicked_country is None else
             country_rows.groupby(["Country", "Region"]).agg(
                 visits=("Rating", "count"), avg_rating=("Rating", "mean")).reset_index())
            .round(2), width="stretch", hide_index=True, height=220,
        )

        cross = pd.crosstab(m["Region"], m["VisitModeLabel"])
        cross = cross.loc[cross.sum(axis=1).sort_values(ascending=False).head(10).index]
        fig = px.imshow(cross, text_auto=True, aspect="auto",
                         color_continuous_scale=["#F7F4EE", "#C97A3D", "#163A36"],
                         title="Visit-Mode Concentration Across Top 10 Regions")
        fig.update_layout(height=440, margin=dict(t=50))
        st.plotly_chart(fig, width="stretch")

    # ------------------------------------------------------------------
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            trend = m.groupby(["VisitYear", "VisitMonth"]).size().reset_index(name="Visits")
            trend["Period"] = pd.to_datetime(trend["VisitYear"].astype(str) + "-" + trend["VisitMonth"].astype(str) + "-01")
            trend = trend.sort_values("Period")
            fig = px.line(trend, x="Period", y="Visits", markers=True, title="Monthly Visit Volume",
                          color_discrete_sequence=["#163A36"])
            fig.update_layout(height=400, margin=dict(t=50))
            st.plotly_chart(fig, width="stretch")
        with col2:
            fig = px.histogram(m, x="Rating", nbins=5, color="VisitModeLabel", barmode="group",
                                title="Rating Distribution by Visit Mode")
            fig.update_layout(height=400, margin=dict(t=50), bargap=0.15)
            st.plotly_chart(fig, width="stretch")

        st.markdown('<div class="section-label">Customer Journey</div>', unsafe_allow_html=True)
        st.caption("Traveler origin → visit mode → attraction category → rating band, in one flow.")
        journey = m.copy()
        journey["RatingBand"] = journey["Rating"].apply(lambda r: "4–5 ★" if r >= 4 else ("3 ★" if r == 3 else "1–2 ★"))
        top_continents = journey["Continent"].value_counts().head(5).index
        journey = journey[journey["Continent"].isin(top_continents)]

        stages = ["Continent", "VisitModeLabel", "AttractionCategory", "RatingBand"]
        top_cats = journey["AttractionCategory"].value_counts().head(6).index
        journey = journey[journey["AttractionCategory"].isin(top_cats)]

        all_labels, label_lookup = [], {}
        for stage in stages:
            for val in journey[stage].dropna().unique():
                key = f"{stage}:{val}"
                if key not in label_lookup:
                    label_lookup[key] = len(all_labels)
                    all_labels.append(str(val))

        sources, targets, values = [], [], []
        for i in range(len(stages) - 1):
            s_col, t_col = stages[i], stages[i + 1]
            pair_counts = journey.groupby([s_col, t_col]).size().reset_index(name="n")
            for _, r in pair_counts.iterrows():
                sk, tk = f"{s_col}:{r[s_col]}", f"{t_col}:{r[t_col]}"
                if sk in label_lookup and tk in label_lookup:
                    sources.append(label_lookup[sk])
                    targets.append(label_lookup[tk])
                    values.append(r["n"])

        fig = go.Figure(go.Sankey(
            node=dict(label=all_labels, pad=12, thickness=14,
                      color=["#163A36", "#1F5C55", "#C97A3D", "#D9A441"] * (len(all_labels) // 4 + 1)),
            link=dict(source=sources, target=targets, value=values, color="rgba(199,122,61,0.35)"),
        ))
        fig.update_layout(title="Traveler Origin → Visit Mode → Category → Rating", height=520, margin=dict(t=50))
        st.plotly_chart(fig, width="stretch")

    # ------------------------------------------------------------------
    with tab4:
        ub_full = lookup.get("user_behavior")
        if ub_full is None:
            st.info("Segmentation data not available.")
        else:
            ub = ub_full[ub_full["UserId"].isin(m["UserId"].unique())]
            if ub.empty:
                st.info("No segmented travelers in the current filter selection.")
            else:
                st.markdown(
                    "Travelers are grouped by **visit frequency, average rating given, and category "
                    "diversity** using KMeans clustering."
                )
                seg_summary = (
                    ub.groupby("SegmentName")
                    .agg(Users=("UserId", "count"), AvgVisits=("num_visits", "mean"),
                         AvgRating=("avg_rating", "mean"), AvgCategoryDiversity=("num_categories", "mean"))
                    .reset_index().sort_values("Users", ascending=False)
                )
                col1, col2 = st.columns([1, 1.2])
                with col1:
                    fig = px.pie(seg_summary, names="SegmentName", values="Users", hole=0.5, title="Traveler Segments")
                    fig.update_traces(textinfo="percent+label")
                    fig.update_layout(height=420, margin=dict(t=50))
                    st.plotly_chart(fig, width="stretch")
                with col2:
                    fig = px.scatter(
                        ub.sample(min(4000, len(ub)), random_state=42),
                        x="num_visits", y="avg_rating", color="SegmentName", size="num_categories", size_max=14,
                        opacity=0.65, title="Segment Map — Visits vs. Avg Rating (bubble = category diversity)",
                    )
                    fig.update_layout(height=420, margin=dict(t=50))
                    st.plotly_chart(fig, width="stretch")

                seg_summary_display = seg_summary.round(2).rename(columns={
                    "SegmentName": "Segment", "AvgVisits": "Avg. Visits",
                    "AvgRating": "Avg. Rating", "AvgCategoryDiversity": "Avg. Category Diversity",
                })
                seg_summary_display["Recommended Strategy"] = seg_summary["SegmentName"].apply(A.segment_strategy)
                st.dataframe(seg_summary_display, width="stretch", hide_index=True)
                st.download_button("⬇️ Download segment summary (CSV)", A.to_csv_bytes(seg_summary_display),
                                    file_name="segment_summary.csv", mime="text/csv")
