import streamlit as st

import advanced as A


def render(data):
    lookup = data["lookup"]
    st.markdown("## 🗺️ AI Trip Planner")
    st.caption("Tell us your travel style — we'll lay out a day-by-day attraction itinerary.")

    c1, c2, c3 = st.columns(3)
    with c1:
        category = st.selectbox("Preferred category", ["Any"] + lookup["category_list"])
        visit_mode = st.selectbox("Travel mode", ["Any"] + lookup["visit_mode_list"])
    with c2:
        num_days = st.slider("Number of days", 1, 7, 3)
        attractions_per_day = st.slider("Attractions per day", 1, 5, 2)
    with c3:
        min_rating = st.slider("Minimum attraction rating", 1.0, 5.0, 3.5, 0.5)

    if st.button("Plan My Trip", type="primary", width="stretch"):
        itinerary = A.plan_trip(data, category, visit_mode, num_days, attractions_per_day, min_rating)
        if not itinerary or all(day.empty for day in itinerary):
            st.warning("Not enough attractions match these preferences — try lowering the minimum rating "
                       "or broadening the category.")
        else:
            st.markdown("### Your itinerary")
            all_rows = []
            for d, day_df in enumerate(itinerary, start=1):
                st.markdown(f"#### 📅 Day {d}")
                if day_df.empty:
                    st.caption("No further attractions matched — consider fewer days or a broader category.")
                    continue
                cols = st.columns(min(3, len(day_df)))
                for i, (_, row) in enumerate(day_df.iterrows()):
                    with cols[i % len(cols)]:
                        st.markdown(
                            f"""<div class="rec-card">
                                    <div class="rec-title">{row['Attraction']}</div>
                                    <div class="rec-meta">{row['AttractionCategory']}</div>
                                    <div class="rec-meta">⭐ {row['avg_rating']:.2f} · 👥 {int(row['num_ratings']):,} visits</div>
                                </div>""",
                            unsafe_allow_html=True,
                        )
                day_df = day_df.copy()
                day_df["Day"] = d
                all_rows.append(day_df)

            if all_rows:
                import pandas as pd
                full_plan = pd.concat(all_rows, ignore_index=True)
                avg_rating = full_plan["avg_rating"].mean()
                st.markdown("---")
                st.metric("Planned Attractions", len(full_plan))
                st.metric("Average Rating Across Itinerary", f"{avg_rating:.2f} / 5")
                st.download_button("⬇️ Download itinerary (CSV)", A.to_csv_bytes(full_plan),
                                    file_name="trip_itinerary.csv", mime="text/csv")
