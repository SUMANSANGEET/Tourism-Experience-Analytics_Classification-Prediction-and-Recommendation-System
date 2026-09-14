import pandas as pd
import plotly.express as px
import streamlit as st

import advanced as A
from utils import (
    get_returning_user_ids, get_user_history, recommend_collaborative,
    recommend_for_new_visitor, recommend_similar_to,
)


def _rich_card(row, data, user_id=None, extra_badge=None):
    aid = row.get("AttractionId")
    pct = A.popularity_percentile(data, aid) if aid is not None else None
    suited = A.best_suited_for(data, aid) if aid is not None else "—"
    badge = extra_badge if extra_badge is not None else ""
    lines = [f"""<div class="rec-card">{badge}
                 <div class="rec-title">{row.get('Attraction', '—')}</div>
                 <div class="rec-meta">{row.get('Category', row.get('AttractionCategory', '—'))}</div>"""]
    if "avg_rating" in row:
        lines.append(f"<div class='rec-meta'>⭐ {row['avg_rating']:.2f} · 👥 {int(row.get('num_ratings', 0)):,} visits</div>")
    if suited and suited != "—":
        lines.append(f"<div class='rec-meta'>👨‍👩‍👧 Best suited for: {suited}</div>")
    if pct is not None:
        lines.append(f"<div class='rec-meta'>📊 {pct:.0f}th popularity percentile</div>")
    lines.append("</div>")
    st.markdown("".join(lines), unsafe_allow_html=True)

    if user_id is not None and aid is not None:
        with st.expander("🎯 Why this recommendation?"):
            reasons = A.why_recommended_collaborative(data, user_id, aid)
            if reasons:
                for r in reasons:
                    st.markdown(f"✓ {r}")
            else:
                st.caption("Popularity and category match drove this pick.")


def render(data):
    lookup = data["lookup"]
    st.markdown("## 🎯 Recommend Attractions")
    st.caption("Three recommendation paths — pick the one that matches your traveler, and compare methods below.")

    mode = st.radio(
        "Who is this recommendation for?",
        ["🔁 Returning traveler (has visit history)", "🆕 New visitor (no history yet)"],
        horizontal=True,
    )

    last_user_id, last_category = None, None

    if mode.startswith("🔁"):
        st.markdown('<div class="section-label">Returning traveler — collaborative filtering</div>', unsafe_allow_html=True)
        user_ids = get_returning_user_ids(data)
        col1, col2 = st.columns([1, 2])
        with col1:
            user_id = st.selectbox("Select a User ID (from travelers with rating history)", user_ids, index=0)
            n = st.slider("How many recommendations?", 3, 10, 5)
        with col2:
            hist = get_user_history(data, user_id)
            st.markdown(f"**Visit history for User {user_id}** ({len(hist)} recorded visits)")
            if not hist.empty:
                st.dataframe(
                    hist[["Attraction", "AttractionCategory", "VisitModeLabel", "Rating", "VisitYear"]]
                    .rename(columns={"AttractionCategory": "Category", "VisitModeLabel": "Visit Mode"}),
                    width="stretch", hide_index=True, height=180,
                )
        last_user_id = user_id

        recs = recommend_collaborative(data, user_id, n=n)
        st.write("")
        st.markdown("#### Recommended for this traveler")
        if recs.empty:
            st.info("No collaborative-filtering signal available for this user yet — try the New Visitor path instead.")
        else:
            cols = st.columns(min(3, len(recs)))
            for i, row in recs.iterrows():
                with cols[i % len(cols)]:
                    badge = f'<span class="rec-badge">{row["MatchScore"]:.0f}% match</span>'
                    _rich_card(row, data, user_id=user_id, extra_badge=badge)
            st.download_button("⬇️ Download these recommendations (CSV)", A.to_csv_bytes(recs),
                                file_name=f"recommendations_user_{user_id}.csv", mime="text/csv")

    else:
        st.markdown('<div class="section-label">New visitor — content + popularity blend</div>', unsafe_allow_html=True)
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
            last_category = category

            recs = recommend_for_new_visitor(data, category, n=n)
            st.write("")
            st.markdown("#### Recommended attractions")
            if recs.empty:
                st.info("No attractions matched — try a broader category.")
            else:
                cols = st.columns(min(3, len(recs)))
                for i, row in recs.iterrows():
                    with cols[i % len(cols)]:
                        badge = f'<span class="rec-badge">⭐ {row["avg_rating"]:.2f}</span>'
                        _rich_card(row, data, user_id=None, extra_badge=badge)
                st.caption(
                    "Ranked by category match, review volume, and average rating — a content + popularity "
                    "blend that avoids the cold-start problem collaborative filtering has for new travelers."
                )
                st.download_button("⬇️ Download these recommendations (CSV)", A.to_csv_bytes(recs),
                                    file_name="recommendations_new_visitor.csv", mime="text/csv")
        else:
            full_names = lookup["full_attraction_names"]
            name_to_id = {name: aid for aid, name in full_names.items()}
            col1, col2 = st.columns([2, 1])
            with col1:
                chosen_name = st.selectbox("Pick an attraction you already have in mind",
                                            sorted(name_to_id.keys()), index=0)
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
                        badge = f'<span class="rec-badge">{row["Similarity"]:.0f}% similar</span>'
                        _rich_card(row, data, user_id=None, extra_badge=badge)
                st.caption(
                    "Content-based similarity computed across the full 1,698-attraction catalog "
                    "(category + average rating profile) — works even for attractions with little "
                    "or no rating history."
                )

    st.markdown("---")
    st.markdown("### ⚖️ Recommendation Method Comparison")
    st.caption("Same traveler/category, three methods, side by side.")
    cmp_user = last_user_id if last_user_id is not None else get_returning_user_ids(data)[0]
    cmp_category = last_category if last_category is not None else lookup["category_list"][0]
    comp_df = A.compare_recommendation_methods(data, cmp_user, cmp_category, n=3)
    if not comp_df.empty:
        fig = px.bar(comp_df, x="Score", y="Attraction", color="Method", orientation="h", barmode="group",
                     title=f"Collaborative vs. Popularity+Category — Traveler {cmp_user}, Category '{cmp_category}'")
        fig.update_layout(height=380, margin=dict(t=50), yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig, width="stretch")
        best_row = comp_df.sort_values("Score", ascending=False).iloc[0]
        st.success(f"**Final hybrid pick:** {best_row['Attraction']} (via {best_row['Method']}, "
                   f"score {best_row['Score']:.0f}%)")
    else:
        st.info("Not enough data to compare methods for this traveler/category combination.")
