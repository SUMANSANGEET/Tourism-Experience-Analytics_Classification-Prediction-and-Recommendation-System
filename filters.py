"""Global interactive filter panel — applied across the analytics pages."""
import streamlit as st


def render_filter_panel(master):
    with st.sidebar.expander("🔎 Global Filters", expanded=False):
        continents = st.multiselect("Continent", sorted(master["Continent"].dropna().unique()))
        region_pool = master if not continents else master[master["Continent"].isin(continents)]
        regions = st.multiselect("Region", sorted(region_pool["Region"].dropna().unique()))
        country_pool = region_pool if not regions else region_pool[region_pool["Region"].isin(regions)]
        countries = st.multiselect("Country", sorted(country_pool["Country"].dropna().unique()))
        categories = st.multiselect("Attraction Category", sorted(master["AttractionCategory"].dropna().unique()))
        modes = st.multiselect("Visit Mode", sorted(master["VisitModeLabel"].dropna().unique()))
        years = st.multiselect("Visit Year", sorted(master["VisitYear"].dropna().unique(), reverse=True))
        min_rating = st.slider("Minimum Rating", 1, 5, 1)
        if st.button("Clear filters", width="stretch"):
            for k in ["gf_continents", "gf_regions", "gf_countries", "gf_categories", "gf_modes", "gf_years"]:
                st.session_state.pop(k, None)
            st.rerun()

    return {
        "continents": continents, "regions": regions, "countries": countries,
        "categories": categories, "modes": modes, "years": years, "min_rating": min_rating,
    }


def apply_filters(master, f):
    df = master
    if f["continents"]:
        df = df[df["Continent"].isin(f["continents"])]
    if f["regions"]:
        df = df[df["Region"].isin(f["regions"])]
    if f["countries"]:
        df = df[df["Country"].isin(f["countries"])]
    if f["categories"]:
        df = df[df["AttractionCategory"].isin(f["categories"])]
    if f["modes"]:
        df = df[df["VisitModeLabel"].isin(f["modes"])]
    if f["years"]:
        df = df[df["VisitYear"].isin(f["years"])]
    if f["min_rating"] > 1:
        df = df[df["Rating"] >= f["min_rating"]]
    return df


def active_filter_count(f):
    return sum(bool(v) for k, v in f.items() if k != "min_rating") + (1 if f["min_rating"] > 1 else 0)
