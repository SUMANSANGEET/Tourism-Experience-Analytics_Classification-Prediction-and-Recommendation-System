"""
Advanced analytics & feature layer for Tourism Experience Analytics.

Everything here is *additive* to utils.py: it reconstructs the exact
train/test split used to produce the shipped artifacts (using the saved
encoders, so it lines up with the already-trained models with zero
retraining), and builds every extra capability requested in the app-review:
deep model diagnostics, SHAP explainability, geographic aggregation,
automated insight generation, an AI trip planner, traveler 360 profiles,
recommendation-method comparison, and CSV/report export helpers.
"""
import os
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    r2_score, mean_squared_error, mean_absolute_error,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "Tourism Dataset")
RANDOM_STATE = 42


# ------------------------------------------------------------------------
# Reconstruct the exact train/test split used to produce the shipped
# artifacts (same cleaning, same encoders — loaded, not refit — same
# random_state/test_size/stratify), so every deep-diagnostic feature below
# runs against genuinely held-out data with zero retraining.
# ------------------------------------------------------------------------
@st.cache_resource(show_spinner="Reconstructing evaluation split for deep diagnostics...")
def reconstruct_split(_data):
    encoders = _data["encoders"]["features"]
    target_le = _data["encoders"]["target"]

    transaction = pd.read_excel(os.path.join(DATASET_DIR, "Transaction.xlsx"))
    user = pd.read_excel(os.path.join(DATASET_DIR, "User.xlsx"))
    city = pd.read_excel(os.path.join(DATASET_DIR, "City.xlsx"))
    country = pd.read_excel(os.path.join(DATASET_DIR, "Country.xlsx"))
    region = pd.read_excel(os.path.join(DATASET_DIR, "Region.xlsx"))
    continent = pd.read_excel(os.path.join(DATASET_DIR, "Continent.xlsx"))
    mode_lkp = pd.read_excel(os.path.join(DATASET_DIR, "Mode.xlsx"))
    type_lkp = pd.read_excel(os.path.join(DATASET_DIR, "Type.xlsx"))
    item = pd.read_excel(os.path.join(DATASET_DIR, "Item.xlsx"))

    city = city.copy()
    city["CityName"] = city["CityName"].fillna("Unknown")
    user = user.copy()
    user["CityId"] = user["CityId"].fillna(-1)

    type_map = dict(zip(type_lkp["AttractionTypeId"].astype(str), type_lkp["AttractionType"]))
    item = item.copy()
    item["AttractionCategory"] = (
        item["AttractionTypeId"].astype(str).map(type_map).fillna(item["AttractionTypeId"].astype(str))
    )

    mode_map = dict(zip(mode_lkp["VisitModeId"], mode_lkp["VisitMode"]))
    transaction = transaction.copy()
    transaction["VisitModeLabel"] = transaction["VisitMode"].map(mode_map)

    geo = (
        user.merge(city, on="CityId", how="left", suffixes=("", "_city"))
        .merge(country, on="CountryId", how="left", suffixes=("", "_country"))
        .merge(region, on="RegionId", how="left", suffixes=("", "_region"))
        .merge(continent, on="ContinentId", how="left", suffixes=("", "_continent"))
    )
    master = (
        transaction.merge(geo, on="UserId", how="left")
        .merge(item[["AttractionId", "AttractionCategory", "Attraction"]], on="AttractionId", how="left")
        .dropna(subset=["CityName"])
        .reset_index(drop=True)
    )

    for c in ["Continent", "Region", "Country", "AttractionCategory"]:
        master[c + "_enc"] = encoders[c].transform(master[c].astype(str))
    master["VisitModeTarget"] = target_le.transform(master["VisitModeLabel"])

    train_df, test_df = train_test_split(
        master, test_size=0.2, random_state=RANDOM_STATE, stratify=master["VisitModeTarget"]
    )
    attr_avg_train = train_df.groupby("AttractionId")["Rating"].mean()
    user_avg_train = train_df.groupby("UserId")["Rating"].mean()
    global_avg = train_df["Rating"].mean()
    for d in (train_df, test_df):
        d["AttractionAvgRating"] = d["AttractionId"].map(attr_avg_train).fillna(global_avg)
        d["UserAvgRating"] = d["UserId"].map(user_avg_train).fillna(global_avg)

    return {"master": master, "train_df": train_df, "test_df": test_df,
            "attr_avg_train": attr_avg_train, "user_avg_train": user_avg_train,
            "global_avg": global_avg}


@st.cache_resource(show_spinner="Computing deep model diagnostics...")
def deep_classification_diagnostics(_data, _split):
    cls_features = _data["feature_lists"]["cls_features"]
    target_le = _data["encoders"]["target"]
    test_df = _split["test_df"]
    X_test, y_test = test_df[cls_features], test_df["VisitModeTarget"]
    model = _data["cls_model"]

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)
    labels = target_le.classes_

    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, target_names=labels, output_dict=True, zero_division=0)

    # One-vs-rest ROC-AUC per class (multiclass)
    roc_data = {}
    y_test_arr = np.array(y_test)
    for i, label in enumerate(labels):
        y_bin = (y_test_arr == i).astype(int)
        try:
            fpr, tpr, _ = roc_curve(y_bin, proba[:, i])
            auc = roc_auc_score(y_bin, proba[:, i])
            roc_data[label] = {"fpr": fpr, "tpr": tpr, "auc": auc}
        except Exception:
            roc_data[label] = {"fpr": np.array([0, 1]), "tpr": np.array([0, 1]), "auc": float("nan")}

    return {"confusion_matrix": cm, "labels": labels, "report": report, "roc_data": roc_data,
            "y_test": y_test, "preds": preds, "proba": proba}


@st.cache_resource(show_spinner="Computing regression diagnostics...")
def deep_regression_diagnostics(_data, _split):
    reg_features = _data["feature_lists"]["reg_features"]
    test_df = _split["test_df"]
    X_test, y_test = test_df[reg_features], test_df["Rating"]
    model = _data["reg_model"]
    preds = model.predict(X_test)
    residuals = y_test.values - preds
    return {
        "y_test": y_test, "preds": preds, "residuals": residuals,
        "r2": r2_score(y_test, preds),
        "rmse": mean_squared_error(y_test, preds) ** 0.5,
        "mae": mean_absolute_error(y_test, preds),
    }


# ------------------------------------------------------------------------
# SHAP explainability
# ------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_shap_explainer(_model):
    import shap
    return shap.TreeExplainer(_model)


def shap_explain_row(explainer, X_row, class_names=None, predicted_class_idx=None):
    """Return a Series of SHAP contributions for a single-row DataFrame, for the
    predicted class if this is a multiclass classifier."""
    import shap
    shap_values = explainer.shap_values(X_row)
    if isinstance(shap_values, list):
        # older SHAP API: list of arrays, one per class
        vals = shap_values[predicted_class_idx][0]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # shap>=0.44 style: (n_samples, n_features, n_classes)
        vals = shap_values[0, :, predicted_class_idx]
    else:
        vals = shap_values[0]
    return pd.Series(vals, index=X_row.columns).sort_values(key=np.abs, ascending=False)


# ------------------------------------------------------------------------
# Confidence tiering
# ------------------------------------------------------------------------
def confidence_tier(prob):
    if prob >= 0.75:
        return "🟢 High confidence", "Strong signal from the model — safe to act on directly."
    elif prob >= 0.50:
        return "🟡 Moderate confidence", "Reasonable signal, but consider collecting more traveler detail."
    else:
        return "🔴 Low confidence", "Weak signal — treat this as a hint, not a certainty."


# ------------------------------------------------------------------------
# Geography aggregation (for the interactive map)
# ------------------------------------------------------------------------
def geo_aggregate(master, group_col="Country"):
    agg = (
        master.groupby(group_col)
        .agg(
            visits=("Rating", "count"),
            avg_rating=("Rating", "mean"),
            dominant_mode=("VisitModeLabel", lambda s: s.mode().iat[0] if not s.mode().empty else "—"),
        )
        .reset_index()
        .sort_values("visits", ascending=False)
    )
    return agg


# ------------------------------------------------------------------------
# "Why is this attraction popular?" breakdown
# ------------------------------------------------------------------------
def attraction_breakdown(master, attraction_name):
    sub = master[master["Attraction"] == attraction_name]
    if sub.empty:
        return {}
    return {
        "by_mode": sub["VisitModeLabel"].value_counts(),
        "by_country": sub["Country"].value_counts().head(8),
        "by_region": sub["Region"].value_counts().head(8),
        "by_month": sub.groupby("VisitMonth").size(),
        "rating_dist": sub["Rating"].value_counts().sort_index(),
        "n": len(sub),
        "avg_rating": sub["Rating"].mean(),
    }


# ------------------------------------------------------------------------
# Automated insight generator — every line computed live, nothing hardcoded
# ------------------------------------------------------------------------
def generate_insights(master, seg_summary=None):
    insights = []

    top_cat = master["AttractionCategory"].value_counts().idxmax()
    top_cat_share = master["AttractionCategory"].value_counts(normalize=True).max() * 100
    insights.append(("🔥", f"**{top_cat}** attractions draw the most visits "
                            f"({top_cat_share:.0f}% of all recorded visits)."))

    best_cat = master.groupby("AttractionCategory")["Rating"].mean().idxmax()
    best_cat_rating = master.groupby("AttractionCategory")["Rating"].mean().max()
    insights.append(("⭐", f"**{best_cat}** has the highest average satisfaction "
                            f"({best_cat_rating:.2f}/5)."))

    top_mode = master["VisitModeLabel"].value_counts().idxmax()
    top_mode_share = master["VisitModeLabel"].value_counts(normalize=True).max() * 100
    insights.append(("👥", f"**{top_mode}** is the dominant visitor segment "
                            f"({top_mode_share:.0f}% of visits)."))

    monthly = master.groupby("VisitMonth").size()
    peak_month = monthly.idxmax()
    trough_month = monthly.idxmin()
    peak_vs_avg = (monthly.max() / monthly.mean() - 1) * 100
    month_name = pd.Timestamp(2000, int(peak_month), 1).strftime("%B")
    insights.append(("📈", f"Demand peaks in **{month_name}**, "
                            f"{peak_vs_avg:.0f}% above the monthly average."))

    attr_stats = master.groupby("Attraction").agg(
        visits=("Rating", "count"), avg_rating=("Rating", "mean")
    )
    overall_avg = master["Rating"].mean()
    volume_threshold = attr_stats["visits"].quantile(0.75)
    underperformers = attr_stats[
        (attr_stats["visits"] >= volume_threshold) & (attr_stats["avg_rating"] < overall_avg)
    ].sort_values("visits", ascending=False)
    if not underperformers.empty:
        worst = underperformers.index[0]
        insights.append(("⚠️", f"**{worst}** has high traffic but below-average satisfaction "
                                f"({underperformers.loc[worst, 'avg_rating']:.2f} vs "
                                f"{overall_avg:.2f} overall) — a service-review candidate."))

    if seg_summary is not None and not seg_summary.empty:
        best_segment = seg_summary.sort_values("AvgRating", ascending=False).iloc[0]
        insights.append(("🎯", f"**{best_segment['Segment']}** travelers show the strongest "
                                f"engagement (avg rating {best_segment['AvgRating']:.2f}, "
                                f"{best_segment['AvgVisits']:.1f} visits/traveler) — the best "
                                f"target for personalized recommendation pushes."))

    return insights


# ------------------------------------------------------------------------
# Traveler 360 profile
# ------------------------------------------------------------------------
def traveler_profile(data, user_id):
    master = data["lookup"]["master_sample"]
    hist = master[master["UserId"] == user_id].sort_values(["VisitYear", "VisitMonth"], ascending=False)
    if hist.empty:
        return None
    ub = data["lookup"]["user_behavior"]
    seg_row = ub[ub["UserId"] == user_id]
    segment = seg_row["SegmentName"].iat[0] if not seg_row.empty else "Unclassified"
    return {
        "history": hist,
        "total_visits": len(hist),
        "avg_rating": hist["Rating"].mean(),
        "fav_category": hist["AttractionCategory"].mode().iat[0] if not hist["AttractionCategory"].mode().empty else "—",
        "fav_mode": hist["VisitModeLabel"].mode().iat[0] if not hist["VisitModeLabel"].mode().empty else "—",
        "countries_visited_from": hist["Country"].nunique(),
        "category_diversity": hist["AttractionCategory"].nunique(),
        "segment": segment,
        "mode_dist": hist["VisitModeLabel"].value_counts(),
        "category_dist": hist["AttractionCategory"].value_counts(),
        "rating_trend": hist.sort_values(["VisitYear", "VisitMonth"])[["VisitYear", "VisitMonth", "Rating"]],
    }


# ------------------------------------------------------------------------
# Recommendation method comparison
# ------------------------------------------------------------------------
def compare_recommendation_methods(data, user_id, category, n=3):
    from utils import recommend_collaborative, recommend_for_new_visitor
    rows = []
    collab = recommend_collaborative(data, user_id, n=n)
    for _, r in collab.iterrows():
        rows.append({"Method": "Collaborative Filtering", "Attraction": r["Attraction"], "Score": r["MatchScore"]})
    pop = recommend_for_new_visitor(data, category, n=n)
    for _, r in pop.iterrows():
        score = (r["avg_rating"] / 5) * 100
        rows.append({"Method": "Popularity + Category", "Attraction": r["Attraction"], "Score": round(score, 1)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------
# "Why recommended?" reasoning
# ------------------------------------------------------------------------
def why_recommended_collaborative(data, user_id, attraction_id):
    user_item = data["user_item"]
    item_sim = data["item_sim"]
    reasons = []
    if user_id in user_item.index:
        rated = user_item.loc[user_id]
        rated = rated[rated > 0]
        if attraction_id in item_sim.columns:
            sims = item_sim[attraction_id]
            top_similar = sims.loc[rated.index].sort_values(ascending=False).head(2)
            names = data["lookup"]["attraction_names"]
            for aid, sim in top_similar.items():
                if sim > 0.3:
                    reasons.append(f"Similar to **{names.get(aid, aid)}**, which you rated "
                                    f"{rated.get(aid, 0):.0f}/5 (similarity {sim*100:.0f}%)")
    cats = data["lookup"]["attraction_category"]
    pop = data["lookup"]["popularity"]
    row = pop[pop["AttractionId"] == attraction_id]
    if not row.empty:
        reasons.append(f"Category: **{row.iloc[0]['AttractionCategory']}**")
        reasons.append(f"Average rating **{row.iloc[0]['avg_rating']:.1f}/5** across "
                        f"{int(row.iloc[0]['num_ratings'])} visits")
    return reasons


def popularity_percentile(data, attraction_id):
    pop = data["lookup"]["popularity"]
    if attraction_id not in pop["AttractionId"].values:
        return None
    rank = (pop["num_ratings"] <= pop.loc[pop["AttractionId"] == attraction_id, "num_ratings"].iat[0]).mean()
    return rank * 100


def best_suited_for(data, attraction_id):
    master = data["lookup"]["master_sample"]
    name = data["lookup"]["attraction_names"].get(attraction_id)
    if name is None:
        return "—"
    sub = master[master["Attraction"] == name]
    if sub.empty:
        return "—"
    return sub["VisitModeLabel"].mode().iat[0] if not sub["VisitModeLabel"].mode().empty else "—"


# ------------------------------------------------------------------------
# AI Trip Planner
# ------------------------------------------------------------------------
def plan_trip(data, category, visit_mode, num_days, attractions_per_day, min_rating=0.0):
    pop = data["lookup"]["popularity"].copy()
    if category and category != "Any":
        pop = pop[pop["AttractionCategory"] == category]
    if pop.empty:
        pop = data["lookup"]["popularity"].copy()
    pop = pop[pop["avg_rating"] >= min_rating]
    if pop.empty:
        return []

    if visit_mode and visit_mode != "Any":
        master = data["lookup"]["master_sample"]
        mode_attractions = set(master.loc[master["VisitModeLabel"] == visit_mode, "Attraction"])
        boosted = pop[pop["Attraction"].isin(mode_attractions)]
        if len(boosted) >= attractions_per_day:
            pop = pd.concat([boosted, pop[~pop["Attraction"].isin(mode_attractions)]]).drop_duplicates("Attraction")

    pop = pop.sort_values(["num_ratings", "avg_rating"], ascending=False)
    total_needed = num_days * attractions_per_day
    chosen = pop.head(total_needed).reset_index(drop=True)

    itinerary = []
    for d in range(num_days):
        day_slice = chosen.iloc[d * attractions_per_day: (d + 1) * attractions_per_day]
        itinerary.append(day_slice)
    return itinerary


# ------------------------------------------------------------------------
# Business strategy per segment
# ------------------------------------------------------------------------
SEGMENT_STRATEGY = {
    "Power Travelers": "High-value loyalty & early-access perks — protect this segment's satisfaction closely.",
    "Engaged Explorers": "Cross-category recommendations — they already show category diversity, "
                          "so introduce adjacent categories they haven't tried.",
    "Casual Visitors": "Popular-attraction promotions and simple, low-friction offers.",
    "Steady Regulars": "Personalized discovery nudges — reward consistency with tailored picks.",
    "Infrequent / New": "Onboarding nudges and welcome offers to drive a second visit.",
}


def segment_strategy(segment_name):
    return SEGMENT_STRATEGY.get(segment_name, "Monitor engagement and personalize as more data accrues.")


# ------------------------------------------------------------------------
# CSV / report download helpers
# ------------------------------------------------------------------------
def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def build_executive_report_html(master, lookup, insights, seg_summary):
    rows = "".join(f"<li>{icon} {text}</li>" for icon, text in insights)
    seg_rows = "".join(
        f"<tr><td>{r['Segment']}</td><td>{r['Users']}</td><td>{r['AvgVisits']:.1f}</td>"
        f"<td>{r['AvgRating']:.2f}</td></tr>"
        for _, r in seg_summary.iterrows()
    ) if seg_summary is not None else ""
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
    <title>Tourism Experience Analytics — Executive Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; color: #16302E; max-width: 780px; margin: 40px auto; }}
        h1 {{ color: #163A36; border-bottom: 3px solid #C97A3D; padding-bottom: 8px; }}
        h2 {{ color: #1F5C55; margin-top: 32px; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
        th, td {{ border: 1px solid #DDD3BE; padding: 8px 12px; text-align: left; }}
        th {{ background: #EFE8DA; }}
        .kpi {{ display: inline-block; background: #F7F4EE; border-left: 4px solid #C97A3D;
                padding: 10px 16px; margin: 6px 10px 6px 0; border-radius: 6px; }}
        .kpi b {{ display: block; font-size: 1.3rem; color: #163A36; }}
    </style></head><body>
    <h1>🌋 Tourism Experience Analytics — Executive Report</h1>
    <p>Generated from {len(master):,} transactions across
    {master['UserId'].nunique():,} travelers and {master['Attraction'].nunique()} attractions.</p>

    <h2>Key Metrics</h2>
    <div class="kpi"><b>{len(master):,}</b>Transactions</div>
    <div class="kpi"><b>{master['UserId'].nunique():,}</b>Travelers</div>
    <div class="kpi"><b>{master['Rating'].mean():.2f}/5</b>Avg. Rating</div>
    <div class="kpi"><b>{master['Country'].nunique()}</b>Countries Reached</div>

    <h2>Automated Insights</h2>
    <ul>{rows}</ul>

    <h2>Traveler Segments</h2>
    <table><tr><th>Segment</th><th>Users</th><th>Avg Visits</th><th>Avg Rating</th></tr>{seg_rows}</table>
    </body></html>"""
    return html.encode("utf-8")
