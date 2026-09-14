"""Shared data-loading, prediction, and recommendation helpers for the
Tourism Experience Analytics Streamlit app. Everything here is cached so
the (already-trained) models and similarity matrices load once per
session, not once per interaction.
"""

import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from lightgbm import Booster
from lightgbm import LGBMClassifier

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")


def _load(name):
    with open(os.path.join(ARTIFACT_DIR, name), "rb") as f:
        return pickle.load(f)

def _load_lgb_classifier(name):
    return _load(name)


@st.cache_resource(show_spinner=False)
def load_all():
    """Load every trained artifact once and hand back a single namespace dict."""
    data = {
        "reg_model": _load("best_regressor.pkl"),
        "cls_model": _load_lgb_classifier("best_classifier.pkl"),
        "encoders": _load("encoders.pkl"),
        "item_sim": _load("item_similarity.pkl"),
        "content_sim": _load("content_similarity.pkl"),
        "user_item": _load("user_item_matrix.pkl"),
        "lookup": _load("lookup_tables.pkl"),
        "kmeans": _load("kmeans_segmenter.pkl"),
        "seg_scaler": _load("segment_scaler.pkl"),
        "seg_name_map": _load("segment_name_map.pkl"),
        "feature_lists": _load("feature_lists.pkl"),
    }

    return data


# ---------------------------------------------------------------------
# Encoding helpers (gracefully fall back to the most frequent class for
# any category never seen during training)
# ---------------------------------------------------------------------
def safe_encode(le, value):
    classes = list(le.classes_)
    if value in classes:
        return int(le.transform([value])[0])
    return 0  # fallback bucket


def encode_row(encoders, continent, region, country, category):
    return {
        "Continent_enc": safe_encode(encoders["Continent"], continent),
        "Region_enc": safe_encode(encoders["Region"], region),
        "Country_enc": safe_encode(encoders["Country"], country),
        "AttractionCategory_enc": safe_encode(encoders["AttractionCategory"], category),
    }


# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------
def predict_visit_mode(data, continent, region, country, category, year, month,
                        attraction_avg_rating, user_avg_rating):
    enc = encode_row(data["encoders"]["features"], continent, region, country, category)
    feats = data["feature_lists"]["cls_features"]
    row = {
        "VisitYear": year, "VisitMonth": month,
        "Continent_enc": enc["Continent_enc"], "Region_enc": enc["Region_enc"],
        "Country_enc": enc["Country_enc"], "AttractionCategory_enc": enc["AttractionCategory_enc"],
        "UserAvgRating": user_avg_rating, "AttractionAvgRating": attraction_avg_rating,
    }
    X = pd.DataFrame([row])[feats]
    model = data["cls_model"]
    pred_idx = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    target_le = data["encoders"]["target"]
    labels = target_le.classes_
    pred_label = target_le.inverse_transform([pred_idx])[0]
    proba_series = pd.Series(proba, index=labels).sort_values(ascending=False)
    return pred_label, proba_series


def predict_rating(data, continent, region, country, category, year, month, attraction_avg_rating):
    enc = encode_row(data["encoders"]["features"], continent, region, country, category)
    feats = data["feature_lists"]["reg_features"]
    row = {
        "VisitYear": year, "VisitMonth": month,
        "Continent_enc": enc["Continent_enc"], "Region_enc": enc["Region_enc"],
        "Country_enc": enc["Country_enc"], "AttractionCategory_enc": enc["AttractionCategory_enc"],
        "AttractionAvgRating": attraction_avg_rating,
    }
    X = pd.DataFrame([row])[feats]
    pred = data["reg_model"].predict(X)[0]
    return float(np.clip(pred, 1, 5))


# ---------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------
def recommend_collaborative(data, user_id, n=5, exclude_rated=True):
    user_item = data["user_item"]
    item_sim = data["item_sim"]
    if user_id not in user_item.index:
        return pd.DataFrame()
    rated = user_item.loc[user_id]
    rated = rated[rated > 0]
    if rated.empty:
        return pd.DataFrame()
    scores = pd.Series(0.0, index=user_item.columns)
    for attraction_id, rating in rated.items():
        scores = scores.add(item_sim[attraction_id] * rating, fill_value=0)
    if exclude_rated:
        scores = scores.drop(index=rated.index, errors="ignore")
    scores = scores.sort_values(ascending=False).head(n)
    names = data["lookup"]["attraction_names"]
    cats = data["lookup"]["attraction_category"]
    out = pd.DataFrame({
        "AttractionId": scores.index,
        "Attraction": [names.get(i, f"Attraction {i}") for i in scores.index],
        "Category": [cats.get(i, "—") for i in scores.index],
        "MatchScore": scores.values,
    })
    if not out.empty:
        out["MatchScore"] = (out["MatchScore"] / out["MatchScore"].max() * 100).round(1)
    return out.reset_index(drop=True)


def recommend_similar_to(data, attraction_id, n=5):
    """Content-based: find attractions similar to one the visitor already has
    in mind, searched across the full 1,698-attraction catalog (handles
    cold-start attractions with no rating history)."""
    content_sim = data["content_sim"]
    if attraction_id not in content_sim.index:
        return pd.DataFrame()
    scores = content_sim[attraction_id].drop(index=attraction_id, errors="ignore")
    scores = scores.sort_values(ascending=False).head(n)
    names = data["lookup"]["full_attraction_names"]
    cats = data["lookup"]["full_attraction_category"]
    out = pd.DataFrame({
        "AttractionId": scores.index,
        "Attraction": [names.get(i, f"Attraction {i}") for i in scores.index],
        "Category": [cats.get(i, "—") for i in scores.index],
        "Similarity": (scores.values * 100).round(1),
    })
    return out.reset_index(drop=True)


def recommend_for_new_visitor(data, category, region=None, n=5):
    """Cold-start recommendation: blend category match with overall popularity."""
    popularity = data["lookup"]["popularity"].copy()
    if category and category != "Any":
        matched = popularity[popularity["AttractionCategory"] == category]
        if matched.empty:
            matched = popularity
    else:
        matched = popularity
    matched = matched.sort_values(["num_ratings", "avg_rating"], ascending=False).head(n)
    return matched.reset_index(drop=True)


def get_user_history(data, user_id):
    master_sample = data["lookup"]["master_sample"]
    hist = master_sample[master_sample["UserId"] == user_id]
    return hist.sort_values(["VisitYear", "VisitMonth"], ascending=False)


def get_returning_user_ids(data, limit=500):
    return sorted(data["user_item"].index.tolist())[:limit]