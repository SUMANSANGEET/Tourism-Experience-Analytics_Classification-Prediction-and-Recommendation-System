"""
Tourism Experience Analytics — Training Pipeline
==================================================
Loads the raw multi-table tourism dataset, cleans & joins it, engineers
leakage-safe features, trains the regression (rating) and classification
(visit-mode) models, builds a hybrid recommendation engine (item-based
collaborative filtering + content-based fallback), runs a KMeans user
segmentation, and serializes every artifact the Streamlit app needs.

Run once with:  python train_pipeline.py
Produces:        ./artifacts/*.pkl  +  ./artifacts/lookup_tables.pkl
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    import xgboost as xgb

    HAS_XGB = True
except Exception:
    HAS_XGB = False

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "Tourism Dataset")
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)


def save(obj, name):
    with open(os.path.join(ARTIFACT_DIR, name), "wb") as f:
        pickle.dump(obj, f)
    print(f"  saved artifacts/{name}")


# ----------------------------------------------------------------------
# 1. LOAD
# ----------------------------------------------------------------------
print("1) Loading raw tables ...")
transaction = pd.read_excel(os.path.join(DATA_DIR, "Transaction.xlsx"))
user = pd.read_excel(os.path.join(DATA_DIR, "User.xlsx"))
city = pd.read_excel(os.path.join(DATA_DIR, "City.xlsx"))
country = pd.read_excel(os.path.join(DATA_DIR, "Country.xlsx"))
region = pd.read_excel(os.path.join(DATA_DIR, "Region.xlsx"))
continent = pd.read_excel(os.path.join(DATA_DIR, "Continent.xlsx"))
mode_lkp = pd.read_excel(os.path.join(DATA_DIR, "Mode.xlsx"))
type_lkp = pd.read_excel(os.path.join(DATA_DIR, "Type.xlsx"))
item = pd.read_excel(os.path.join(DATA_DIR, "Item.xlsx"))  # 30 transacted attractions
updated_item = pd.read_excel(os.path.join(DATA_DIR, "Updated_Item.xlsx"))  # full 1698-attraction catalog

print(f"   transaction={transaction.shape} user={user.shape} item={item.shape} "
      f"updated_item={updated_item.shape}")

# ----------------------------------------------------------------------
# 2. CLEAN
# ----------------------------------------------------------------------
print("2) Cleaning ...")
city["CityName"] = city["CityName"].fillna("Unknown")
user["CityId"] = user["CityId"].fillna(-1)

# AttractionTypeId is inconsistently encoded (numeric id OR the category name itself)
type_map = dict(zip(type_lkp["AttractionTypeId"].astype(str), type_lkp["AttractionType"]))


def clean_category(df):
    df = df.copy()
    df["AttractionCategory"] = (
        df["AttractionTypeId"].astype(str).map(type_map).fillna(df["AttractionTypeId"].astype(str))
    )
    return df


item = clean_category(item)
updated_item = clean_category(updated_item)

mode_map = dict(zip(mode_lkp["VisitModeId"], mode_lkp["VisitMode"]))
transaction["VisitModeLabel"] = transaction["VisitMode"].map(mode_map)

assert transaction["Rating"].between(1, 5).all()
assert transaction["VisitModeLabel"].isnull().sum() == 0

# ----------------------------------------------------------------------
# 3. FEATURE ENGINEERING — MASTER TABLE
# ----------------------------------------------------------------------
print("3) Building master dataset ...")
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
print(f"   master shape = {master.shape}")

# ----------------------------------------------------------------------
# 4. ENCODE + LEAKAGE-SAFE SPLIT
# ----------------------------------------------------------------------
print("4) Encoding + splitting ...")
cat_cols = ["Continent", "Region", "Country", "AttractionCategory"]
encoders = {}
for c in cat_cols:
    le = LabelEncoder()
    master[c + "_enc"] = le.fit_transform(master[c].astype(str))
    encoders[c] = le

target_le = LabelEncoder()
master["VisitModeTarget"] = target_le.fit_transform(master["VisitModeLabel"])

train_df, test_df = train_test_split(
    master, test_size=0.2, random_state=RANDOM_STATE, stratify=master["VisitModeTarget"]
)

attr_avg_train = train_df.groupby("AttractionId")["Rating"].mean()
user_avg_train = train_df.groupby("UserId")["Rating"].mean()
global_avg = train_df["Rating"].mean()

for d in (train_df, test_df):
    d["AttractionAvgRating"] = d["AttractionId"].map(attr_avg_train).fillna(global_avg)
    d["UserAvgRating"] = d["UserId"].map(user_avg_train).fillna(global_avg)

# Full-table versions (used later so every row, incl. cold-start, has these features)
master["AttractionAvgRating"] = master["AttractionId"].map(attr_avg_train).fillna(global_avg)
master["UserAvgRating"] = master["UserId"].map(user_avg_train).fillna(global_avg)

# ----------------------------------------------------------------------
# 5. REGRESSION — RATING PREDICTION
# ----------------------------------------------------------------------
print("5) Training regression models (rating prediction) ...")
reg_features = [
    "VisitYear", "VisitMonth", "Continent_enc", "Region_enc",
    "Country_enc", "AttractionCategory_enc", "AttractionAvgRating",
]
X_train_r, y_train_r = train_df[reg_features], train_df["Rating"]
X_test_r, y_test_r = test_df[reg_features], test_df["Rating"]

reg_models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1),
}
if HAS_XGB:
    reg_models["XGBoost"] = xgb.XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1, random_state=RANDOM_STATE, n_jobs=-1
    )

reg_rows, fitted_reg = [], {}
for name, model in reg_models.items():
    model.fit(X_train_r, y_train_r)
    preds = model.predict(X_test_r)
    reg_rows.append({
        "Model": name,
        "R2": r2_score(y_test_r, preds),
        "RMSE": mean_squared_error(y_test_r, preds) ** 0.5,
        "MAE": mean_absolute_error(y_test_r, preds),
    })
    fitted_reg[name] = model
reg_results_df = pd.DataFrame(reg_rows).sort_values("R2", ascending=False).reset_index(drop=True)
best_reg_name = reg_results_df.iloc[0]["Model"]
best_reg_model = fitted_reg[best_reg_name]
print(reg_results_df)
print(f"   best regressor = {best_reg_name}")

# ----------------------------------------------------------------------
# 6. CLASSIFICATION — VISIT MODE PREDICTION
# ----------------------------------------------------------------------
print("6) Training classification models (visit-mode prediction) ...")
cls_features = [
    "VisitYear", "VisitMonth", "Continent_enc", "Region_enc", "Country_enc",
    "AttractionCategory_enc", "UserAvgRating", "AttractionAvgRating",
]
X_train_c, y_train_c = train_df[cls_features], train_df["VisitModeTarget"]
X_test_c, y_test_c = test_df[cls_features], test_df["VisitModeTarget"]

cls_models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=250, max_depth=12, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=RANDOM_STATE),
}
if HAS_XGB:
    cls_models["XGBoost"] = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1, random_state=RANDOM_STATE,
        n_jobs=-1, eval_metric="mlogloss",
    )

cls_rows, fitted_cls = [], {}
for name, model in cls_models.items():
    model.fit(X_train_c, y_train_c)
    preds = model.predict(X_test_c)
    cls_rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test_c, preds),
        "Precision": precision_score(y_test_c, preds, average="weighted", zero_division=0),
        "Recall": recall_score(y_test_c, preds, average="weighted", zero_division=0),
        "F1": f1_score(y_test_c, preds, average="weighted", zero_division=0),
    })
    fitted_cls[name] = model
cls_results_df = pd.DataFrame(cls_rows).sort_values("F1", ascending=False).reset_index(drop=True)
best_cls_name = cls_results_df.iloc[0]["Model"]
best_cls_model = fitted_cls[best_cls_name]
print(cls_results_df)
print(f"   best classifier = {best_cls_name}")

# ----------------------------------------------------------------------
# 7. RECOMMENDATION ENGINE — item-based CF + content-based fallback
# ----------------------------------------------------------------------
print("7) Building recommendation engine ...")
user_item = train_df.pivot_table(index="UserId", columns="AttractionId", values="Rating", aggfunc="mean").fillna(0)
item_sim = cosine_similarity(user_item.T.values)
item_sim_df = pd.DataFrame(item_sim, index=user_item.columns, columns=user_item.columns)

attraction_names = item.drop_duplicates("AttractionId").set_index("AttractionId")["Attraction"]
attraction_category = item.drop_duplicates("AttractionId").set_index("AttractionId")["AttractionCategory"]

# content features over the FULL 1698-attraction catalog (handles cold start)
cat_le_full = LabelEncoder()
updated_item["AttractionCategory_enc"] = cat_le_full.fit_transform(updated_item["AttractionCategory"].astype(str))
attr_rating_full = master.groupby("AttractionId")["Rating"].mean()
content_features = updated_item.drop_duplicates("AttractionId").set_index("AttractionId")[
    ["AttractionCategory_enc"]
].copy()
content_features["AvgRating"] = content_features.index.map(attr_rating_full).fillna(global_avg)
content_scaled = StandardScaler().fit_transform(content_features.values)
content_sim = cosine_similarity(content_scaled)
content_sim_df = pd.DataFrame(content_sim, index=content_features.index, columns=content_features.index)

# popularity table used for cold-start / new-visitor recommendations
popularity = (
    master.groupby(["AttractionId", "Attraction", "AttractionCategory"])
    .agg(avg_rating=("Rating", "mean"), num_ratings=("Rating", "count"))
    .reset_index()
    .sort_values(["num_ratings", "avg_rating"], ascending=False)
)

# ----------------------------------------------------------------------
# 8. USER SEGMENTATION — KMeans on behavioural features
# ----------------------------------------------------------------------
print("8) Running user segmentation (KMeans) ...")
user_behavior = (
    master.groupby("UserId")
    .agg(
        num_visits=("TransactionId", "count"),
        avg_rating=("Rating", "mean"),
        num_categories=("AttractionCategory", "nunique"),
        fav_mode=("VisitModeLabel", lambda s: s.mode().iat[0] if not s.mode().empty else "Unknown"),
    )
    .reset_index()
)
seg_features = ["num_visits", "avg_rating", "num_categories"]
seg_scaler = StandardScaler()
seg_X = seg_scaler.fit_transform(user_behavior[seg_features])

best_k, best_score, best_model_km = 4, None, None
from sklearn.metrics import silhouette_score

for k in (3, 4, 5):
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(seg_X)
    try:
        score = silhouette_score(seg_X, labels)
    except Exception:
        score = -1
    if best_score is None or score > best_score:
        best_score, best_k, best_model_km = score, k, km

user_behavior["Segment"] = best_model_km.predict(seg_X)

# Human-readable segment naming, ranked by avg_rating * num_visits ("value")
seg_summary = user_behavior.groupby("Segment")[seg_features].mean()
seg_summary["value_score"] = seg_summary["num_visits"] * seg_summary["avg_rating"]
seg_order = seg_summary.sort_values("value_score", ascending=False).index.tolist()
base_names = ["Power Travelers", "Engaged Explorers", "Casual Visitors", "Steady Regulars", "Infrequent / New"]
# extend generically if KMeans ever picks more clusters than we have hand-written labels for
while len(base_names) < len(seg_order):
    base_names.append(f"Segment {len(base_names) + 1}")
seg_names_ranked = base_names[: len(seg_order)]
segment_name_map = dict(zip(seg_order, seg_names_ranked))
user_behavior["SegmentName"] = user_behavior["Segment"].map(segment_name_map)
print(user_behavior["SegmentName"].value_counts())
print(f"   best_k={best_k} silhouette={best_score:.3f}")

# ----------------------------------------------------------------------
# 9. LOOKUP TABLES for the Streamlit UI (dropdowns etc.)
# ----------------------------------------------------------------------
print("9) Assembling UI lookup tables ...")
continent_list = sorted(master["Continent"].dropna().unique().tolist())
region_by_continent = master.groupby("Continent")["Region"].unique().apply(lambda x: sorted(x.tolist())).to_dict()
country_by_region = master.groupby("Region")["Country"].unique().apply(lambda x: sorted(x.tolist())).to_dict()
category_list = sorted(item["AttractionCategory"].dropna().unique().tolist())
visit_mode_list = [m for m in target_le.classes_.tolist()]

full_attraction_names = updated_item.drop_duplicates("AttractionId").set_index("AttractionId")["Attraction"]
full_attraction_category = updated_item.drop_duplicates("AttractionId").set_index("AttractionId")["AttractionCategory"]

lookup_tables = {
    "continent_list": continent_list,
    "region_by_continent": region_by_continent,
    "country_by_region": country_by_region,
    "category_list": category_list,
    "visit_mode_list": visit_mode_list,
    "attraction_names": attraction_names,
    "attraction_category": attraction_category,
    "full_attraction_names": full_attraction_names,
    "full_attraction_category": full_attraction_category,
    "popularity": popularity,
    "reg_results_df": reg_results_df,
    "cls_results_df": cls_results_df,
    "best_reg_name": best_reg_name,
    "best_cls_name": best_cls_name,
    "global_avg_rating": global_avg,
    "master_sample": master[
        ["UserId", "Continent", "Region", "Country", "AttractionCategory", "Attraction",
         "VisitYear", "VisitMonth", "VisitModeLabel", "Rating"]
    ],
    "user_behavior": user_behavior,
    "attr_avg_train": attr_avg_train,
    "user_avg_train": user_avg_train,
}

# ----------------------------------------------------------------------
# 10. SAVE ARTIFACTS
# ----------------------------------------------------------------------
print("10) Saving artifacts ...")
save(best_reg_model, "best_regressor.pkl")
save(best_cls_model, "best_classifier.pkl")
save({"features": encoders, "target": target_le, "content_category": cat_le_full}, "encoders.pkl")
save(item_sim_df, "item_similarity.pkl")
save(content_sim_df, "content_similarity.pkl")
save(user_item, "user_item_matrix.pkl")
save(lookup_tables, "lookup_tables.pkl")
save(best_model_km, "kmeans_segmenter.pkl")
save(seg_scaler, "segment_scaler.pkl")
save(segment_name_map, "segment_name_map.pkl")
save({"reg_features": reg_features, "cls_features": cls_features, "seg_features": seg_features}, "feature_lists.pkl")

print("\nDONE. All artifacts written to:", os.path.abspath(ARTIFACT_DIR))
