# 🌋 Tourism Experience Analytics — Streamlit App

An interactive tourism intelligence platform built on a real 52,930-transaction,
33,530-user, 8-table tourism dataset (Bali, Yogyakarta, and Malang attractions).
Originally a 5-page classification/regression/recommendation app; this version
expands it into a full 8-page tourism intelligence product — global filters,
automated insights, a traveler 360° profile, an AI trip planner, SHAP
explainability, and deep model diagnostics — while keeping every original
prediction and recommendation exactly as trained (no retraining involved).

## What's inside

| Page | What it does |
|---|---|
| 🏝️ **Executive Dashboard** | KPI cards + automated, dynamically-computed insight cards. Reacts to the global filter panel. |
| 📊 **Interactive Analytics** | Tabs: Attraction Analytics (with "why is this attraction popular?" breakdowns), Geography (choropleth map, click a country to filter), Trends & Ratings (+ a Sankey customer-journey diagram), Customer Behaviour (KMeans segments + business strategy per segment). |
| 👤 **Traveler 360°** | Full profile for any returning traveler — visit history, rating trend, favourite category/mode, segment, and personalized recommendations. |
| 🧭 **Predict Visit Mode** | The original classifier, plus confidence tiers (🟢/🟡/🔴), a SHAP "why did the model predict this?" breakdown, and a live what-if scenario simulator. |
| 🎯 **Recommend Attractions** | The original 3 recommendation modes (collaborative / content-similarity / new-visitor), now with richer cards (rating, visit count, "best suited for", popularity percentile, "why this recommendation?"), plus a side-by-side method comparison. |
| 🗺️ **AI Trip Planner** | Pick a category, travel mode, trip length, and minimum rating — get a day-by-day itinerary. |
| 🧪 **Model Performance** | Regression + classification model comparisons (now including the actually-deployed LightGBM models — see note below), confusion matrix, ROC curves, per-class F1 breakdown, feature importance, actual-vs-predicted scatter, residuals. |
| 💼 **Business Insights** | Automated insight generator (nothing hardcoded — every line is computed from the live data), segment strategy table, business recommendations, and CSV/HTML report downloads. |

## Two things I fixed while extending this

1. **`requirements.txt` was missing `lightgbm`.** `utils.py` already imported it
   and the shipped `best_classifier.pkl` / `best_regressor.pkl` are both
   LightGBM models — so a fresh `pip install -r requirements.txt` would have
   installed fine but the app would have crashed on the first import. Added
   `lightgbm` and `shap` (needed for the new explainability feature).
2. **The Model Performance comparison table didn't include the deployed
   model.** `lookup_tables.pkl`'s `reg_results_df` / `cls_results_df` only
   list Random Forest / XGBoost / Gradient Boosting — but the actual
   `best_regressor.pkl` / `best_classifier.pkl` are LightGBM models that beat
   all three (reconstructed weighted F1 ≈ 0.47 vs. the table's best entry of
   ≈ 0.46). The Model Performance page now reconstructs the exact original
   held-out test split (same cleaning, same saved encoders, same
   `random_state=42` split) and adds a row for the model that's actually
   running, so what recruiters see matches what the app does. Also fixed
   `pipeline/train_pipeline.py`'s `DATA_DIR`, which pointed at a `dataset/`
   folder that doesn't exist in this project (the real folder is
   `Tourism Dataset/`).

## Project structure

```
├── app.py                     # Entry point: theme, global filters, page routing
├── utils.py                   # Original artifact loading, prediction & recommendation logic (unchanged)
├── advanced.py                 # New: split reconstruction, SHAP, insights, trip planner, traveler profiles, downloads
├── filters.py                  # Global filter panel widget + apply logic
├── page_views/                 # One module per page (see table above)
├── artifacts/                  # Pre-trained models & lookup tables (pickled) — loaded at runtime, no training
├── pipeline/train_pipeline.py   # Full retraining pipeline (only needed if you change the data/features)
├── Tourism Dataset/             # Raw source tables — only read by the Model Performance page (to reconstruct
│                                 the eval split) and by train_pipeline.py (if you retrain)
├── .streamlit/config.toml       # App theme
└── requirements.txt
```

The app loads everything from `artifacts/` at startup — **no training happens
at request time.** The Model Performance page does a one-time (cached)
reconstruction of the original train/test split from `Tourism Dataset/` to
compute confusion matrices, ROC curves, and SHAP values against genuinely
held-out data, using the *already-trained* models — it does not retrain
anything.

To regenerate the artifacts after changing the pipeline or refreshing the
data:

```bash
cd pipeline
python train_pipeline.py
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)

Push this folder to a repo, point Streamlit Community Cloud at `app.py`, and
deploy — no secrets needed. First load per session is a few seconds slower
while the Model Performance page's split reconstruction runs the first time
it's visited; every subsequent view is instant (cached for the life of the
deployment).

## Honest limitations (also shown in-app)

- Rating regression R² sits around 0.11 — ratings cluster tightly at 4–5, so
  there's limited variance left for location/time/category features to
  explain.
- Visit-mode classification weighted F1 is ≈ 0.47 on a 5-class, imbalanced
  problem (Couples/Family vastly outnumber Business) — well above the ~0.20
  random-guess baseline, but not high-precision.
- The recommendation engine's collaborative-filtering half only has training
  signal for the 30 attractions with transaction history; the content-based
  and popularity paths (and the AI Trip Planner) fall back gracefully for
  everything else in the full 1,698-attraction catalog.
