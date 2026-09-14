# 🌋 Tourism Experience Analytics — Streamlit App

An interactive tourism intelligence platform built on a real 52,930-transaction,
33,530-user, 8-table tourism dataset (Bali, Yogyakarta, and Malang attractions).

It combines three ML products behind one interface:

1. **Classification** — predicts a traveler's visit mode (Business / Couples /
   Family / Friends / Solo) from their location, travel dates, and attraction
   interest.
2. **Regression** — estimates the rating a traveler is likely to give an
   attraction.
3. **Recommendation** — a hybrid engine: item-based collaborative filtering
   for returning travelers with rating history, and a content + popularity
   blend for brand-new visitors (cold start).

It also ships recruiter-friendly visual analytics: popular attractions, top
regions, visit-mode mix, seasonality, and a KMeans-based traveler segmentation.

## Project structure

```
TourismExperienceAnalytics/
├── app.py                    # Streamlit application (5 pages)
├── utils.py                  # Artifact loading, prediction & recommendation logic
├── requirements.txt
├── .streamlit/config.toml    # App theme
├── artifacts/                # Pre-trained models & lookup tables (pickled)
├── pipeline/
│   └── train_pipeline.py     # Full data-cleaning + training pipeline (re-run to retrain)
└── dataset/                  # Raw source tables (Transaction, User, City, ...)
```

The app loads everything from `artifacts/` at startup — **no training happens
at request time.** To regenerate the artifacts after changing the pipeline or
refreshing the data, run:

```bash
cd pipeline
python train_pipeline.py
```

This reads `../dataset/*.xlsx`, retrains every model, and rewrites
`../artifacts/*.pkl`.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## App pages

| Page | What it does |
|---|---|
| 🏝️ Overview | KPI summary, project description, quick snapshot charts |
| 📊 Visual Insights | Popular attractions, regions & geography, user segments, trends & ratings |
| 🧭 Predict Visit Mode | Form → predicted visit mode + confidence + estimated rating |
| 🎯 Recommend Attractions | Returning-traveler (collaborative filtering) or new-visitor (content/popularity) recommendations |
| 🧪 Model Performance | Regression & classification model comparison, feature importance |

## Deployment

### Option 1 — Streamlit Community Cloud (recommended, free)
1. Push this folder to a public (or private) GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. **New app** → select the repo/branch → set **Main file path** to `app.py`.
4. Deploy. Streamlit Cloud installs `requirements.txt` automatically.
5. Because `artifacts/*.pkl` are committed to the repo, no training happens
   on deploy — the app boots straight from the saved models.

### Option 2 — Hugging Face Spaces
1. Create a new **Space** → SDK: **Streamlit**.
2. Upload all files in this folder (keep the folder structure — `artifacts/`,
   `.streamlit/`, `pipeline/`, `dataset/` included, or at minimum `app.py`,
   `utils.py`, `requirements.txt`, `.streamlit/`, and `artifacts/`).
3. The Space auto-builds and serves the app; no extra config needed since
   `app.py` is already the entry point Streamlit Spaces expect.

### Option 3 — Quick local share via ngrok
```bash
pip install -r requirements.txt
streamlit run app.py &
ngrok http 8501
```
Share the `https://*.ngrok-free.app` URL ngrok prints — useful for a fast
demo link without deploying anywhere.

## Data notes

- `Item.xlsx` holds the 30 attractions that actually appear in the transaction
  log — these back the collaborative-filtering recommender and the rating
  history shown per traveler.
- `Updated_Item.xlsx` is the full 1,698-attraction catalog, used to build a
  broader content-similarity space so new/cold-start attractions still have a
  sensible fallback.
- `AttractionTypeId` in the raw `Item`/`Updated_Item` tables is inconsistently
  encoded (numeric ID for most rows, the category name itself for a batch of
  others) — the pipeline resolves this into one clean `AttractionCategory`
  column before anything else runs.
- All target-derived aggregate features (`AttractionAvgRating`,
  `UserAvgRating`) are computed **after** the train/test split, on the
  training fold only, to avoid leakage.

## Model results (current artifacts)

**Regression (rating prediction)** — best model auto-selected by R²:
Random Forest, R² ≈ 0.11. Ratings cluster tightly around 4–5 (median 4), so
this R² range is expected for review-style data; `AttractionAvgRating`
dominates feature importance.

**Classification (visit mode)** — best model auto-selected by weighted F1:
XGBoost, F1 ≈ 0.46 on a 5-class, imbalanced target (Couples/Family dominate,
Business is a minority class). `UserAvgRating` and geography are the leading
signals.

Full comparison tables and feature-importance charts for every candidate
model are in the app's **🧪 Model Performance** page.
