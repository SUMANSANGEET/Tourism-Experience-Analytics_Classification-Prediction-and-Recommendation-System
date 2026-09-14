# 🌍 Tourism Experience Analytics

<<<<<<< HEAD
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
=======
### Classification • Prediction • Recommendation • Explainable AI • Interactive Streamlit Analytics

<p align="center">

**An end-to-end Machine Learning and Analytics platform for understanding traveler behavior, predicting tourism ratings, classifying visit modes, and recommending relevant attractions.**

</p>
>>>>>>> 914fdf4d29592aa927cae4e3e28c76814d36f7af

<p align="center">

<a href="https://ida8jsmnknyqdvtkqeafcd.streamlit.app/">
<img src="https://img.shields.io/badge/🚀%20LIVE%20DEMO-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
</a>

<img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Pandas-Analytics-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
<img src="https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
<img src="https://img.shields.io/badge/Plotly-Interactive%20Visuals-3F4F75?style=for-the-badge&logo=plotly&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-Deployment-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>

</p>

---

## 🚀 Live Interactive Application

### 👉 [🌐 Open Tourism Experience Analytics — Live Streamlit App](https://ida8jsmnknyqdvtkqeafcd.streamlit.app/)

> **Explore the dashboard interactively:** analyze tourism behavior, explore attraction performance, evaluate machine-learning models, generate predictions, and discover recommendations.

---

# 🎯 Executive Summary

**Tourism Experience Analytics** is an end-to-end **Data Analytics + Machine Learning + Recommendation System** developed to convert complex tourism data into actionable business intelligence.

The platform answers three core questions:

### ⭐ 1. What happened?

Understand historical tourism behavior through interactive analytics and visualizations.

### 🔮 2. What is likely to happen?

Predict attraction ratings and classify traveler visit modes using machine-learning models.

### 🎯 3. What should the traveler explore next?

Generate attraction recommendations using a content-based recommendation approach.

---

# 💼 Business Problem

Tourism organizations collect large amounts of data about:

* Travelers
* Attractions
* Cities
* Countries
* Regions
* Continents
* Visit modes
* Ratings
* Attraction categories
* Historical interactions

But raw data does not automatically provide business value.

The challenge is to transform this information into a system capable of:

```text
RAW TOURISM DATA
       │
       ▼
DATA QUALITY & CLEANING
       │
       ▼
EXPLORATORY ANALYTICS
       │
       ▼
BUSINESS INSIGHTS
       │
       ├───────────────┐
       ▼               ▼
PREDICTION       CLASSIFICATION
       │               │
       └───────┬───────┘
               ▼
       RECOMMENDATION
               │
               ▼
       EXPLAINABLE AI
               │
               ▼
       INTERACTIVE APP
               │
               ▼
       BUSINESS DECISIONS
```
<<<<<<< HEAD
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
=======

---

# 🌟 What Makes This Project Different?

This is **not just an EDA project**.

It combines multiple stages of a real-world analytics workflow:

| Capability               | Implementation                         |
| ------------------------ | -------------------------------------- |
| 📊 Data Analytics        | Tourism behavior & attraction analysis |
| 🧹 Data Engineering      | Cleaning, validation & integration     |
| 📈 Business Intelligence | KPI-driven visual analytics            |
| 🤖 Regression            | Attraction rating prediction           |
| 🧭 Classification        | Visit-mode prediction                  |
| 🎯 Recommendation        | Personalized attraction discovery      |
| 🔍 Explainable AI        | Feature importance / SHAP              |
| 🧪 Model Benchmarking    | Multiple candidate models              |
| 🖥️ Deployment           | Interactive Streamlit application      |
| 📦 Version Control       | Git + GitHub                           |

---

# 🧭 Application Journey

The application follows a business-user-friendly analytical journey:

```text
┌──────────────────────────────┐
│       🌍 TOURISM DATA        │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│      📊 EXPLORE DATA         │
│  KPIs • Trends • Geography   │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│      🏛️ EXPLORE ATTRACTIONS │
│ Ratings • Types • Popularity │
└──────────────┬───────────────┘
               ▼
       ┌───────┴────────┐
       ▼                ▼
┌─────────────┐   ┌─────────────┐
│ ⭐ PREDICT  │   │ 🧭 CLASSIFY │
│   RATING    │   │ VISIT MODE  │
└──────┬──────┘   └──────┬──────┘
       └────────┬─────────┘
                ▼
       ┌──────────────────┐
       │ 🎯 RECOMMEND     │
       │   ATTRACTIONS    │
       └────────┬─────────┘
                ▼
       ┌──────────────────┐
       │ 🔍 EXPLAIN MODEL │
       └──────────────────┘
```

---

# 📊 Interactive Analytics

The dashboard is designed around **interactive visual storytelling** rather than static tables.

## 👥 Traveler Analytics

Explore:

* Traveler distribution
* Origin patterns
* Visit behavior
* Traveler segmentation
* Visit-mode composition

### Questions answered

> Which traveler groups are most represented?

> Which visit modes dominate?

> How does traveler behavior differ across destinations?

---

# 🌍 Geographic Intelligence

Analyze tourism patterns across:

```text
🌎 Continent
   ↓
🌏 Region
   ↓
🇺🇳 Country
   ↓
🏙️ City
```

### Business questions

* Which destinations attract the most visitors?
* Which regions show stronger tourism activity?
* How do attraction ratings differ geographically?
* Which destinations could represent growth opportunities?

---

# 🏛️ Attraction Analytics

Analyze attraction-level performance through:

* ⭐ Rating distributions
* 🏆 Top-rated attractions
* 📊 Popularity
* 🏷️ Attraction categories
* 📍 Geographic distribution
* 📈 Historical performance

A minimum-review threshold can also be applied when identifying top attractions to avoid misleading results from attractions with very few ratings.

---

# ⭐ Rating Prediction

## Regression Problem

### Objective

Predict the expected rating of an attraction using available tourism, traveler, geographic, and behavioral features.

```text
INPUT FEATURES
      │
      ▼
FEATURE ENGINEERING
      │
      ▼
REGRESSION MODELS
      │
      ▼
MODEL BENCHMARKING
      │
      ▼
BEST MODEL
      │
      ▼
PREDICTED RATING
```

### Evaluation Metrics

The regression workflow evaluates models using:

* **R²**
* **MAE**
* **RMSE**

### Why it matters

Rating prediction can help tourism businesses:

* Identify attractions with strong predicted performance
* Understand rating-driving variables
* Prioritize experience improvements
* Support attraction ranking
* Improve destination discovery

---

# 🧭 Visit Mode Classification

## Classification Problem

The system predicts a traveler's likely visit mode from relevant tourism characteristics.

Examples include:

```text
👨‍💼 Business
👨‍👩‍👧 Family
🧍 Solo
👫 Couples
👥 Groups
```

### Machine Learning Workflow

```text
Traveler / Tourism Features
             │
             ▼
      Feature Engineering
             │
             ▼
      Candidate Models
             │
             ▼
      Model Benchmarking
             │
             ▼
       Best Classifier
             │
             ▼
    Predicted Visit Mode
```

### Evaluation

Classification performance can be examined using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

### Business Value

Visit-mode prediction enables:

* Traveler segmentation
* Personalized marketing
* Targeted campaigns
* Recommendation personalization
* Customer-experience optimization

---

# 🎯 Attraction Recommendation Engine

The platform also includes a **content-based recommendation system**.

## Recommendation Logic

```text
Traveler Preferences
        +
Attraction Metadata
        +
Available Historical Signals
        │
        ▼
Feature Representation
        │
        ▼
Similarity Calculation
        │
        ▼
Candidate Attractions
        │
        ▼
Ranking
        │
        ▼
🎯 Recommended Attractions
```

### Recommendation output can include

```text
🏛️ Attraction
📍 Location
⭐ Rating
🏷️ Attraction Type
🌎 Region
📊 Relevance
```

### Cold-Start Strategy

When sufficient traveler history is unavailable, the system can use attraction metadata and popularity-related signals as a fallback.

This makes the recommendation component more practical for new users.

---

# 🔍 Explainable AI

A prediction is more useful when users can understand **why** the model made it.

The project incorporates explainability through feature-importance analysis and SHAP-based analysis where supported by the installed environment.

### Instead of:

```text
Prediction = 4.3
```

The objective is to provide:

```text
Prediction = 4.3

+
Important contributing features

+
Model explanation
```

This helps bridge the gap between:

**Machine Learning → Business Understanding**

---

# 🧪 Model Performance Laboratory

The application includes model-performance analysis to compare candidate models before selecting the model used by the application.

### Regression

```text
Candidate Model 1 ──► R² / MAE / RMSE
Candidate Model 2 ──► R² / MAE / RMSE
Candidate Model 3 ──► R² / MAE / RMSE
                         │
                         ▼
                    🏆 WINNER
```

### Classification

```text
Candidate Model 1 ──► Accuracy / F1
Candidate Model 2 ──► Accuracy / F1
Candidate Model 3 ──► Accuracy / F1
                         │
                         ▼
                    🏆 WINNER
```

This creates a reproducible:

> **Benchmark → Compare → Select → Deploy**

workflow.

---

# 🧹 Data Quality & Preparation

Before modeling, the project performs a structured data-quality workflow.

```text
RAW DATA
   │
   ▼
Schema Inspection
   │
   ▼
Missing Values
   │
   ▼
Duplicates
   │
   ▼
Data Types
   │
   ▼
Category Validation
   │
   ▼
Relationship Validation
   │
   ▼
Cleaning & Transformation
   │
   ▼
MASTER DATASET
```

### Key activities

* Missing-value analysis
* Duplicate detection
* Data-type correction
* Category standardization
* Lookup validation
* Relationship checks
* Feature engineering
* Outlier investigation
* Multi-table integration

---

# 🔗 Data Architecture

The project integrates multiple tourism datasets to create a richer analytical model.

### Data sources include

```text
Transaction.xlsx
User.xlsx
City.xlsx
Country.xlsx
Region.xlsx
Continent.xlsx
Mode.xlsx
Type.xlsx
Updated_Item.xlsx
```

Conceptually:

```text
                    ┌──────────────┐
                    │     USER     │
                    └──────┬───────┘
                           │
                           ▼
┌─────────────┐     ┌───────────────┐     ┌──────────────┐
│    CITY     │────►│ TRANSACTION   │◄────│  ATTRACTION  │
└──────┬──────┘     └───────────────┘     └──────┬───────┘
       │                                          │
       ▼                                          ▼
┌─────────────┐                             ┌──────────────┐
│  COUNTRY    │                             │ ATTRACTION   │
└──────┬──────┘                             │    TYPE      │
       │                                    └──────────────┘
       ▼
┌─────────────┐
│   REGION    │
└──────┬──────┘
       ▼
┌─────────────┐
│  CONTINENT  │
└─────────────┘
```

---

# 📈 Visual Storytelling

The dashboard focuses on visuals that help users move from **observation → interpretation → decision**.

### KPI Layer

```text
┌─────────────┬─────────────┬─────────────┬─────────────┐
│ 👥 Travelers│ 🏛️ Attractions│ ⭐ Avg Rating│ 📊 Visits │
└─────────────┴─────────────┴─────────────┴─────────────┘
```

### Analytical Layer

```text
📊 Distribution
📈 Trends
🗺️ Geography
🏆 Rankings
🔎 Relationships
🧭 Segmentation
```

### Predictive Layer

```text
⭐ Rating Prediction
🧭 Visit Mode Classification
🎯 Recommendations
🔍 Explainability
```

---

# 🛠️ Technology Stack

### 🐍 Programming

* Python

### 📊 Data Analytics

* Pandas
* NumPy

### 📈 Visualization

* Plotly
* Matplotlib

### 🤖 Machine Learning

* Scikit-learn
* XGBoost

### 🔍 Explainable AI

* SHAP

### 🖥️ Application

* Streamlit

### 📓 Development

* Jupyter Notebook
* Git
* GitHub

### 📁 Data

* Excel
* Structured tourism datasets

---

# 📁 Repository Structure

```text
Tourism Experience Analytics/
│
├── app.py
├── Tourism Experience Analytics.ipynb
├── README.md
├── requirements.txt
├── .gitignore
│
├── artifacts/
│   └── best_classifier.txt
│
├── pipeline/
│   └── train_pipeline.py
│
├── convert_classifier.py
├── utils.py
│
├── .streamlit/
│   └── config.toml
│
└── ...
```

> Large raw datasets, credentials, API keys, environment files, and unnecessary generated files should remain outside the public repository.

---

# ⚙️ Run Locally

## 1. Clone
>>>>>>> 914fdf4d29592aa927cae4e3e28c76814d36f7af

```bash
git clone https://github.com/YOUR_USERNAME/Tourism-Experience-Analytics.git
```

<<<<<<< HEAD
## Run locally
=======
## 2. Enter the project

```bash
cd Tourism-Experience-Analytics
```

## 3. Create a virtual environment

```bash
python -m venv venv
```

## 4. Activate

### Windows

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

## 5. Install dependencies
>>>>>>> 914fdf4d29592aa927cae4e3e28c76814d36f7af

```bash
pip install -r requirements.txt
```

## 6. Launch

```bash
streamlit run app.py
```

<<<<<<< HEAD
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
=======
---

# 🎮 Interactive User Journey

### 01 — Explore

Start with the dashboard overview and understand the tourism ecosystem.

### 02 — Investigate

Drill down into traveler, attraction, geographic and behavioral patterns.

### 03 — Compare

Benchmark candidate machine-learning models.

### 04 — Predict

Generate an attraction-rating prediction.

### 05 — Classify

Predict the likely visit mode.

### 06 — Recommend

Generate relevant attractions.

### 07 — Explain

Inspect important model features and understand predictions.

---

# 💼 Business Use Cases

## 🏨 Tourism Platforms

Improve destination and attraction discovery.

## 🎯 Personalized Marketing

Segment travelers and deliver targeted campaigns.

## 🏛️ Attraction Management

Identify high-performing and underperforming attractions.

## ⭐ Experience Optimization

Understand factors associated with attraction ratings.

## 🌍 Destination Planning

Identify geographic tourism patterns and opportunities.

## 🤖 Intelligent Discovery

Recommend relevant attractions to travelers.

---

# 📌 Key Questions This Project Answers

| Business Question                    | Analytics / ML Solution |
| ------------------------------------ | ----------------------- |
| Who are the travelers?               | Traveler analytics      |
| Where do travelers come from?        | Geographic analysis     |
| Which attractions perform best?      | Attraction analytics    |
| What are the rating patterns?        | Rating analysis         |
| Can ratings be predicted?            | Regression              |
| Can visit mode be predicted?         | Classification          |
| What should a traveler visit?        | Recommendation system   |
| Why did the model make a prediction? | Explainability          |
| Which model performs best?           | Model benchmarking      |

---

# 🧠 End-to-End Skills Demonstrated

### Data Analytics

* Data cleaning
* Exploratory Data Analysis
* KPI development
* Business interpretation
* Statistical analysis
* Interactive visualization

### Machine Learning

* Feature engineering
* Regression
* Classification
* Model comparison
* Model evaluation
* Prediction

### Recommendation Systems

* Content-based recommendation
* Similarity-based ranking
* Cold-start fallback

### Explainable AI

* Feature importance
* SHAP-based interpretation

### Deployment

* Streamlit
* Model integration
* Interactive dashboard development
* Git/GitHub

---

# 📊 From Data to Decision

The core philosophy of the project is:

```text
             DATA
              │
              ▼
          INSIGHTS
              │
              ▼
          PATTERNS
              │
              ▼
         PREDICTIONS
              │
              ▼
       RECOMMENDATIONS
              │
              ▼
       BUSINESS ACTION
```

The goal is not simply to build a model.

The goal is to create an **end-to-end decision-support experience**.

---

# 🔮 Future Enhancements

Potential next-generation improvements include:

* 🔄 Real-time tourism data APIs
* 🤝 Hybrid recommendation systems
* 🧠 Collaborative filtering
* 🗺️ Advanced geospatial analytics
* 📈 Tourism demand forecasting
* 👥 Advanced traveler clustering
* 💬 Review sentiment analysis
* ☁️ Cloud database integration
* 🔁 Automated model retraining
* 📊 MLflow experiment tracking
* 🔐 Role-based dashboard access
* ⚡ Real-time recommendation APIs

---

# 🏆 Project Highlights

```text
┌────────────────────────────────────────────────────────┐
│                 TOURISM ANALYTICS                      │
├────────────────────────────────────────────────────────┤
│                                                        │
│  📊 EDA                  → Understand                  │
│  🧹 Data Engineering     → Prepare                    │
│  📈 Visualization        → Discover                   │
│  ⭐ Regression           → Predict                    │
│  🧭 Classification       → Segment                    │
│  🎯 Recommendation       → Personalize                │
│  🔍 Explainability       → Understand Why             │
│  🖥️ Streamlit            → Deploy                     │
│                                                        │
└────────────────────────────────────────────────────────┘
```

---

# 👨‍💻 Author

## **Suman Sangeet**

🎓 **PGDM — Big Data Analytics**

### Areas of Interest

`Data Analytics` • `Business Intelligence` • `Python` • `SQL` • `Machine Learning` • `Power BI` • `Data Visualization` • `AI`

---

# 🚀 Explore the Project

### 🌐 Live Application

**[👉 Launch Tourism Experience Analytics](https://ida8jsmnknyqdvtkqeafcd.streamlit.app/)**

### 💻 Source Code

**[👉 Explore the GitHub Repository](https://github.com/YOUR_USERNAME/Tourism-Experience-Analytics)**

---

# ⭐ Final Takeaway

> ### **Raw Tourism Data → Interactive Analytics → Predictive Intelligence → Personalized Recommendations → Business Decisions**

This project demonstrates the complete journey from **data preparation and exploratory analysis to machine learning, explainability, recommendation systems and production-style interactive deployment**.

**Built with Python • Pandas • Scikit-learn • XGBoost • Plotly • SHAP • Streamlit**
>>>>>>> 914fdf4d29592aa927cae4e3e28c76814d36f7af
