# 🌍 Tourism Experience Analytics

### Classification • Prediction • Recommendation • Explainable AI • Interactive Streamlit Analytics

<p align="center">

**An end-to-end Machine Learning and Analytics platform for understanding traveler behavior, predicting tourism ratings, classifying visit modes, and recommending relevant attractions.**

</p>

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

```bash
git clone https://github.com/YOUR_USERNAME/Tourism-Experience-Analytics.git
```

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

```bash
pip install -r requirements.txt
```

## 6. Launch

```bash
streamlit run app.py
```

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
