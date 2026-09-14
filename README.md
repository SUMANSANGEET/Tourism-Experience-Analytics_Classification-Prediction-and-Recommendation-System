# 🌍 Tourism Experience Analytics

### Classification • Rating Prediction • Recommendation • Interactive Business Intelligence

<p align="center">

<a href="https://ida8jsmnknyqdvtkqeafcd.streamlit.app/">
<img src="https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit-FF4B4B?style=for-the-badge" alt="Live Demo"/>
</a>

<a href="https://github.com/SUMANSANGEET/Tourism-Experience-Analytics_Classification-Prediction-and-Recommendation-System">
<img src="https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github" alt="GitHub"/>
</a>

<img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>

<img src="https://img.shields.io/badge/Streamlit-Interactive%20Analytics-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>

<img src="https://img.shields.io/badge/Machine%20Learning-Predictive%20Analytics-8A2BE2?style=for-the-badge" alt="Machine Learning"/>

</p>

<p align="center">

<b>Transforming tourism data into actionable traveler insights, predictive intelligence, and personalized recommendations.</b>

</p>

---

## 🚀 Live Interactive Experience

### 🎯 Explore the deployed application

👉 **[Launch Tourism Experience Analytics](https://ida8jsmnknyqdvtkqeafcd.streamlit.app/)**

The application converts raw tourism transaction and attraction data into an interactive analytics platform where users can explore:

* 📊 Traveler demographics and behavior
* 🌍 Geographic tourism patterns
* ⭐ Attraction ratings
* 🧭 Visit-mode preferences
* 🏙️ Destination performance
* 📈 Tourism trends
* 🤖 Machine-learning predictions
* 🎯 Personalized attraction recommendations
* 🔍 Model explainability and performance

> **Goal:** Move beyond descriptive dashboards toward a decision-support system that answers **what happened, why it happened, what may happen next, and what should be recommended.**

---

# 🧭 Executive Overview

Tourism businesses generate large volumes of information across travelers, destinations, attractions, countries, regions, visit modes, and ratings.

However, raw tourism data does not automatically answer critical business questions:

> **Who are our travelers?**

> **What destinations and attractions drive engagement?**

> **What factors influence ratings?**

> **Which visit mode is likely for a traveler?**

> **What attraction should we recommend next?**

This project addresses these questions by combining:

**Business Intelligence + Exploratory Data Analysis + Machine Learning + Recommendation Systems + Interactive Visualization**

into a single recruiter-friendly analytics platform.

---

# 💼 Business Problem

Tourism organizations need to understand traveler behavior in order to:

* Improve destination experiences
* Identify high-performing attractions
* Understand customer preferences
* Predict traveler ratings
* Segment visitors
* Optimize tourism offerings
* Personalize attraction recommendations
* Support marketing and destination planning

### Traditional approach

```text
Raw Tourism Data
       ↓
Static Reports
       ↓
Historical Insights
```

### This project

```text
Raw Tourism Data
       ↓
Data Quality & Preparation
       ↓
Exploratory Analytics
       ↓
Business Intelligence
       ↓
Predictive Modeling
       ↓
Classification
       ↓
Recommendation Engine
       ↓
Interactive Decision Support
```

---

# 🎯 Project Objectives

### 01 — Understand Traveler Behavior

Analyze:

* Traveler origin
* Destination
* Visit mode
* Attraction category
* Rating behavior
* Geographic patterns

### 02 — Identify Tourism Performance Drivers

Discover:

* Top attractions
* Popular attraction categories
* High-performing destinations
* Rating patterns
* Visit-mode trends

### 03 — Predict Traveler Ratings

Build regression models capable of estimating attraction ratings from available traveler and attraction attributes.

### 04 — Classify Visit Mode

Predict likely traveler visit modes such as:

* Family
* Business
* Couples
* Solo
* Other available categories

### 05 — Personalize Recommendations

Recommend relevant attractions using content-based similarity and attraction metadata.

### 06 — Enable Interactive Decision Making

Deliver insights through a Streamlit-based analytics platform rather than static charts.

---

# 📊 Analytics Questions Answered

The platform is designed around real-world business questions.

| Business Question                           | Analytical Solution           |
| ------------------------------------------- | ----------------------------- |
| Where do travelers originate?               | Geographic analysis           |
| Which attractions are most popular?         | Attraction ranking            |
| Which attractions receive stronger ratings? | Rating analysis               |
| What visit modes dominate?                  | Classification & segmentation |
| Which regions generate tourism activity?    | Regional analysis             |
| How do traveler types differ?               | Behavioral analysis           |
| What influences ratings?                    | Predictive modeling           |
| What visit mode might a traveler belong to? | Classification                |
| What attraction should be recommended?      | Recommendation engine         |
| How reliable are the models?                | Model benchmarking            |

---

# 🏗️ Solution Architecture

```text
                    ┌───────────────────────┐
                    │   Tourism Data Files  │
                    │                       │
                    │ Transaction           │
                    │ User                  │
                    │ City                  │
                    │ Country               │
                    │ Region                │
                    │ Continent             │
                    │ Mode                  │
                    │ Type                  │
                    │ Attraction Metadata   │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ Data Quality & ETL    │
                    │                       │
                    │ Cleaning              │
                    │ Validation            │
                    │ Missing Values        │
                    │ Data Integration      │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ Master Tourism Table  │
                    └───────────┬───────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
       ┌─────────────┐   ┌─────────────┐   ┌──────────────┐
       │     EDA     │   │ ML Models   │   │ Recommender  │
       │             │   │             │   │              │
       │ Trends      │   │ Regression  │   │ Similarity   │
       │ Ratings     │   │ Classifier  │   │ Attractions  │
       │ Geography   │   │ Prediction  │   │ Personalize  │
       └──────┬──────┘   └──────┬──────┘   └──────┬───────┘
              │                 │                 │
              └─────────────────┼─────────────────┘
                                ▼
                    ┌───────────────────────┐
                    │ Interactive Streamlit │
                    │ Analytics Platform    │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ Business Decisions    │
                    │ & Traveler Insights   │
                    └───────────────────────┘
```

---

# 🗂️ Data Ecosystem

The project integrates multiple tourism datasets.

| Dataset             | Purpose                           |
| ------------------- | --------------------------------- |
| `Transaction.xlsx`  | Traveler-attraction interactions  |
| `User.xlsx`         | Traveler information              |
| `City.xlsx`         | City-level geographic information |
| `Country.xlsx`      | Country mapping                   |
| `Region.xlsx`       | Regional hierarchy                |
| `Continent.xlsx`    | Continental hierarchy             |
| `Mode.xlsx`         | Visit-mode lookup                 |
| `Type.xlsx`         | Attraction-type lookup            |
| `Updated_Item.xlsx` | Enriched attraction metadata      |

### Data Engineering Workflow

```text
Multiple Excel Sources
        ↓
Schema Inspection
        ↓
Data Quality Audit
        ↓
Missing-Value Analysis
        ↓
Data Type Standardization
        ↓
Lookup Integration
        ↓
Feature Engineering
        ↓
Master Analytical Dataset
```

---

# 🧹 Data Quality & Preparation

The analytical pipeline includes:

* Missing-value assessment
* Duplicate detection
* Data-type validation
* Lookup-table integration
* Categorical normalization
* Numerical feature preparation
* Relationship validation
* Feature engineering
* Analytical dataset creation

### Data Quality Principle

> **Reliable analytics starts with reliable data.**

Rather than immediately training models, the project first establishes a consistent analytical foundation.

---

# 📈 Exploratory Data Analysis

The EDA layer focuses on discovering behavioral and business patterns.

## 🌍 Traveler Geography

Analyze:

* Country of origin
* Region
* Continent
* Destination
* Traveler distribution

### Business Insight

Geographic patterns can help tourism organizations understand where their visitors originate and which markets may require targeted campaigns.

---

# 🧭 Visit Mode Analysis

Analyze the distribution of traveler behavior across available visit modes.

Example analytical questions:

* Which visit mode dominates?
* How does visit mode differ by region?
* Are business travelers concentrated in specific destinations?
* Which destinations attract more family or solo travelers?

---

# ⭐ Rating Distribution

Understand:

* Average rating
* Rating frequency
* High-rated attractions
* Low-rated attractions
* Rating patterns by attraction type

### Customer Engagement Perspective

Ratings provide a behavioral signal that can be used to identify:

> **What travelers value, what experiences perform well, and where experience quality may need improvement.**

---

# 🏆 Attraction Performance

The platform can identify high-performing attractions using:

* Rating
* Number of reviews/interactions
* Attraction category
* Geographic location

Rather than ranking attractions only by average rating, the analysis can consider minimum interaction thresholds to reduce misleading rankings from attractions with very few observations.

---

# 📅 Tourism Trends

Time-based analysis helps identify:

* Seasonal patterns
* Changes in traveler activity
* Rating trends
* Visit-mode changes
* Destination demand patterns

These insights can support:

* Capacity planning
* Marketing campaigns
* Staffing decisions
* Destination management

---

# 🔗 Relationship Analysis

The project explores relationships between variables such as:

```text
Traveler Characteristics
        ↓
Visit Mode
        ↓
Attraction Type
        ↓
Destination
        ↓
Rating
```

Correlation and comparative analysis help identify potential relationships between traveler behavior, attraction characteristics, and experience ratings.

---

# 🤖 Machine Learning Layer

The project contains two major predictive components.

---

## 📈 1. Rating Prediction — Regression

### Objective

Predict a traveler's attraction rating based on available analytical features.

### Workflow

```text
Features
   ↓
Preprocessing
   ↓
Train / Validation / Test
   ↓
Candidate Models
   ↓
Model Benchmarking
   ↓
Best Model Selection
   ↓
Rating Prediction
```

### Evaluation Metrics

The regression pipeline can evaluate models using:

* R²
* MAE
* RMSE

### Why these metrics?

**R²**

Measures the proportion of variance explained by the model.

**MAE**

Shows average absolute prediction error.

**RMSE**

Penalizes larger prediction errors more strongly.

---

# 🧭 2. Visit Mode Classification

### Objective

Predict the likely visit mode of a traveler based on available features.

```text
Traveler / Attraction Features
            ↓
      Feature Engineering
            ↓
       Classification
            ↓
     Model Benchmarking
            ↓
      Best Classifier
            ↓
 Predicted Visit Mode
```

### Evaluation

Classification performance can be evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

The goal is not simply to maximize accuracy, but to understand how well the model performs across different traveler categories.

---

# 🏆 Model Benchmarking

Instead of assuming a single algorithm is best, candidate models are compared systematically.

```text
Candidate Models
       ↓
Cross-Model Evaluation
       ↓
Performance Comparison
       ↓
Best Model Selection
       ↓
Production Model
```

This approach demonstrates a practical machine-learning workflow:

> **Experiment → Evaluate → Compare → Select → Deploy**

---

# 🎯 Recommendation Engine

One of the key customer-engagement features is the attraction recommendation component.

### Recommendation concept

```text
Selected Attraction
        ↓
Attraction Metadata
        ↓
Feature Representation
        ↓
Similarity Calculation
        ↓
Rank Similar Attractions
        ↓
Recommended Attractions
```

Recommendations can leverage attraction metadata such as:

* Attraction type
* Destination
* Category
* Location
* Other available descriptive attributes

### Customer Engagement Value

Instead of asking:

> "What attractions are popular?"

the platform moves toward:

> **"What attraction is relevant to this traveler or selected experience?"**

This creates a more personalized tourism experience.

---

# 🧠 Recommendation Strategy

The system follows a content-based approach.

### Example

If a traveler selects:

**Historical Museum**

the system can search for attractions with similar characteristics and return:

```text
Recommended Attraction #1
Recommended Attraction #2
Recommended Attraction #3
Recommended Attraction #4
Recommended Attraction #5
```

This creates an experience-discovery workflow rather than a simple static ranking.

---

# 📊 Interactive Dashboard Experience

The Streamlit application is designed to encourage exploration.

Users can interact with analytical components rather than simply reading static reports.

### Potential exploration flow

```text
🌍 Explore Geography
        ↓
🧭 Understand Traveler Behavior
        ↓
⭐ Analyze Ratings
        ↓
🏆 Discover Attractions
        ↓
🤖 Explore Predictions
        ↓
🎯 Generate Recommendations
```

---

# 🎨 Visual Analytics

The application emphasizes visual storytelling through interactive charts such as:

* 📊 Bar charts
* 📈 Trend charts
* 🥧 Distribution charts
* 🌍 Geographic analysis
* 🔥 Correlation visualizations
* 🎯 Recommendation outputs
* 📋 Interactive tables
* 📈 Model performance comparisons

Interactive visualizations allow users to filter, compare, investigate, and discover patterns.

---

# 👥 Customer / Traveler Engagement

The project is not limited to technical machine learning.

It focuses on the traveler journey:

```text
Traveler
   ↓
Behavior
   ↓
Experience
   ↓
Rating
   ↓
Prediction
   ↓
Recommendation
   ↓
Better Discovery
```

### Engagement Opportunities

The platform can help tourism businesses:

* Understand customer preferences
* Discover popular experiences
* Identify underperforming attractions
* Personalize recommendations
* Improve destination strategy
* Support customer experience optimization

---

# 💼 Business Use Cases

## 🏨 Hospitality

Identify attraction preferences of different traveler segments.

## ✈️ Travel Platforms

Build recommendation experiences around traveler interests.

## 🌍 Destination Management

Identify high-performing and emerging tourism destinations.

## 📣 Tourism Marketing

Develop campaigns based on traveler origin and behavioral patterns.

## 🎯 Personalization

Recommend attractions based on similarity and traveler behavior.

## 📊 Business Intelligence

Monitor tourism performance through interactive analytics.

---

# 🛠️ Technology Stack

### Programming

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square\&logo=python\&logoColor=white)

### Data Analytics

![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square\&logo=pandas\&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square\&logo=numpy\&logoColor=white)

### Visualization

![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat-square\&logo=plotly\&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square)

### Machine Learning

![Scikit Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square\&logo=scikit-learn\&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=flat-square)

### Application

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square\&logo=streamlit\&logoColor=white)

### Development

* Jupyter Notebook
* Git
* GitHub
* Python
* Excel

---

# 📂 Project Structure

```text
Tourism Experience Analytics/
│
├── app.py
│
├── data/
│   ├── Transaction.xlsx
│   ├── User.xlsx
│   ├── City.xlsx
│   ├── Country.xlsx
│   ├── Region.xlsx
│   ├── Continent.xlsx
│   ├── Mode.xlsx
│   ├── Type.xlsx
│   └── Updated_Item.xlsx
│
├── models/
│   └── Trained ML Models
│
├── artifacts/
│   └── Model / Analytical Artifacts
│
├── notebooks/
│   └── Exploratory Analysis & Modeling
│
├── utils/
│   └── Supporting Modules
│
├── screenshots/
│   └── Dashboard Images
│
├── requirements.txt
├── README.md
└── .gitignore
```

> **Note:** The exact folder structure may vary depending on the deployed version of the project.

---

# ⚙️ Local Installation

## 1. Clone Repository

```bash
git clone https://github.com/SUMANSANGEET/Tourism-Experience-Analytics_Classification-Prediction-and-Recommendation-System.git
```

## 2. Navigate to Project

```bash
cd Tourism-Experience-Analytics_Classification-Prediction-and-Recommendation-System
```

## 3. Create Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## 5. Launch Streamlit

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 🔐 Configuration

If API keys, credentials, or secrets are required in future versions, store them securely using environment variables or Streamlit secrets.

Never commit:

```text
.env
secrets.toml
API keys
Passwords
Private credentials
```

---

# 📸 Dashboard Preview

Add your best application screenshots here.

Recommended screenshots:

### 🏠 Executive Dashboard

```text
screenshots/dashboard.png
```

### 📊 Exploratory Analytics

```text
screenshots/eda.png
```

### 🤖 Model Performance

```text
screenshots/model-performance.png
```

### 🎯 Recommendation Engine

```text
screenshots/recommendations.png
```

### 🧭 Classification

```text
screenshots/classification.png
```

Example Markdown:

```markdown
![Executive Dashboard](screenshots/dashboard.png)

![Exploratory Analytics](screenshots/eda.png)

![Model Performance](screenshots/model-performance.png)

![Recommendation Engine](screenshots/recommendations.png)
```

---

# 🧪 Analytical Workflow

```text
1. Data Collection
        ↓
2. Data Quality Audit
        ↓
3. Data Cleaning
        ↓
4. Data Integration
        ↓
5. Feature Engineering
        ↓
6. Exploratory Data Analysis
        ↓
7. Statistical / Behavioral Analysis
        ↓
8. Regression Modeling
        ↓
9. Classification Modeling
        ↓
10. Recommendation Engine
        ↓
11. Model Evaluation
        ↓
12. Streamlit Deployment
        ↓
13. Business Insights
```

---

# 📌 Key Skills Demonstrated

This project demonstrates practical capabilities in:

### 📊 Data Analytics

* Exploratory Data Analysis
* Business KPI analysis
* Trend analysis
* Segmentation
* Customer behavior analysis

### 🐍 Python

* Pandas
* NumPy
* Data preprocessing
* Feature engineering
* Automation

### 📈 Visualization

* Plotly
* Interactive charts
* Dashboard design
* Visual storytelling

### 🤖 Machine Learning

* Regression
* Classification
* Model benchmarking
* Feature engineering
* Model evaluation

### 🎯 Recommendation Systems

* Content-based filtering
* Similarity analysis
* Personalized discovery

### 🌐 Deployment

* Streamlit
* GitHub
* Cloud deployment

### 💼 Business Analytics

* Problem framing
* Insight generation
* Decision support
* Customer engagement analysis

---

# 📈 From Data to Decision

The project's analytical philosophy is:

```text
DATA
 ↓
"What happened?"
 ↓
INSIGHT
 ↓
"Why did it happen?"
 ↓
PREDICTION
 ↓
"What may happen next?"
 ↓
RECOMMENDATION
 ↓
"What should we do?"
```

This transforms the project from a traditional data-analysis exercise into an **end-to-end analytics and decision-support platform**.

---

# 🔮 Future Enhancements

Potential next-generation improvements include:

* 🧠 Advanced personalized recommendation models
* 👥 Traveler segmentation using clustering
* 🌍 Interactive geographic maps
* 📅 Tourism demand forecasting
* ⭐ Sentiment analysis of traveler reviews
* 🔍 SHAP-based model explainability
* 📱 Mobile-friendly experience
* ☁️ Automated data pipelines
* 🔄 Real-time tourism data ingestion
* 🎯 Personalized traveler profiles
* 📊 Advanced KPI monitoring
* 🤖 Conversational tourism analytics assistant

---

# 🏆 Project Impact

### The platform combines three major analytical capabilities:

```text
             TOURISM ANALYTICS
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
   UNDERSTAND    PREDICT     RECOMMEND
       │            │            │
       ▼            ▼            ▼
    Traveler      Rating       Relevant
    Behavior      / Mode       Attractions
       │            │            │
       └────────────┼────────────┘
                    ▼
             BETTER DECISIONS
                    │
                    ▼
          BETTER CUSTOMER EXPERIENCE
```

---

# 💼 Recruiter Snapshot

### What this project demonstrates

| Area              | Demonstrated Capability                        |
| ----------------- | ---------------------------------------------- |
| Data Analytics    | EDA, behavioral analysis, KPI development      |
| Data Preparation  | Cleaning, integration, feature engineering     |
| Visualization     | Interactive dashboards and visual storytelling |
| Machine Learning  | Regression + Classification                    |
| Recommendation    | Content-based attraction recommendations       |
| Business Thinking | Tourism and customer-experience use cases      |
| Deployment        | Streamlit cloud application                    |
| Engineering       | Modular project structure                      |
| Version Control   | Git + GitHub                                   |

---

# 👨‍💻 About the Project

**Project:** Tourism Experience Analytics — Classification, Prediction & Recommendation System

**Focus:** Data Analytics • Machine Learning • Business Intelligence • Recommendation Systems

**Deployment:** Streamlit

**Repository:**
[GitHub Repository](https://github.com/SUMANSANGEET/Tourism-Experience-Analytics_Classification-Prediction-and-Recommendation-System)

**Live Application:**
[Launch Interactive App](https://ida8jsmnknyqdvtkqeafcd.streamlit.app/)

---

# ⭐ If You Find This Project Interesting

Explore the interactive application and repository.

If you find the project useful:

⭐ Star the repository
🍴 Fork the project
💬 Share feedback
🔗 Connect for collaboration

---

## 🚀 Explore the Project

<p align="center">

<a href="https://ida8jsmnknyqdvtkqeafcd.streamlit.app/">
<img src="https://img.shields.io/badge/🚀%20EXPLORE%20LIVE%20APPLICATION-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Explore Live Application"/>
</a>

 

<a href="https://github.com/SUMANSANGEET/Tourism-Experience-Analytics_Classification-Prediction-and-Recommendation-System">
<img src="https://img.shields.io/badge/⭐%20VIEW%20SOURCE%20CODE-181717?style=for-the-badge&logo=github&logoColor=white" alt="View Source Code"/>
</a>

</p>

---

### 📌 Built as an end-to-end analytics project focused on turning tourism data into actionable insights, predictive intelligence, and personalized customer experiences.
