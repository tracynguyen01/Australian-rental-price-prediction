# 🏠 Australian Rental Price Prediction

An end-to-end data analytics and ML project that explores the Australian rental market, identifies rental pricing patterns, and predicts weekly rent based on property characteristics.

---

## 🚀 Live Dashboard

🔗 Streamlit App: [Launch Prediction App](https://australian-rental-price-prediction-fqoxvtlf33qtrlpxrnx7il.streamlit.app/)

🔗 Power BI:
[View Interactive Dashboard](https://app.powerbi.com/view?r=eyJrIjoiOGRmZWM1OTUtMmZjOS00YmFhLTkxMTMtNTNmMzRjNTFiZjliIiwidCI6ImYwMjU1MTVhLTNhNGUtNDJhNC1hYmZkLWI2MjliNWI3NmQ4NSJ9)

---

## 📊 Dashboard Preview

![Dashboard Preview](assets/AU_rental_price_prediction.png)

---

## 📌 Project Overview

Rental prices vary considerably depending on property characteristics such as location, number of bedrooms, furnishing status, property level, and other listing attributes. Understanding these relationships can help property owners, renters, and property-related businesses make more informed pricing decisions.

This project develops an end-to-end rental analytics solution with two main objectives:

1. **Rental Market Analysis** – explore rental listings to identify pricing patterns and compare weekly rents across different property characteristics.
2. **Rental Price Prediction** – develop and evaluate regression models capable of estimating the weekly rent of a property from its available features.

Rather than presenting the ML model in isolation, the project turns the analytical results two user-facing products: an interactive Power BI dashboard for exploring rental market trends and model performance

### Business Value

The solution demonstrates how rental listing data can support practical decision-making.

For **property owners and property managers**, predicted rental prices can provide a data-driven reference when setting or reviewing weekly rent.

For **renters**, market-level analysis can provide useful context for comparing rental prices across different property types and characteristics.

For **property and real-estate businesses**, the analytical workflow can support market monitoring, pricing analysis, and identification of segments where rental prices differ substantially.

The model is intended as a **decision-support tool rather than a replacement for professional property valuation**, particularly because prediction errors become larger for some higher-priced properties.

---

## 🎯 Project Objectives

- Explore the distribution and characteristics of Australian rental listings.
- Identify patterns associated with weekly rental prices.
- Prepare and transform rental data for predictive modelling.
- Build and compare multiple regression approaches.
- Tune model hyperparameters to improve predictive performance.
- Evaluate model generalisation using unseen test data.
- Communicate market and model results through Power BI.
  
---

## ⚙️ Project Workflow

```text
Rental Listing Data
    ↓
Exploratory Data Analysis
    ↓
Data Cleaning & Preparation
    ↓
Feature Engineering
    ↓
Model Development
    ↓
Hyperparameter Tuning
    ↓
Model Evaluation & Comparison
    ↓
Selected KNN Model

```

---

## 🔍 Exploratory Data Analysis

Exploratory analysis was used to understand the rental market before building prediction models.

The analysis examined areas such as:

- weekly rental price distribution;
- rental prices across different numbers of bedrooms;
- furnished, semi-furnished, and unfurnished properties;
- property-level differences;
- suburb-level rental patterns;
- relationships between property characteristics and weekly rent.

The analysis also helped identify unusual and high-priced rental observations that could influence prediction performance.

---

## 🧹 Data Preparation

The modelling workflow included data preparation steps required to make the rental listing data suitable for machine learning.

Key tasks included:

- inspecting and cleaning the dataset;
- preparing numerical and categorical variables;
- transforming features for modelling;
- scaling features where required;
- separating the data into training, validation, and testing sets;
- preparing engineered features for model development and evaluation.

These steps created a consistent modelling dataset while reducing the risk of evaluating models on information already seen during training.

---

## 🤖 Machine Learning

Multiple regression approaches were developed and compared:

- Linear Regression
- ElasticNet Regression
- K-Nearest Neighbours (KNN) Regression

Linear Regression and ElasticNet provided useful linear baselines, while KNN was explored as a non-linear approach capable of predicting prices based on similar properties in the feature space.

### Hyperparameter Tuning

KNN performance was further investigated using:

- RandomizedSearchCV
- GridSearchCV

The tuning process evaluated different values for the number of neighbours and distance metrics.

The selected configuration used:

| Parameter | Selected Value |
|---|---:|
| Model | KNN Regression |
| Number of Neighbours | 40 |
| Distance Metric | Manhattan |
| `p` | 1 |
| Test RMSE | $48.99/week |

The experiments showed that Manhattan-distance KNN models performed consistently well, with models using approximately 30–40 neighbours producing similar test performance.

---

## 🤖 Models Compared
| Model             | RMSE  |
| ----------------- | ----- |
| KNN Regression    | 48.99 |
| ElasticNet        | 54.65 |
| Linear Regression | 54.66 |

✅ KNN Regression achieved the best predictive performance by capturing non-linear rental relationships more effectively than linear models.

---

## 💼 Business Applications

The project demonstrates several potential real-world applications.

### Rental Pricing Support

Property owners and managers can use model estimates as an additional reference point when evaluating an appropriate weekly rental price.

### Market Benchmarking

Users can compare rental prices across property characteristics and market segments to better understand how a listing compares with similar properties.

### Portfolio & Market Analysis

Property-related businesses can use interactive dashboards to monitor rental patterns and identify segments associated with higher or lower weekly rents.

### Decision Support

Combining predictive modelling with interactive visual analytics makes the results more accessible to non-technical users who may not work directly with Python or machine learning models.

---

## 📈 Key Insights
- Larger floor areas generally increase rental prices
- Bedrooms and bathrooms positively correlate with rent
- Seasonal trends influence rental demand and pricing
- Non-linear models outperform traditional linear approaches
  
---
## 🛠️ Tech Stack
| Area | Technologies |
|---|---|
| Programming | Python |
| Data Manipulation | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Visualisation | Matplotlib |
| Model Tuning | GridSearchCV, RandomizedSearchCV |
| Dashboard | Microsoft Power BI |
| Web Application | Streamlit |
| Development | Jupyter Notebook / Google Colab |
| Version Control | Git, GitHub |

---
## 📂 Repository Structure

```text
Australian-rental-price-prediction/
│
├── assets/
│   └── AU_rental_price_prediction.png
│
├── dataset/
│   ├── rental_training.csv
│   ├── rental_validation.csv
│   └── rental_testing.csv
│
├── notebooks/
│   ├── setup_env_&_baseline_experiment.ipynb
│   ├── Linear_experiment.ipynb
│   ├── ElasticNet_experiment.ipynb
│   └── KNN_experiment.ipynb
│
├── streamlit_dashboard/
│   └── app.py
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## ⚠️ Limitations

Several limitations should be considered when interpreting the results:

- Predictions depend on the property characteristics represented in the available dataset.
- Rental markets can change over time, meaning a model trained on historical listings may require retraining as market conditions change.
- Higher-priced properties showed greater prediction errors in the model evaluation.
- KNN predictions depend on similarity to observations available in the training data, which can make unusual properties more difficult to estimate accurately.
- The predicted rental price should therefore be treated as an analytical estimate rather than a formal property valuation.

---

## 🔮 Future Improvements

Potential extensions include:

- incorporating additional location and neighbourhood characteristics;
- adding geographic visualisations;
- introducing time-based rental-market analysis;
- evaluating additional ensemble or boosting models;
- adding model explainability to show which property characteristics contribute most to price estimates;
- monitoring model performance as new rental data becomes available;
- automating model retraining and dashboard refresh workflows.

---

## ▶️ Run Locally

Clone the repository:

```bash
git clone https://github.com/tracynguyen01/Australian-rental-price-prediction.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Streamlit dashboard:

```bash
streamlit run streamlit_dashboard/app.py
```
---
## 👩‍💻 Author
**Ngoc Bao Tran (Tracy) Nguyen**
