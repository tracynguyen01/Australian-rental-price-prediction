# 🏠 Australian Rental Price Prediction

Machine learning dashboard for predicting Australian rental prices using property features, furnishing status, and seasonal rental trends.

---

## 📌 Project Overview

This project explores how machine learning models can estimate rental prices across Australian properties using structured housing data.

The workflow includes:
- Data cleaning & preprocessing
- Feature engineering
- Model training & evaluation
- Interactive Streamlit dashboard deployment

The project compares multiple regression models and identifies the strongest-performing approach for capturing non-linear rental pricing patterns.

---

## 🚀 Live Dashboard

🔗 Streamlit App:  
https://australian-rental-price-prediction-fqoxvtlf33qtrlpxrnx7il.streamlit.app/

---

## 📊 Dashboard Preview

### Main Dashboard
![Dashboard Preview](assests/AU_rental_price_prediction.png)

---

## ⚙️ Machine Learning Pipeline

```text
Data Cleaning
    ↓
Feature Engineering
    ↓
Encoding
    ↓
Feature Scaling
    ↓
Hyperparameter Tuning
    ↓
Model Evaluation
```

---
## 🤖 Models Compared
| Model             | RMSE  |
| ----------------- | ----- |
| KNN Regression    | 48.99 |
| ElasticNet        | 54.65 |
| Linear Regression | 54.66 |

✅ KNN Regression achieved the best predictive performance by capturing non-linear rental relationships more effectively than linear models.

---
## 📈 Key Insights
- Larger floor areas generally increase rental prices
- Bedrooms and bathrooms positively correlate with rent
- Seasonal trends influence rental demand and pricing
- Non-linear models outperform traditional linear approaches
  
---
## 🛠️ Tech Stack
- Python
- Pandas
- Scikit-learn
- Plotly
- Streamlit

---
## 📂 Repository Structure
