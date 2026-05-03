# House Price Prediction with Feature Engineering and Advanced ML

An end-to-end regression project on the Kaggle House Prices dataset — covering feature engineering, preprocessing pipelines, model comparison, hyperparameter tuning, and residual analysis.

---

## Project Overview

This project applies the complete ML regression workflow to predict house sale prices, with a focus on feature engineering and model benchmarking.

**Workflow:** EDA → Feature Engineering → Preprocessing → Model Training → Tuning → Evaluation

---

## Feature Engineering

New features created from existing columns:

| Feature | Description |
|---------|-------------|
| TotalSF | Total square footage (basement + 1st + 2nd floor) |
| TotalBath | Combined full and half bathrooms |
| HouseAge | Years since construction at time of sale |
| RemodAge | Years since last remodel |
| IsNew | Whether house was sold in year it was built |
| HasGarage | Binary flag for garage presence |
| HasPool | Binary flag for pool presence |
| HasFireplace | Binary flag for fireplace presence |
| HasBsmt | Binary flag for basement presence |

---

## Models Trained

| Model | CV RMSE (log scale) |
|-------|-------------------|
| Linear Regression | ~0.168 |
| Decision Tree | ~0.207 |
| Random Forest | ~0.144 |
| XGBoost (tuned) | ~0.135 |

XGBoost with GridSearchCV tuning achieved the lowest RMSE.

---

## ML Pipeline

```
Raw Data
   ↓
Feature Engineering (9 new features)
   ↓
ColumnTransformer
   ├── Numeric: Median Imputation → Standard Scaling
   └── Categorical: Mode Imputation → One-Hot Encoding
   ↓
Model Training (Linear Regression / Decision Tree / Random Forest / XGBoost)
   ↓
5-Fold Cross Validation
   ↓
GridSearchCV Tuning (XGBoost)
   ↓
Residual Analysis + Feature Importance
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas & NumPy | Data manipulation and feature engineering |
| Matplotlib & Seaborn | Visualizations |
| Scikit-learn | Pipelines, preprocessing, and baseline models |
| XGBoost | Best performing model |

---

## Project Structure

```
house-price-prediction/
│
├── main.py                     # Main script
├── train.csv                   # Training dataset
├── test.csv                    # Test dataset
├── sample_submission.csv       # Submission format
├── data_description.txt        # Feature descriptions
└── README.md
```

---

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/mahrukhmobin/house-price-prediction.git

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost

# 3. Run the script
python main.py
```

---

## Learnings

- Creating meaningful features from raw data using domain knowledge
- Applying log transformation to handle skewed target variables
- Comparing regression models using cross-validated RMSE
- Tuning XGBoost hyperparameters with GridSearchCV
- Interpreting residual plots and feature importance charts

---

*Built by [Mahrukh Mobin](https://github.com/mahrukhmobin) — Computer Engineering Student @ UET Lahore*
