import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

df= pd.read_csv("HousePriceProject/house-prices-advanced-regression-techniques/train.csv")
print(df.info())
print(df.head())
print(df.describe())
print(df.isnull().sum().sort_values(ascending=False).head(15))

plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(df["SalePrice"], kde=True, bins=30)
plt.title("Distribution of SalePrice")
plt.show()

df["TotalSF"]= df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
df["TotalBath"]= df["FullBath"] + 0.5*df["HalfBath"] + df["BsmtFullBath"] + 0.5*df["BsmtHalfBath"]
df["HouseAge"]= df["YrSold"] - df["YearBuilt"]
df["RemodAge"]= df["YrSold"] - df["YearRemodAdd"]
df["IsNew"]= (df["YearBuilt"] == df["YrSold"]).astype(int)
df["HasGarage"]= (df["GarageArea"] > 0).astype(int)
df["HasPool"]= (df["PoolArea"] > 0).astype(int)
df["HasFireplace"]= (df["Fireplaces"] > 0).astype(int)
df["HasBsmt"]= (df["TotalBsmtSF"] > 0).astype(int)

y= df["SalePrice"]
X= df.drop(columns=["SalePrice", "Id"])
y= np.log1p(y)
num_features= X.select_dtypes(include=[np.number]).columns.tolist()
cat_features= X.select_dtypes(include=["object"]).columns.tolist()

numeric_transformer= Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer= Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor= ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features)
    ]
)

models= {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "XGBoost": XGBRegressor(n_estimators=500, learning_rate=0.05, random_state=42, n_jobs=-1)
}

cv= KFold(n_splits=5, shuffle=True, random_state=42)
cv_results= {}

for name, model in models.items():
    pipe= Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    scores= cross_val_score(pipe, X, y, cv=cv, scoring="neg_root_mean_squared_error")
    cv_results[name]= (-scores.mean(), scores.std())

results_df= pd.DataFrame(cv_results, index=["RMSE", "StdDev"]).T
print("\nCross-Validation Results:")
print(results_df)

plt.figure(figsize=(8, 5))
sns.barplot(x=results_df.index, y=results_df["RMSE"], palette="viridis")
plt.title("Model Comparison")
plt.ylabel("CV RMSE")
plt.show()

param_grid= {
    "model__n_estimators": [300, 500],
    "model__max_depth": [3, 5, 7],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__subsample": [0.8, 1.0],
    "model__colsample_bytree": [0.8, 1.0]
}

xgb_pipe= Pipeline(steps=[("preprocessor", preprocessor),
                           ("model", XGBRegressor(random_state=42, n_jobs=-1))])

grid= GridSearchCV(xgb_pipe, param_grid, cv=3,
                    scoring="neg_root_mean_squared_error", n_jobs=-1, verbose=1)
grid.fit(X, y)
print("Best XGB Params:", grid.best_params_)
print("Best XGB RMSE:", -grid.best_score_)
best_xgb= grid.best_estimator_
y_pred= best_xgb.predict(X)
residuals= y - y_pred

plt.figure(figsize=(8, 5))
sns.histplot(residuals, kde=True, bins=30)
plt.title("Residual Distribution (Best XGB)")
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residual Plot (Best XGB)")
plt.show()

feature_names= (
    num_features +
    list(grid.best_estimator_["preprocessor"]
         .named_transformers_["cat"]
         .named_steps["onehot"]
         .get_feature_names_out(cat_features))
)

importances= best_xgb.named_steps["model"].feature_importances_
feat_imp= pd.DataFrame({"Feature": feature_names, "Importance": importances})
feat_imp.sort_values(by="Importance", ascending=False, inplace=True)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feat_imp.head(15))
plt.title("Top 15 Feature Importances")
plt.tight_layout()
plt.show()
