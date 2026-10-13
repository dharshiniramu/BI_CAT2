# ============================================================
# AMAZON SALES - ML DEMAND FORECASTING
# Model: Automatic model selection
# Dataset: Amazon.xlsx
# Records used: Full dataset
# ============================================================


# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

import pandas as pd
import numpy as np
from xgboost import XGBRegressor

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

df = pd.read_excel("Amazon.xlsx")

print("Original dataset shape:")
print(df.shape)


# ------------------------------------------------------------
# 3. USE THE FULL DATASET
# ------------------------------------------------------------

print("\nDataset used for project:")
print(df.shape)


# ------------------------------------------------------------
# 4. CHECK COLUMN NAMES
# ------------------------------------------------------------

print("\nColumns:")
print(df.columns.tolist())


# ------------------------------------------------------------
# 5. CONVERT ORDERDATE TO DATETIME
# ------------------------------------------------------------

df["OrderDate"] = pd.to_datetime(
    df["OrderDate"],
    errors="coerce"
)


# Remove rows where date could not be converted
df = df.dropna(subset=["OrderDate"])


# ------------------------------------------------------------
# 6. CHECK MISSING VALUES
# ------------------------------------------------------------

print("\nMissing values:")
print(df.isnull().sum())


# ------------------------------------------------------------
# 7. SORT DATA CHRONOLOGICALLY
# ------------------------------------------------------------

df = df.sort_values(
    by=["ProductID", "OrderDate"]
).reset_index(drop=True)


# ------------------------------------------------------------
# 8. CREATE YEAR AND MONTH
# ------------------------------------------------------------

df["Year"] = df["OrderDate"].dt.year

df["Month"] = df["OrderDate"].dt.month


# ------------------------------------------------------------
# 9. CREATE RETURNED AND CANCELLED INDICATORS
# ------------------------------------------------------------

df["Returned"] = (
    df["OrderStatus"].str.lower() == "returned"
).astype(int)


df["Cancelled"] = (
    df["OrderStatus"].str.lower() == "cancelled"
).astype(int)


# ------------------------------------------------------------
# 10. CREATE EFFECTIVE QUANTITY
# ------------------------------------------------------------
#
# Effective Demand =
# Gross Ordered Units
# - Returned Units
# - Cancelled Units
#
# Therefore:
# Returned / Cancelled quantity = 0
# Other order statuses = Quantity
#
# ------------------------------------------------------------

df["EffectiveQuantity"] = np.where(
    (df["Returned"] == 1) |
    (df["Cancelled"] == 1),
    0,
    df["Quantity"]
)


# ------------------------------------------------------------
# 11. CREATE MONTHLY PRODUCT-LEVEL DATA
# ------------------------------------------------------------

monthly = (
    df.groupby(
        [
            "ProductID",
            "Year",
            "Month"
        ],
        as_index=False
    )
    .agg(
        ProductName=("ProductName", "first"),
        Category=("Category", "first"),
        Brand=("Brand", "first"),
        Country=("Country", "first"),
        Quantity=("EffectiveQuantity", "sum"),

        GrossQuantity=("Quantity", "sum"),

        Price=("UnitPrice", "mean"),

        Discount=("Discount", "mean"),

        ReturnedQuantity=("Quantity", lambda x: 0),

        ReturnRate=("Returned", "mean"),

        CancellationRate=("Cancelled", "mean"),

        SellingDays=("OrderDate", "nunique")
    )
)


# ------------------------------------------------------------
# 12. CALCULATE RETURNED QUANTITY
# ------------------------------------------------------------

returned_quantity = (
    df[df["Returned"] == 1]
    .groupby(
        [
            "ProductID",
            "Year",
            "Month"
        ])["Quantity"]
    .sum()
    .reset_index(name="ReturnedQuantity")
)


monthly = monthly.drop(
    columns=["ReturnedQuantity"]
)


monthly = monthly.merge(
    returned_quantity,
    on=["ProductID", "Year", "Month"],
    how="left"
)


monthly["ReturnedQuantity"] = (
    monthly["ReturnedQuantity"]
    .fillna(0)
)


# ------------------------------------------------------------
# 13. CALCULATE CANCELLED QUANTITY
# ------------------------------------------------------------

cancelled_quantity = (
    df[df["Cancelled"] == 1]

    # Make the target the next calendar month. Missing product-months represent
    # zero observed demand rather than being silently skipped.
    .groupby(
        [
            "ProductID",
            "Year",
            "Month"
        ])["Quantity"]
    .sum()
    .reset_index(name="CancelledQuantity")
)


monthly = monthly.merge(
    cancelled_quantity,
    on=["ProductID", "Year", "Month"],
    how="left"
)


monthly["CancelledQuantity"] = (
    monthly["CancelledQuantity"]
    .fillna(0)
)


# ------------------------------------------------------------
# 14. CREATE DATE COLUMN FOR EACH PRODUCT-MONTH
# ------------------------------------------------------------

monthly["Date"] = pd.to_datetime(
    monthly["Year"].astype(str)
    + "-"
    + monthly["Month"].astype(str)
    + "-01"
)


# ------------------------------------------------------------
# 15. SORT MONTHLY DATA
# ------------------------------------------------------------

monthly = monthly.sort_values(
    by=["ProductID", "Date"]
).reset_index(drop=True)


series_keys = ["ProductID"]


# ------------------------------------------------------------
# 16. CALCULATE SALES VELOCITY
# ------------------------------------------------------------
#
# Project definition:
#
# Sales Velocity =
# Units Sold / Number of Selling Days
#
# ------------------------------------------------------------

monthly["SalesVelocity"] = (
    monthly["Quantity"] /
    monthly["SellingDays"].replace(0, np.nan)
)


monthly["SalesVelocity"] = (
    monthly["SalesVelocity"]
    .fillna(0)
)


# ------------------------------------------------------------
# 17. CREATE PREVIOUS-PERIOD QUANTITY
# ------------------------------------------------------------

monthly["PreviousQuantity"] = (
    monthly
    .groupby(series_keys)["Quantity"]
    .shift(1)
)


monthly["QuantityLag2"] = (
    monthly
    .groupby(series_keys)["Quantity"]
    .shift(2)
)


# ------------------------------------------------------------
# 18. CREATE 3-PERIOD ROLLING AVERAGE
# ------------------------------------------------------------
#
# Important:
# shift(1) is used so the current month's quantity
# is NOT included in its own rolling average.
#
# This prevents data leakage.
#
# ------------------------------------------------------------

monthly["RollingAverage"] = (
    monthly
    .groupby(series_keys)["Quantity"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            window=3,
            min_periods=1
        )
        .mean()
    )
)


# ------------------------------------------------------------
# 19. ADD LEAKAGE-SAFE HISTORY AND SEASONAL FEATURES
# ------------------------------------------------------------

monthly["HistoricalAverage"] = (
    monthly
    .groupby(series_keys)["Quantity"]
    .transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )
)


monthly["SeasonalAverage"] = (
    monthly
    .groupby(["ProductID", "Month"])["Quantity"]
    .transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )
)


monthly["QuantityTrend"] = (
    monthly["PreviousQuantity"] - monthly["QuantityLag2"]
)


monthly["MonthSin"] = np.sin(
    2 * np.pi * monthly["Month"] / 12
)


monthly["MonthCos"] = np.cos(
    2 * np.pi * monthly["Month"] / 12
)


monthly["SeasonalAverage"] = monthly["SeasonalAverage"].fillna(
    monthly["HistoricalAverage"]
)


# ------------------------------------------------------------
# 20. CREATE FUTURE DEMAND TARGET
# ------------------------------------------------------------
#
# The next calendar month's quantity becomes
# the target selected models must predict.
#
# ------------------------------------------------------------

monthly["FutureDemand"] = (
    monthly
    .groupby(series_keys)["Quantity"]
    .shift(-1)
)


# ------------------------------------------------------------
# 21. REMOVE ROWS WHERE TARGET / HISTORY IS UNAVAILABLE
# ------------------------------------------------------------

model_data = monthly.dropna(
    subset=[
        "PreviousQuantity",
        "QuantityLag2",
        "RollingAverage",
        "HistoricalAverage",
        "SeasonalAverage",
        "FutureDemand"
    ]
).copy()


# ------------------------------------------------------------
# 22. DISPLAY MODEL DATA
# ------------------------------------------------------------

print("\nModel dataset shape:")
print(model_data.shape)


print("\nModel data sample:")
print(
    model_data[
        [
            "ProductID",
            "ProductName",
            "Year",
            "Month",
            "Quantity",
            "PreviousQuantity",
            "RollingAverage",
            "SalesVelocity",
            "FutureDemand"
        ]
    ].head(10)
)


# ------------------------------------------------------------
# 23. DEFINE INPUT FEATURES
# ------------------------------------------------------------
#
# These are the features mentioned in the project:
#
# Product
# Category
# Brand
# Country
# Month
# Year
# Previous-period quantity
# Rolling average quantity
# Sales velocity
# Price
# Discount
# Return rate
# Cancellation rate
#
# ------------------------------------------------------------

features = [
    "ProductID",
    "Category",
    "Brand",
    "Country",
    "Month",
    "Year",
    "Quantity",
    "PreviousQuantity",
    "QuantityLag2",
    "RollingAverage",
    "HistoricalAverage",
    "SeasonalAverage",
    "QuantityTrend",
    "MonthSin",
    "MonthCos",
    "SalesVelocity",
    "Price",
    "Discount",
    "ReturnRate",
    "CancellationRate"
]


# ------------------------------------------------------------
# 24. DEFINE TARGET
# ------------------------------------------------------------

target = "FutureDemand"


X = model_data[features]

y = model_data[target]


# ------------------------------------------------------------
# 25. SORT DATA BY DATE
# ------------------------------------------------------------

model_data = model_data.sort_values(
    by="Date"
).reset_index(drop=True)


X = model_data[features]

y = model_data[target]


# ------------------------------------------------------------
# 26. CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
# ------------------------------------------------------------
#
# First 60%  -> Training
# Next 20%   -> Validation/model selection
# Last 20%   -> Testing/final evaluation
#
# We do NOT randomly shuffle because this is
# a demand forecasting problem.
#
# ------------------------------------------------------------

train_end = int(
    len(model_data) * 0.60
)

validation_end = int(
    len(model_data) * 0.80
)

X_train = X.iloc[:train_end]

X_validation = X.iloc[train_end:validation_end]

X_test = X.iloc[validation_end:]


y_train = y.iloc[:train_end]

y_validation = y.iloc[train_end:validation_end]

y_test = y.iloc[validation_end:]


print("\nTraining records:")
print(len(X_train))


print("\nValidation records:")
print(len(X_validation))


print("\nTesting records:")
print(len(X_test))


# ------------------------------------------------------------
# 27. DEFINE CATEGORICAL FEATURES
# ------------------------------------------------------------

categorical_features = [
    "ProductID",
    "Category",
    "Brand",
    "Country"
]


# ------------------------------------------------------------
# 28. DEFINE NUMERICAL FEATURES
# ------------------------------------------------------------

numerical_features = [
    "Month",
    "Year",
    "Quantity",
    "PreviousQuantity",
    "QuantityLag2",
    "RollingAverage",
    "HistoricalAverage",
    "SeasonalAverage",
    "QuantityTrend",
    "MonthSin",
    "MonthCos",
    "SalesVelocity",
    "Price",
    "Discount",
    "ReturnRate",
    "CancellationRate"
]


# ------------------------------------------------------------
# 29. ONE-HOT ENCODING
# ------------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        ),

        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# ------------------------------------------------------------
# 30. COMPARE CANDIDATE MODELS ON VALIDATION DATA
# ------------------------------------------------------------

candidate_models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1
    ),
    "Extra Trees": ExtraTreesRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.04,
        max_depth=2,
        loss="huber",
        random_state=42
    ),
    "Histogram Gradient Boosting": HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42
    ),
    "XGBoost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=2,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )
}


results = []
trained_pipelines = {}

for model_name, model in candidate_models.items():
    feature_pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model)
        ]
    )

    candidate_pipeline = TransformedTargetRegressor(
        regressor=feature_pipeline,
        func=np.log1p,
        inverse_func=np.expm1
    )

    candidate_pipeline.fit(X_train, y_train)
    candidate_pred = candidate_pipeline.predict(X_validation)
    candidate_mae = mean_absolute_error(y_validation, candidate_pred)

    results.append((model_name, candidate_mae))


best_model_name, best_mae = min(results, key=lambda item: item[1])

# Refit the selected model on all data available before the final test period.
best_feature_pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", candidate_models[best_model_name])
    ]
)

pipeline = TransformedTargetRegressor(
    regressor=best_feature_pipeline,
    func=np.log1p,
    inverse_func=np.expm1
)

pipeline.fit(
    X.iloc[:validation_end],
    y.iloc[:validation_end]
)

y_pred = pipeline.predict(X_test)

print("\nCandidate model MAE:")
for model_name, model_mae in sorted(results, key=lambda item: item[1]):
    print(f"{model_name}: {model_mae:.4f}")

print(f"\nSelected model: {best_model_name}")


# ------------------------------------------------------------
# 33. MAKE PREDICTIONS
# ------------------------------------------------------------

y_pred = pipeline.predict(
    X_test
)


# ------------------------------------------------------------
# 34. EVALUATE MODEL
# ------------------------------------------------------------

mae = mean_absolute_error(
    y_test,
    y_pred
)


rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)


r2 = r2_score(
    y_test,
    y_pred
)


# ------------------------------------------------------------
# 35. PRINT MODEL PERFORMANCE
# ------------------------------------------------------------

print("\n===================================")
print(f"{best_model_name.upper()} MODEL PERFORMANCE")
print("===================================")

print(
    f"MAE  : {mae:.4f}"
)

print(
    f"RMSE : {rmse:.4f}"
)

print(
    f"R²   : {r2:.4f}"
)


# ------------------------------------------------------------
# 36. CREATE PREDICTION DATAFRAME
# ------------------------------------------------------------

predictions = model_data.iloc[
    validation_end:
].copy()


predictions["ActualDemand"] = y_test.values

predictions["PredictedDemand"] = y_pred


# ------------------------------------------------------------
# 37. ROUND PREDICTIONS
# ------------------------------------------------------------

predictions["PredictedDemand"] = (
    predictions["PredictedDemand"]
    .round(2)
)


low_demand_limit = predictions["PredictedDemand"].quantile(1 / 3)
high_demand_limit = predictions["PredictedDemand"].quantile(2 / 3)


def classify_demand(demand):
    if demand <= low_demand_limit:
        return "Low"
    if demand <= high_demand_limit:
        return "Medium"
    return "High"


predictions["DemandClassification"] = (
    predictions["PredictedDemand"]
    .apply(classify_demand)
)


print(
    "\nMonthly demand classification limits "
    f"(33rd/67th percentiles): "
    f"{low_demand_limit:.2f} / {high_demand_limit:.2f}"
)

print(
    predictions["DemandClassification"]
    .value_counts()
    .reindex(["Low", "Medium", "High"], fill_value=0)
)


# ------------------------------------------------------------
# 38. DISPLAY PREDICTIONS
# ------------------------------------------------------------

print("\nActual vs Predicted Demand:")

print(
    predictions[
        [
            "ProductID",
            "ProductName",
            "Date",
            "ActualDemand",
            "PredictedDemand"
        ]
    ].head(20)
)


# ------------------------------------------------------------
# 39. SAVE PREDICTIONS
# ------------------------------------------------------------

output_columns = [
    "ProductID",
    "ProductName",
    "Category",
    "Brand",
    "Country",
    "Date",
    "Year",
    "Month",
    "Quantity",
    "PreviousQuantity",
    "RollingAverage",
    "SalesVelocity",
    "Price",
    "Discount",
    "ReturnRate",
    "CancellationRate",
    "ActualDemand",
    "PredictedDemand",
    "DemandClassification"
]


predictions[
    output_columns
].to_csv(
    "demand_predictions.csv",
    index=False
)


print(
    "\nPredictions saved as: demand_predictions.csv"
)


# ------------------------------------------------------------
# 40. CREATE PRODUCT-LEVEL FORECAST SUMMARY
# ------------------------------------------------------------

latest_forecast = (
    predictions
    .sort_values("Date")
    .groupby("ProductID")
    .tail(1)
    .copy()
)


latest_forecast = latest_forecast[
    [
        "ProductID",
        "ProductName",
        "Category",
        "Brand",
        "Country",
        "Date",
        "SalesVelocity",
        "PreviousQuantity",
        "RollingAverage",
        "PredictedDemand",
        "DemandClassification",
        "ReturnRate",
        "CancellationRate"
    ]
]


# ------------------------------------------------------------
# 41. SAVE PRODUCT-LEVEL FORECAST
# ------------------------------------------------------------

latest_forecast.to_csv(
    "product_demand_forecast.csv",
    index=False
)


print(
    "Product-level forecast saved as: "
    "product_demand_forecast.csv"
)


print(
    "Product classification will be saved as: "
    "product_demand_classification.xlsx"
)


# ------------------------------------------------------------
# 41. SAVE PRODUCT CLASSIFICATIONS TO EXCEL
# ------------------------------------------------------------

classification_columns = [
    "ProductID",
    "ProductName",
    "Category",
    "Brand",
    "Country",
    "Date",
    "PredictedDemand",
    "DemandClassification"
]


latest_forecast[classification_columns].to_excel(
    "product_demand_classification.xlsx",
    index=False
)


print(
    "Product classification saved as: "
    "product_demand_classification.xlsx"
)


# ------------------------------------------------------------
# 42. FINAL MESSAGE
# ------------------------------------------------------------

print("\n===================================")
print("ML DEMAND FORECASTING COMPLETED")
print("===================================")

print(
    f"Model       : {best_model_name}"
)

print(
    "Data Used   : Full dataset"
)

print(
    "Target      : Future Demand"
)

print(
    "Features    : Product, Category, Brand, Country,"
    " Month, Year, Quantity History,"
    " Rolling Average, Seasonal Average,"
    " Quantity Trend, Cyclical Month,"
    " Sales Velocity, Price, Discount,"
    " Return Rate, Cancellation Rate"
)

print(
    "Output 1    : demand_predictions.csv"
)

print(
    "Output 2    : product_demand_forecast.csv"
)