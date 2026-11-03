import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor


# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v2_regression_features.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v2_xgboost_predictions.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading V2 regression dataset...")

df = pd.read_csv(DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"])

print(f"Initial rows: {len(df)}")


# ============================================================
# DEFINE FEATURES AND TARGET
# ============================================================

identifier_columns = [
    "Date",
    "Ticker",
    "Sector"
]

target_column = "Future_Return_5D"

feature_columns = [
    column
    for column in df.columns
    if column not in identifier_columns + [target_column]
]

print("\nFeatures:")
print(feature_columns)

print(f"\nNumber of features: {len(feature_columns)}")


# ============================================================
# REMOVE MISSING VALUES
# ============================================================

df = df.dropna(
    subset=feature_columns + [target_column]
).copy()

print(
    f"\nRows after removing missing values: "
    f"{len(df)}"
)


# ============================================================
# SORT CHRONOLOGICALLY
# ============================================================

df = df.sort_values(
    "Date"
).reset_index(drop=True)


# ============================================================
# TIME-BASED TRAIN / TEST SPLIT
# ============================================================

unique_dates = sorted(
    df["Date"].unique()
)

split_index = int(
    len(unique_dates) * 0.80
)

train_dates = unique_dates[:split_index]
test_dates = unique_dates[split_index:]


train_df = df[
    df["Date"].isin(train_dates)
].copy()

test_df = df[
    df["Date"].isin(test_dates)
].copy()


# ============================================================
# SEPARATE FEATURES AND TARGET
# ============================================================

X_train = train_df[feature_columns]
y_train = train_df[target_column]

X_test = test_df[feature_columns]
y_test = test_df[target_column]


print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

print(f"Training rows: {len(train_df)}")
print(f"Testing rows:  {len(test_df)}")

print(
    f"Training period: "
    f"{train_df['Date'].min().date()} "
    f"to "
    f"{train_df['Date'].max().date()}"
)

print(
    f"Testing period:  "
    f"{test_df['Date'].min().date()} "
    f"to "
    f"{test_df['Date'].max().date()}"
)


# ============================================================
# XGBOOST REGRESSOR
# ============================================================

model = XGBRegressor(
    objective="reg:squarederror",

    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,

    subsample=0.8,
    colsample_bytree=0.8,

    random_state=42,
    n_jobs=-1
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining XGBoost Regressor...")

model.fit(
    X_train,
    y_train
)

print("Training complete.")


# ============================================================
# MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(
    X_test
)


# ============================================================
# REGRESSION METRICS
# ============================================================

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


print("\n" + "=" * 60)
print("XGBOOST REGRESSION RESULTS")
print("=" * 60)

print(f"MAE:  {mae:.4f}%")
print(f"RMSE: {rmse:.4f}%")
print(f"R²:   {r2:.4f}")


# ============================================================
# MEAN BASELINE
# ============================================================

baseline_prediction = y_train.mean()

baseline_pred = np.full(
    len(y_test),
    baseline_prediction
)

baseline_mae = mean_absolute_error(
    y_test,
    baseline_pred
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        baseline_pred
    )
)

baseline_r2 = r2_score(
    y_test,
    baseline_pred
)


print("\n" + "=" * 60)
print("MEAN BASELINE")
print("=" * 60)

print(
    f"Baseline prediction: "
    f"{baseline_prediction:.4f}%"
)

print(f"MAE:  {baseline_mae:.4f}%")
print(f"RMSE: {baseline_rmse:.4f}%")
print(f"R²:   {baseline_r2:.4f}")


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_df = test_df[
    [
        "Date",
        "Ticker",
        "Sector",
        target_column
    ]
].copy()

prediction_df[
    "Predicted_Return_5D"
] = y_pred


prediction_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# DISPLAY SAMPLE PREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

print(
    prediction_df.head(10)
)


# ============================================================
# OUTPUT LOCATION
# ============================================================

print("\nPredictions saved to:")

print(OUTPUT_PATH)