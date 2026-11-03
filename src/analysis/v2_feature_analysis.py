from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif


# ============================================================
# PROJECT SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "v2_features.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

data = pd.read_csv(DATA_PATH)

data["Date"] = pd.to_datetime(data["Date"])

print("=" * 70)
print("STOCKVISION-AI V2 FEATURE AUDIT")
print("=" * 70)

print(f"\nRows: {len(data):,}")
print(f"Columns: {len(data.columns)}")


# ============================================================
# FEATURE LIST
# ============================================================

FEATURES = [
    "Price_vs_MA30",
    "Volatility_20",
    "ATR_Percent",
    "Volume_Ratio_20",
    "RSI_14",

    "Nifty_Return",
    "Nifty_Volatility_20",
    "Market_Breadth",

    "Sector_Return",
    "Sector_Volatility_20",
    "Sector_Breadth",

    "Stock_vs_Market",
    "Stock_vs_Sector"
]


# ============================================================
# 1. MISSING VALUE CHECK
# ============================================================

print("\n" + "=" * 70)
print("1. MISSING VALUES")
print("=" * 70)

missing = data[FEATURES + ["Next_Day_Return", "Target"]].isna().sum()

print(missing[missing > 0])

if missing.sum() == 0:
    print("No missing values found.")


# ============================================================
# 2. TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("2. TARGET DISTRIBUTION")
print("=" * 70)

target_counts = data["Target"].value_counts().sort_index()

target_percent = (
    data["Target"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)

target_summary = pd.DataFrame({
    "Count": target_counts,
    "Percentage": target_percent.round(2)
})

target_summary.index = [
    "DOWN",
    "NEUTRAL",
    "UP"
]

print(target_summary)


# ============================================================
# 3. FEATURE CORRELATION
# ============================================================

print("\n" + "=" * 70)
print("3. FEATURE CORRELATION")
print("=" * 70)

correlation_matrix = data[FEATURES].corr()

# Find highly correlated feature pairs

high_correlations = []

for i in range(len(FEATURES)):

    for j in range(i + 1, len(FEATURES)):

        feature_a = FEATURES[i]
        feature_b = FEATURES[j]

        correlation = correlation_matrix.loc[
            feature_a,
            feature_b
        ]

        if abs(correlation) >= 0.70:

            high_correlations.append({
                "Feature_A": feature_a,
                "Feature_B": feature_b,
                "Correlation": correlation
            })


if high_correlations:

    high_corr_df = pd.DataFrame(
        high_correlations
    ).sort_values(
        "Correlation",
        key=abs,
        ascending=False
    )

    print(high_corr_df.to_string(index=False))

else:

    print(
        "No feature pairs with "
        "|correlation| >= 0.70."
    )


# ============================================================
# 4. MUTUAL INFORMATION WITH TARGET
# ============================================================

print("\n" + "=" * 70)
print("4. MUTUAL INFORMATION WITH TARGET")
print("=" * 70)

# Remove rows containing missing values
# before calculating mutual information.

mi_data = data[
    FEATURES + ["Target"]
].dropna()

X = mi_data[FEATURES]
y = mi_data["Target"]

mi_scores = mutual_info_classif(
    X,
    y,
    random_state=42
)

mi_summary = pd.DataFrame({
    "Feature": FEATURES,
    "Mutual_Information": mi_scores
}).sort_values(
    "Mutual_Information",
    ascending=False
)

print(
    mi_summary.to_string(index=False)
)


# ============================================================
# 5. FEATURE STATISTICS BY TARGET CLASS
# ============================================================

print("\n" + "=" * 70)
print("5. FEATURE MEANS BY TARGET CLASS")
print("=" * 70)

class_means = (
    data
    .groupby("Target")[FEATURES]
    .mean()
    .T
)

class_means.columns = [
    "DOWN",
    "NEUTRAL",
    "UP"
]

print(
    class_means.to_string()
)


# ============================================================
# 6. FEATURE STANDARD DEVIATION BY TARGET
# ============================================================

print("\n" + "=" * 70)
print("6. FEATURE STANDARD DEVIATION BY TARGET")
print("=" * 70)

class_std = (
    data
    .groupby("Target")[FEATURES]
    .std()
    .T
)

class_std.columns = [
    "DOWN",
    "NEUTRAL",
    "UP"
]

print(
    class_std.to_string()
)


# ============================================================
# 7. SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FEATURE AUDIT COMPLETED")
print("=" * 70)