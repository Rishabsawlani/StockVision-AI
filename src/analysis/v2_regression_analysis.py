import pandas as pd
from pathlib import Path
from sklearn.feature_selection import mutual_info_regression


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


# ============================================================
# LOAD DATA
# ============================================================

print("Loading V2 regression dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")


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
# 1. MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing_values = df[feature_columns + [target_column]].isna().sum()

print(missing_values[missing_values > 0])


# ============================================================
# 2. TARGET STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("TARGET STATISTICS")
print("=" * 60)

target = df[target_column].dropna()

print(f"Count:  {len(target)}")
print(f"Mean:   {target.mean():.4f}%")
print(f"Median: {target.median():.4f}%")
print(f"Std:    {target.std():.4f}%")
print(f"Min:    {target.min():.4f}%")
print(f"Max:    {target.max():.4f}%")


# ============================================================
# 3. TARGET QUANTILES
# ============================================================

print("\n" + "=" * 60)
print("TARGET QUANTILES")
print("=" * 60)

quantiles = target.quantile(
    [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
)

print(quantiles)


# ============================================================
# 4. TARGET BY STOCK
# ============================================================

print("\n" + "=" * 60)
print("TARGET STATISTICS BY STOCK")
print("=" * 60)

stock_stats = (
    df.groupby("Ticker")[target_column]
    .agg(["count", "mean", "std", "min", "max"])
    .sort_values("mean")
)

print(stock_stats)


# ============================================================
# 5. PEARSON CORRELATION
# ============================================================

print("\n" + "=" * 60)
print("FEATURE vs TARGET CORRELATION")
print("=" * 60)

correlations = (
    df[feature_columns + [target_column]]
    .corr()[target_column]
    .drop(target_column)
    .sort_values(key=abs, ascending=False)
)

print(correlations)


# ============================================================
# 6. MUTUAL INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("MUTUAL INFORMATION WITH TARGET")
print("=" * 60)

# Remove rows containing missing values.
mi_data = df[feature_columns + [target_column]].dropna()

X_mi = mi_data[feature_columns]
y_mi = mi_data[target_column]

mi_scores = mutual_info_regression(
    X_mi,
    y_mi,
    random_state=42
)

mi_results = pd.Series(
    mi_scores,
    index=feature_columns
).sort_values(ascending=False)

print(mi_results)


# ============================================================
# 7. FEATURE CORRELATION
# ============================================================

print("\n" + "=" * 60)
print("HIGH FEATURE CORRELATIONS")
print("=" * 60)

feature_corr = df[feature_columns].corr()

high_corr_pairs = []

for i in range(len(feature_columns)):
    for j in range(i + 1, len(feature_columns)):

        feature_1 = feature_columns[i]
        feature_2 = feature_columns[j]

        correlation = feature_corr.loc[
            feature_1,
            feature_2
        ]

        if abs(correlation) >= 0.75:

            high_corr_pairs.append(
                (
                    feature_1,
                    feature_2,
                    correlation
                )
            )


high_corr_pairs.sort(
    key=lambda x: abs(x[2]),
    reverse=True
)

if high_corr_pairs:

    for feature_1, feature_2, correlation in high_corr_pairs:

        print(
            f"{feature_1:<25} <-> "
            f"{feature_2:<25} : "
            f"{correlation:.4f}"
        )

else:

    print("No feature pairs above |correlation| >= 0.75")


# ============================================================
# 8. BASIC TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("TARGET RANGE DISTRIBUTION")
print("=" * 60)

ranges = pd.cut(
    target,
    bins=[
        float("-inf"),
        -5,
        -2.5,
        -1,
        0,
        1,
        2.5,
        5,
        float("inf")
    ]
)

range_distribution = (
    ranges
    .value_counts()
    .sort_index()
)

range_percentage = (
    range_distribution / len(target) * 100
)

distribution_table = pd.DataFrame({
    "Count": range_distribution,
    "Percentage": range_percentage.round(2)
})

print(distribution_table)


print("\n" + "=" * 60)
print("REGRESSION AUDIT COMPLETE")
print("=" * 60)