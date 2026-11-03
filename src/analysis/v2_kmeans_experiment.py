
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# --------------------------------------------------
# 1. Load V2 dataset
# --------------------------------------------------

DATA_PATH = "data/processed/v2_regression_features.csv"

df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])

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
    "Stock_vs_Sector",
]

TARGET = "Future_Return_5D"

# Keep rows with complete features.
# Target is NOT used to fit K-Means.
valid = df.dropna(subset=FEATURES).copy()

X = valid[FEATURES]

# --------------------------------------------------
# 2. Scale features
# --------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Rows used:", len(valid))
print("Features used:", len(FEATURES))


# --------------------------------------------------
# 3. Test different numbers of clusters
# --------------------------------------------------

results = []

for k in range(2, 7):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    # silhouette score is only valid if there are at least 2 clusters
    # this score represents how well-separated the clusters are
    score = silhouette_score(
        X_scaled,
        labels,
        sample_size=min(5000, len(valid)),
        random_state=42
    )

    sizes = pd.Series(labels).value_counts(normalize=True)

    results.append({
        "K": k,
        "Silhouette": score,
        "Smallest_Cluster_Pct": sizes.min() * 100,
        "Largest_Cluster_Pct": sizes.max() * 100,
        "Inertia": model.inertia_,
    })

    valid[f"Cluster_K{k}"] = labels

results_df = pd.DataFrame(results)

print("\n========== CLUSTER EVALUATION ==========")
print(results_df.round(4).to_string(index=False))


# --------------------------------------------------
# 4. Inspect cluster characteristics for K=3
# --------------------------------------------------

chosen_k = 3

cluster_col = f"Cluster_K{chosen_k}"

print("\n========== CLUSTER SIZES ==========")
print(valid[cluster_col].value_counts().sort_index())

print("\n========== FEATURE PROFILES ==========")
print(
    valid.groupby(cluster_col)[FEATURES]
    .mean()
    .round(4)
    .to_string()
)


# --------------------------------------------------
# 5. Evaluate future returns AFTER clustering
# --------------------------------------------------

evaluation_df = valid.dropna(subset=[TARGET]).copy()

print("\n========== FUTURE 5-DAY RETURNS BY CLUSTER ==========")

summary = evaluation_df.groupby(cluster_col)[TARGET].agg(
    Count="count",
    Mean="mean",
    Median="median",
    Std="std",
)

print(summary.round(4).to_string())

print("\nOverall future-return distribution:")
print(evaluation_df[TARGET].describe().round(4))


# --------------------------------------------------
# 6. Save results
# --------------------------------------------------

valid.to_csv(
    "data/processed/v2_kmeans_results.csv",
    index=False
)

results_df.to_csv(
    "data/processed/v2_kmeans_scores.csv",
    index=False
)

print("\nSaved:")
print("data/processed/v2_kmeans_results.csv")
print("data/processed/v2_kmeans_scores.csv")
