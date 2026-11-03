import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.data.data_processor import load_stock_data
from src.features.feature_engineer import create_features


def prepare_analysis_data(ticker):

    # Load validated stock data
    stock_data = load_stock_data(ticker)

    # Create all engineered features
    data = create_features(stock_data)

    # Create next-day return
    data["Target_Return_1D"] = (
        (data["Close"].shift(-1) - data["Close"])
        / data["Close"]
    ) * 100

    # Create classification target
    # 1 = next day goes UP
    # 0 = next day goes DOWN
    data["Target"] = (data["Target_Return_1D"] > 0).astype(int)

    # Remove rows containing NaN values
    data = data.dropna().copy()

    return data


def create_3class_target(data, threshold):

    data = data.copy()

    # Start every row as NEUTRAL
    data["Target_3Class"] = 1

    # DOWN
    data.loc[
        data["Target_Return_1D"] < -threshold,
        "Target_3Class"
    ] = 0

    # UP
    data.loc[
        data["Target_Return_1D"] > threshold,
        "Target_3Class"
    ] = 2

    return data


if __name__ == "__main__":

    ticker = "TCS.NS"

    data = prepare_analysis_data(ticker)

    # print("\nData Analysis Dataset Created!")

    # print("\nDataset Shape:")
    # print(data.shape)

    # print("\nColumns:")
    # print(data.columns.tolist())

    # print("\nTarget Distribution:")
    # print(data["Target"].value_counts())

    # print("\nTarget Distribution (%):")
    # print(data["Target"].value_counts(normalize=True) * 100)


    # print("\nFeature Statistics by Target:")

    # feature_columns = data.columns.drop(
    #     ["Target_Return_1D", "Target"]
    # )

    # feature_means = data.groupby("Target")[feature_columns].mean().T

    # feature_means.columns = ["DOWN", "UP"]

    # feature_means["Difference"] = (
    #     feature_means["UP"] - feature_means["DOWN"]
    # )

    # feature_means["Absolute_Difference"] = (
    #     feature_means["Difference"].abs()
    # )

    # feature_means = feature_means.sort_values(
    #     by="Absolute_Difference",
    #     ascending=False
    # )

    # print(feature_means)


    # Effect size measures how strongly a feature separates UP from DOWN.
    # It compares the difference between the two group means with their natural variation.
    # Larger absolute values mean stronger separation; values close to 0 mean heavy overlap.

    # down_data = data[data["Target"] == 0]
    # up_data = data[data["Target"] == 1]

    # effect_sizes = {}

    # for feature in feature_columns:

    #     down_mean = down_data[feature].mean()
    #     up_mean = up_data[feature].mean()

    #     down_std = down_data[feature].std()
    #     up_std = up_data[feature].std()

    #     # Pooled standard deviation
    #     pooled_std = np.sqrt(
    #         (down_std ** 2 + up_std ** 2) / 2
    #     )

    #     # Cohen's d
    #     effect_size = (up_mean - down_mean) / pooled_std

    #     effect_sizes[feature] = effect_size


    # effect_size_df = pd.DataFrame(
    #     list(effect_sizes.items()),
    #     columns=["Feature", "Effect_Size"]
    # )

    # effect_size_df["Absolute_Effect_Size"] = (
    #     effect_size_df["Effect_Size"].abs()
    # )

    # effect_size_df = effect_size_df.sort_values(
    #     by="Absolute_Effect_Size",
    #     ascending=False
    # )

    # print("\nEffect Size by Feature:")
    # print(effect_size_df.to_string(index=False))


    # Analyze the actual next-day returns before converting them into UP/DOWN.
    # This tells us whether many of our labels come from extremely small price movements.
    # Very small returns may behave more like noise than meaningful market movements.

    # print("\nNext-Day Return Statistics:")

    # print(
    #     data["Target_Return_1D"].describe(
    #         percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    #     )
    # )


    # print("\nReturns Close to Zero:")

    # for threshold in [0.05, 0.10, 0.20, 0.50, 1.00]:

    #     count = (
    #         data["Target_Return_1D"].abs() < threshold
    #     ).sum()

    #     percentage = count / len(data) * 100

    #     print(
    #         f"|Return| < {threshold:.2f}%: "
    #         f"{count} rows ({percentage:.2f}%)"
    #     )


    import matplotlib.pyplot as plt

    # plt.figure(figsize=(10, 6))

    # plt.hist(
    #     data["Target_Return_1D"],
    #     bins=50
    # )

    # plt.axvline(
    #     0,
    #     linestyle="--"
    # )

    # plt.xlabel("Next-Day Return (%)")
    # plt.ylabel("Frequency")
    # plt.title("Distribution of Next-Day Returns")

    # plt.show()


    # classifying up ,down and neutral on basis of different thresholds 
    # print("\nTarget Distribution at Different Thresholds:")

    # for threshold in [0.25, 0.50, 0.75, 1.00]:

    #     down = (data["Target_Return_1D"] < -threshold).sum()

    #     neutral = (
    #         data["Target_Return_1D"].abs() <= threshold
    #     ).sum()

    #     up = (data["Target_Return_1D"] > threshold).sum()

    #     total = len(data)

    #     print(f"\nThreshold: ±{threshold:.2f}%")

    #     print(
    #         f"DOWN:    {down} "
    #         f"({down / total * 100:.2f}%)"
    #     )

    #     print(
    #         f"NEUTRAL: {neutral} "
    #         f"({neutral / total * 100:.2f}%)"
    #     )

    #     print(
    #         f"UP:      {up} "
    #         f"({up / total * 100:.2f}%)"
        # )

    for threshold in [0.25, 0.50]:

        test_data = create_3class_target(
            data,
            threshold
        )

        print(f"\nThreshold: ±{threshold}%")

        print(
            test_data["Target_3Class"]
            .value_counts()
            .sort_index()
        )

        print(
            test_data["Target_3Class"]
            .value_counts(normalize=True)
            .sort_index() * 100
        )