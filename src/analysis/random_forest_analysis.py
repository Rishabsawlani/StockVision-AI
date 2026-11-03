import numpy as np
import pandas as pd

from pathlib import Path
import sys

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.models.random_forest import train_tuned_random_forest

def evaluate_random_forest(y_test, y_pred, y_prob):

    # Overall accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Treat all three classes equally
    macro_precision = precision_score(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0
    )

    # Give more weight to classes with more actual samples
    weighted_f1 = f1_score(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        average="weighted",
        zero_division=0
    )

    print(f"Accuracy:         {accuracy:.4f}")
    print(f"Macro Precision:  {macro_precision:.4f}")
    print(f"Macro Recall:     {macro_recall:.4f}")
    print(f"Macro F1:         {macro_f1:.4f}")
    print(f"Weighted F1:      {weighted_f1:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1, 2],
            target_names=["DOWN", "NEUTRAL", "UP"],
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1, 2]
        )
    )


def compare_thresholds(ticker):

    # Thresholds we want to compare
    thresholds = [0.25, 0.35, 0.40, 0.50, 0.60, 0.75]

    # Store evaluation metrics for each threshold
    results = []

    for threshold in thresholds:

        print("\n" + "=" * 50)
        print(f"Evaluating threshold: ±{threshold:.2f}%")
        print("=" * 50)

        # Train the model using this threshold
        (
            model,
            X_test_scaled,
            y_test,
            y_pred,
            y_prob,
            feature_names
        ) = train_tuned_random_forest(
            ticker,
            threshold=threshold
        )

        # Keep your existing evaluation code below this point

        # Evaluate predictions against actual labels
        accuracy = accuracy_score(y_test, y_pred)

        macro_f1 = f1_score(
            y_test,
            y_pred,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0
        )

        weighted_f1 = f1_score(
            y_test,
            y_pred,
            labels=[0, 1, 2],
            average="weighted",
            zero_division=0
        )

        print(f"Accuracy:    {accuracy:.4f}")
        print(f"Macro F1:    {macro_f1:.4f}")
        print(f"Weighted F1: {weighted_f1:.4f}")

        print("\nClassification Report:")
        print(
            classification_report(
                y_test,
                y_pred,
                labels=[0, 1, 2],
                target_names=["DOWN", "NEUTRAL", "UP"],
                zero_division=0
            )
        )

        print("\nConfusion Matrix:")
        print(
            confusion_matrix(
                y_test,
                y_pred,
                labels=[0, 1, 2]
            )
        )

        results.append({
            "Threshold": threshold,
            "Accuracy": accuracy,
            "Macro_F1": macro_f1,
            "Weighted_F1": weighted_f1
        })

    results_df = pd.DataFrame(results)

    print("\nThreshold Comparison Summary:")
    print(results_df.to_string(index=False))

    return results_df


def show_confusion_matrix(y_test, y_pred):

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["DOWN", "UP"],
            zero_division=0
        )
    )

    plt.figure(figsize=(6, 5))

    plt.imshow(cm)

    plt.title("Random Forest Confusion Matrix")

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.xticks(
        [0, 1],
        ["DOWN", "UP"]
    )

    plt.yticks(
        [0, 1],
        ["DOWN", "UP"]
    )

    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.colorbar()

    plt.show()


if __name__ == "__main__":

    ticker = "TCS.NS"

    (
        model,
        X_test_scaled,
        y_test,
        y_pred,
        y_prob,
        feature_names
    ) = train_tuned_random_forest(
        ticker , threshold= 0.25
    )

    print("\nRandom Forest Training Successful!")

    print("\nTesting Data Shape:")
    print(X_test_scaled.shape)

    evaluate_random_forest(
        y_test,
        y_pred,
        y_prob
    )

    # show_confusion_matrix(
    #     y_test,
    #     y_pred
    # )

    # print("\nPrediction Distribution:")

    # print("Actual DOWN:", (y_test == 0).sum())
    # print("Actual UP:", (y_test == 1).sum())

    # print("Predicted DOWN:", (y_pred == 0).sum())
    # print("Predicted UP:", (y_pred == 1).sum())


    # print("\nPredicted UP Probability:")

    # print("Minimum:", y_prob.min())
    # print("Maximum:", y_prob.max())
    # print("Mean:", y_prob.mean())
    # print("Median:", np.median(y_prob))

    print("\nTuned Random Forest Feature Importance:")

    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    })

    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=False
    )

    print(feature_importance.to_string(index=False))

    print("\nProbability Distribution:")

    print(
        pd.Series(y_prob).describe()
    )


    threshold_results = compare_thresholds(ticker)