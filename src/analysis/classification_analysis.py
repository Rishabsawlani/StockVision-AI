import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay
)

from pathlib import Path
import sys


def evaluate_classification_model(y_test, y_pred, y_prob):
    """
    Evaluate a binary classification model.

    Parameters:
        y_test: Actual target values
        y_pred: Predicted class labels
        y_prob: Predicted probability for the positive class

    Returns:
        Dictionary containing classification metrics.
    """

    # Overall percentage of correct predictions
    accuracy = accuracy_score(y_test, y_pred)

    # How often predicted UP was actually UP
    precision = precision_score(y_test, y_pred)

    # How many actual UP days the model successfully detected
    recall = recall_score(y_test, y_pred)

    # Harmonic mean of precision and recall
    f1 = f1_score(y_test, y_pred)

    # Measures ranking ability using prediction probabilities
    roc_auc = roc_auc_score(y_test, y_prob)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "ROC_AUC": roc_auc,
        "Confusion_Matrix": cm
    }


if __name__ == "__main__":

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.append(str(PROJECT_ROOT))

    from src.models.logistic_regression import train_logistic_regression

    ticker = "TCS.NS"

    (
        model,
        X_test_scaled,
        y_test,
        y_pred,
        feature_names
    ) = train_logistic_regression(ticker)

    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]

    print("\nThreshold Analysis:")

    for threshold in thresholds:

        threshold_predictions = (
            y_prob >= threshold
        ).astype(int)

        accuracy = accuracy_score(
            y_test,
            threshold_predictions
        )

        precision = precision_score(
            y_test,
            threshold_predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            threshold_predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            threshold_predictions,
            zero_division=0
        )

        predicted_up = threshold_predictions.sum()

        print(
            f"Threshold: {threshold:.2f} | "
            f"Accuracy: {accuracy:.4f} | "
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f} | "
            f"F1: {f1:.4f} | "
            f"Predicted UP: {predicted_up}"
        )

    metrics = evaluate_classification_model(
    y_test,
    y_pred,
    y_prob
    )

    print("\nClassification Evaluation Results:")

    print(f"Accuracy:  {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall:    {metrics['Recall']:.4f}")
    print(f"F1 Score:  {metrics['F1_Score']:.4f}")
    print(f"ROC-AUC:   {metrics['ROC_AUC']:.4f}")

    print("\nConfusion Matrix:")
    print(metrics["Confusion_Matrix"])

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["DOWN", "UP"]
        )
    )

    ConfusionMatrixDisplay(
        confusion_matrix=metrics["Confusion_Matrix"],
        display_labels=["DOWN", "UP"]
    ).plot()

    plt.title("Logistic Regression Confusion Matrix")
    plt.show()

    print("\nPrediction Distribution:")

    print("Actual DOWN:", (y_test == 0).sum())
    print("Actual UP:", (y_test == 1).sum())

    print("Predicted DOWN:", (y_pred == 0).sum())
    print("Predicted UP:", (y_pred == 1).sum())


    print("\nPredicted UP Probability:")

    print("Minimum:", y_prob.min())
    print("Maximum:", y_prob.max())
    print("Mean:", y_prob.mean())
    print("Median:", np.median(y_prob))

    print("\nRandom Forest Feature Importance:")

    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    })

    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=False
    )

    print(feature_importance.to_string(index=False))