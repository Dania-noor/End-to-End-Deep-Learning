import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from S1_DataIngestion import load_data
from S2_PreProcessing import preprocess_data


MODEL_PATH = "models"


def evaluate_model(model_name, X_test, y_test):

    print("\n" + "=" * 60)
    print(f"EVALUATING: {model_name}")
    print("=" * 60)

    model_file = os.path.join(
        MODEL_PATH,
        f"{model_name}.keras"
    )

    model = tf.keras.models.load_model(model_file)

    # RNN and LSTM need reshaped input
    if model_name in ["RNN", "LSTM"]:

        X_test_input = X_test.reshape(
            X_test.shape[0],
            X_test.shape[1],
            X_test.shape[2] * X_test.shape[3]
        )

    else:
        X_test_input = X_test

    # Model evaluation
    loss, accuracy = model.evaluate(
        X_test_input,
        y_test,
        verbose=0
    )

    # Predictions
    probabilities = model.predict(
        X_test_input,
        verbose=0
    )

    predictions = (probabilities >= 0.5).astype(int).flatten()

    # Metrics
    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    print(f"\nTest Loss:      {loss:.4f}")
    print(f"Test Accuracy:  {accuracy:.4f}")
    print(f"Precision:      {precision:.4f}")
    print(f"Recall:         {recall:.4f}")
    print(f"F1 Score:       {f1:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Cat", "Dog"],
            zero_division=0
        )
    )

    return {
        "Model": model_name,
        "Test Loss": loss,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }


def main():

    print("=" * 60)
    print("MODEL EVALUATION PIPELINE")
    print("=" * 60)

    # Load data
    df = load_data()

    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(df)

    results = []

    model_names = [
        "ANN",
        "CNN",
        "RNN",
        "LSTM",
        "Transfer_Learning"
    ]

    for model_name in model_names:

        result = evaluate_model(
            model_name,
            X_test,
            y_test
        )

        results.append(result)

    # Create comparison DataFrame
    results_df = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print("FINAL MODEL COMPARISON")
    print("=" * 60)

    print(
        results_df.to_string(
            index=False
        )
    )

    # Save results
    os.makedirs("results", exist_ok=True)

    results_df.to_csv(
        "results/model_comparison.csv",
        index=False
    )

    print("\nResults saved to:")
    print("results/model_comparison.csv")


if __name__ == "__main__":
    main()