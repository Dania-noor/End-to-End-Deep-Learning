import os
import pandas as pd

from S1_DataIngestion import load_data
from S2_PreProcessing import preprocess_data


def main():

    print("=" * 70)
    print("END-TO-END DEEP LEARNING PIPELINE")
    print("=" * 70)

    # ========================================================
    # STEP 1 - DATA INGESTION
    # ========================================================

    print("\n[1/4] DATA INGESTION")

    df = load_data()

    # ========================================================
    # STEP 2 - DATA PREPROCESSING
    # ========================================================

    print("\n[2/4] DATA PREPROCESSING")

    X_train, X_test, y_train, y_test = preprocess_data(df)

    # ========================================================
    # STEP 3 - CHECK TRAINED MODELS
    # ========================================================

    print("\n[3/4] TRAINED MODELS")

    models = [
        "ANN",
        "CNN",
        "RNN",
        "LSTM",
        "Transfer_Learning"
    ]

    for model in models:

        model_path = os.path.join(
            "models",
            f"{model}.keras"
        )

        if os.path.exists(model_path):
            print(f"✓ {model} model found")

        else:
            print(f"✗ {model} model not found")

    # ========================================================
    # STEP 4 - CHECK RESULTS
    # ========================================================

    print("\n[4/4] RESULTS")

    results_path = "results/model_comparison.csv"

    if os.path.exists(results_path):

        results = pd.read_csv(results_path)

        print("\nFinal Model Comparison:")
        print(results.to_string(index=False))

        best_model = results.loc[
            results["Accuracy"].idxmax()
        ]

        print("\n" + "=" * 70)
        print("BEST MODEL")
        print("=" * 70)

        print("Model:", best_model["Model"])
        print("Accuracy:", f"{best_model['Accuracy'] * 100:.2f}%")
        print("F1 Score:", f"{best_model['F1 Score'] * 100:.2f}%")

    else:

        print("Results file not found.")

    print("\n" + "=" * 70)
    print("PIPELINE CHECK COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()