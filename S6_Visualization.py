import os
import pandas as pd
import matplotlib.pyplot as plt


def create_visualizations():

    # Load model comparison results
    results_path = "results/model_comparison.csv"

    df = pd.read_csv(results_path)

    os.makedirs("results/graphs", exist_ok=True)

    # ========================================================
    # 1. Accuracy Comparison
    # ========================================================

    plt.figure(figsize=(10, 6))

    plt.bar(
        df["Model"],
        df["Accuracy"]
    )

    plt.title("Model Accuracy Comparison")
    plt.xlabel("Model")
    plt.ylabel("Test Accuracy")
    plt.ylim(0, 1)

    plt.xticks(rotation=20)

    plt.tight_layout()

    plt.savefig(
        "results/graphs/accuracy_comparison.png"
    )

    plt.show()

    # ========================================================
    # 2. Precision, Recall and F1 Score
    # ========================================================

    metrics = [
        "Precision",
        "Recall",
        "F1 Score"
    ]

    x = range(len(df))

    width = 0.25

    plt.figure(figsize=(12, 6))

    plt.bar(
        [i - width for i in x],
        df["Precision"],
        width,
        label="Precision"
    )

    plt.bar(
        x,
        df["Recall"],
        width,
        label="Recall"
    )

    plt.bar(
        [i + width for i in x],
        df["F1 Score"],
        width,
        label="F1 Score"
    )

    plt.title("Precision, Recall and F1 Score Comparison")

    plt.xlabel("Model")

    plt.ylabel("Score")

    plt.xticks(
        x,
        df["Model"],
        rotation=20
    )

    plt.ylim(0, 1)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "results/graphs/metrics_comparison.png"
    )

    plt.show()

    # ========================================================
    # 3. Test Loss Comparison
    # ========================================================

    plt.figure(figsize=(10, 6))

    plt.bar(
        df["Model"],
        df["Test Loss"]
    )

    plt.title("Test Loss Comparison")

    plt.xlabel("Model")

    plt.ylabel("Test Loss")

    plt.xticks(rotation=20)

    plt.tight_layout()

    plt.savefig(
        "results/graphs/loss_comparison.png"
    )

    plt.show()

    print("=" * 60)
    print("VISUALIZATION COMPLETED")
    print("=" * 60)

    print("\nGraphs saved in:")

    print("results/graphs/")


if __name__ == "__main__":
    create_visualizations()