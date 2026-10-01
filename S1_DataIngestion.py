import os
import pandas as pd


def load_data():

    data_path = "data"

    files = []

    # Cats
    cats_path = os.path.join(data_path, "cats_set")

    for file in os.listdir(cats_path):

        if file.endswith(".jpg"):
            files.append({
                "filename": file,
                "filepath": os.path.join(cats_path, file),
                "label": 0
            })

    # Dogs
    dogs_path = os.path.join(data_path, "dogs_set")

    for file in os.listdir(dogs_path):

        if file.endswith(".jpg"):
            files.append({
                "filename": file,
                "filepath": os.path.join(dogs_path, file),
                "label": 1
            })

    df = pd.DataFrame(files)

    print("=" * 50)
    print("DATA INGESTION")
    print("=" * 50)

    print("Total images:", len(df))

    print("\nFirst 5 images:")
    print(df.head())

    print("\nClass distribution:")
    print(df["label"].value_counts())

    return df


if __name__ == "__main__":
    load_data()