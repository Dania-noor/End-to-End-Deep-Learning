import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


def preprocess_data(df):

    images = []
    labels = []

    print("=" * 50)
    print("DATA PREPROCESSING")
    print("=" * 50)

    print("Loading images...")

    for _, row in df.iterrows():

        try:
            image = Image.open(row["filepath"])

            # Resize image
            image = image.resize((128, 128))

            # Convert image to array
            image = np.array(image)

            # Make sure image has 3 color channels
            if image.ndim == 2:
                image = np.stack((image,) * 3, axis=-1)

            # Keep only RGB channels
            if image.shape[-1] == 4:
                image = image[:, :, :3]

            images.append(image)
            labels.append(row["label"])

        except Exception as e:
            print("Error loading:", row["filepath"])
            print(e)

    # Convert to NumPy arrays
    X = np.array(images, dtype="float32")
    y = np.array(labels)

    # Normalize pixel values from 0-255 to 0-1
    X = X / 255.0

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nPreprocessing completed!")

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    print("\nTraining data:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

    print("\nTesting data:")
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    print("\nPixel value range:")
    print("Minimum:", X.min())
    print("Maximum:", X.max())

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    from S1_DataIngestion import load_data

    df = load_data()

    preprocess_data(df)