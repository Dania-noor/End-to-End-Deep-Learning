import tensorflow as tf
from tensorflow.keras import layers, models


# ============================================================
# 1. ARTIFICIAL NEURAL NETWORK (ANN)
# ============================================================

def build_ann():

    model = models.Sequential([
        layers.Input(shape=(128, 128, 3)),

        layers.Flatten(),

        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),

        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),

        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# 2. CONVOLUTIONAL NEURAL NETWORK (CNN)
# ============================================================

def build_cnn():

    model = models.Sequential([
        layers.Input(shape=(128, 128, 3)),

        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),

        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),

        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# 3. RECURRENT NEURAL NETWORK (RNN)
# ============================================================

def build_rnn():

    model = models.Sequential([
        layers.Input(shape=(128, 128 * 3)),

        layers.SimpleRNN(64, return_sequences=False),

        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),

        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# 4. LONG SHORT-TERM MEMORY (LSTM)
# ============================================================

def build_lstm():

    model = models.Sequential([
        layers.Input(shape=(128, 128 * 3)),

        layers.LSTM(64),

        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),

        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# 5. TRANSFER LEARNING
# ============================================================

def build_transfer_learning():

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(128, 128, 3),
        include_top=False,
        weights="imagenet"
    )

    # Freeze pretrained layers
    base_model.trainable = False

    model = models.Sequential([
        layers.Input(shape=(128, 128, 3)),

        base_model,

        layers.GlobalAveragePooling2D(),

        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),

        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# TEST ALL MODELS
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("MODEL BUILDING")
    print("=" * 60)

    models_dict = {
        "ANN": build_ann(),
        "CNN": build_cnn(),
        "RNN": build_rnn(),
        "LSTM": build_lstm(),
        "Transfer Learning": build_transfer_learning()
    }

    for name, model in models_dict.items():

        print("\n" + "=" * 60)
        print(name)
        print("=" * 60)

        model.summary()