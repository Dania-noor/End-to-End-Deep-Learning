import os
import numpy as np
import tensorflow as tf

from S1_DataIngestion import load_data
from S2_PreProcessing import preprocess_data
from S3_ModelBuilding import (
    build_ann,
    build_cnn,
    build_rnn,
    build_lstm,
    build_transfer_learning
)


# ============================================================
# SETTINGS
# ============================================================

EPOCHS = 10
BATCH_SIZE = 32

MODEL_PATH = "models"

os.makedirs(MODEL_PATH, exist_ok=True)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_model(model, X_train, y_train, model_name):

    print("\n" + "=" * 60)
    print(f"TRAINING: {model_name}")
    print("=" * 60)

    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        verbose=1
    )

    # Save trained model
    model.save(
        os.path.join(MODEL_PATH, f"{model_name}.keras")
    )

    print(f"\n{model_name} training completed!")
    print(f"Model saved: models/{model_name}.keras")

    return history


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main():

    print("=" * 60)
    print("MODEL TRAINING PIPELINE")
    print("=" * 60)

    # Load dataset
    df = load_data()

    # Preprocess dataset
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # --------------------------------------------------------
    # ANN
    # --------------------------------------------------------

    ann_model = build_ann()

    ann_history = train_model(
        ann_model,
        X_train,
        y_train,
        "ANN"
    )

    # --------------------------------------------------------
    # CNN
    # --------------------------------------------------------

    cnn_model = build_cnn()

    cnn_history = train_model(
        cnn_model,
        X_train,
        y_train,
        "CNN"
    )

    # --------------------------------------------------------
    # RNN
    # --------------------------------------------------------

    X_train_rnn = X_train.reshape(
        X_train.shape[0],
        X_train.shape[1],
        X_train.shape[2] * X_train.shape[3]
    )

    rnn_model = build_rnn()

    rnn_history = train_model(
        rnn_model,
        X_train_rnn,
        y_train,
        "RNN"
    )

    # --------------------------------------------------------
    # LSTM
    # --------------------------------------------------------

    lstm_model = build_lstm()

    lstm_history = train_model(
        lstm_model,
        X_train_rnn,
        y_train,
        "LSTM"
    )

    # --------------------------------------------------------
    # TRANSFER LEARNING
    # --------------------------------------------------------

    transfer_model = build_transfer_learning()

    transfer_history = train_model(
        transfer_model,
        X_train,
        y_train,
        "Transfer_Learning"
    )

    print("\n" + "=" * 60)
    print("ALL MODELS TRAINED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()