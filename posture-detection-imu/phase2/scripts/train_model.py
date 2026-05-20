#!/usr/bin/env python3
"""
Train a 5-class posture model for ONE sensor type.

Usage:
    python3 train_model.py accel
    python3 train_model.py gyro
    python3 train_model.py mag

Each run:
- Loads dataset_{sensor}.npz from data_processed/
- Labels y in {1..5} (supine, prone, side, sit, unknown)
- Remaps y -> 0..4 for training
- Computes per-channel mean/std on TRAIN SET ONLY for that sensor
- Trains a small 1D-CNN suitable for TFLite Micro
- Saves:
    model/posture_model_{sensor}.keras
    model/norm_stats_{sensor}.npz
"""

from pathlib import Path
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SCRIPT_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT   = SCRIPT_DIR.parent
DATA_PROCESSED = PROJECT_ROOT / "data_processed"
MODEL_DIR      = PROJECT_ROOT / "model"


def load_data(sensor: str):
    path = DATA_PROCESSED / f"dataset_{sensor}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    data = np.load(path)
    X = data["X"].astype("float32")   # (N, 100, 3)
    y = data["y"].astype("int32")     # (N,)
    return X, y


def make_splits(X, y_idx, seed=42):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_idx, test_size=0.30, stratify=y_idx, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=seed
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def build_model(window_size, n_channels, n_classes):
    """
    Small 1D CNN to keep TFLite Micro tensor arena reasonable.
    """
    inputs = keras.Input(shape=(window_size, n_channels))
    x = layers.Conv1D(16, 5, activation="relu")(inputs)
    x = layers.Conv1D(32, 5, activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="posture_model")
    return model


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 train_model.py [accel|gyro|mag]")
        raise SystemExit(1)

    sensor = sys.argv[1]
    if sensor not in {"accel", "gyro", "mag"}:
        print("Sensor must be one of: accel, gyro, mag")
        raise SystemExit(1)

    MODEL_DIR.mkdir(exist_ok=True)

    print(f"[{sensor}] Loading dataset...")
    X, y_orig = load_data(sensor)
    print("X shape:", X.shape, "y shape:", y_orig.shape)

    uniq = np.unique(y_orig)
    print("Original labels:", uniq)

    # y_orig values should be 1..5
    if not np.array_equal(np.sort(uniq), np.array([1, 2, 3, 4, 5])):
        print("WARNING: expected labels 1..5; got:", uniq)

    # Remap to 0..4 for training
    y_idx = y_orig - 1
    n_classes = 5

    window_size = X.shape[1]
    n_channels  = X.shape[2]

    X_train, y_train, X_val, y_val, X_test, y_test = make_splits(X, y_idx)
    print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

    # -------- Normalization (TRAIN only) --------
    mean = X_train.mean(axis=(0, 1))   # (3,)
    std  = X_train.std(axis=(0, 1))    # (3,)
    std[std < 1e-6] = 1.0

    norm_path = MODEL_DIR / f"norm_stats_{sensor}.npz"
    np.savez(norm_path, mean=mean, std=std)
    print(f"[{sensor}] Saved normalization stats to:", norm_path)

    X_train_norm = (X_train - mean) / std
    X_val_norm   = (X_val   - mean) / std
    X_test_norm  = (X_test  - mean) / std

    # -------- Build & train --------
    model = build_model(window_size, n_channels, n_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(model.summary())

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True
        )
    ]

    history = model.fit(
        X_train_norm, y_train,
        validation_data=(X_val_norm, y_val),
        epochs=100,
        batch_size=64,
        callbacks=callbacks,
        verbose=2,
    )

    # -------- Evaluation --------
    test_loss, test_acc = model.evaluate(X_test_norm, y_test, verbose=0)
    print(f"[{sensor}] Test accuracy: {test_acc:.4f}")

    y_pred_probs = model.predict(X_test_norm, verbose=0)
    y_pred_idx = y_pred_probs.argmax(axis=1)
    print(classification_report(y_test, y_pred_idx, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_idx))

    # -------- Save as Keras file only --------
    keras_path = MODEL_DIR / f"posture_model_{sensor}.keras"
    model.save(keras_path)
    print(f"[{sensor}] Saved Keras model to:", keras_path)


if __name__ == "__main__":
    main()