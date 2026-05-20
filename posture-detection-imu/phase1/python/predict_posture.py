import os
import sys
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from sklearn.metrics import classification_report, confusion_matrix


# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(BASE_DIR, "../report")

DEFAULT_MODEL_PATH = os.path.join(REPORT_DIR, "best_model_relu.keras")
DEFAULT_SCALER_PATH = os.path.join(REPORT_DIR, "scaler.pkl")

# Features used by the trained model (must match training script)
FEATURE_COLS = ["ax", "ay", "az", "pitch", "roll", "dotS", "dotL"]

# Class names corresponding to labels 1..5
CLASS_NAMES = ["Supine", "Prone", "Side", "Standing", "Unknown"]


# ---------------- LOADERS ----------------
def load_model_and_scaler(model_path, scaler_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    print(f"Loading model from:   {model_path}")
    model = tf.keras.models.load_model(model_path)

    print(f"Loading scaler from:  {scaler_path}")
    scaler = joblib.load(scaler_path)

    return model, scaler


# ---------------- CSV PREDICTION ----------------
def predict_on_csv(input_csv, output_csv, model, scaler):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    print(f"\nReading data from: {input_csv}")
    df = pd.read_csv(input_csv)

    # 🔧 Normalize column names: strip spaces
    df.columns = [c.strip() for c in df.columns]

    print("Columns found in CSV:", list(df.columns))

    # Check required feature columns exist
    missing = [col for col in FEATURE_COLS if col not in df.columns]
    if missing:
        raise RuntimeError(
            f"These required columns are missing from {input_csv}: {missing}\n"
            f"Columns actually present: {list(df.columns)}\n"
            f"Make sure your Arduino header is exactly:\n"
            f"  time_ms,ax,ay,az,pitch,roll,dotS,dotL,label"
        )

    X = df[FEATURE_COLS].values

    # Apply same scaling as training
    X_scaled = scaler.transform(X)

    # Predict probabilities and class indices (0..4)
    y_pred_prob = model.predict(X_scaled)
    y_pred_idx = np.argmax(y_pred_prob, axis=1)

    # Convert to labels 1..5
    y_pred_labels = y_pred_idx + 1

    # Map to class names
    y_pred_names = [CLASS_NAMES[i] for i in y_pred_idx]

    # Add prediction columns
    df["pred_label"] = y_pred_labels
    df["pred_name"] = y_pred_names

    # If ground-truth label exists, compute some metrics
    if "label" in df.columns:
        try:
            y_true = df["label"].astype(int).values
            # Filter to valid range if needed
            mask = (y_true >= 1) & (y_true <= len(CLASS_NAMES))
            y_true = y_true[mask]
            y_pred_for_metric = y_pred_labels[mask]

            print("\nClassification report on this CSV (label column present):")
            print(classification_report(
                y_true - 1,             # shift to 0..4
                y_pred_for_metric - 1,  # shift to 0..4
                target_names=CLASS_NAMES,
                digits=4
            ))

            cm = confusion_matrix(y_true - 1, y_pred_for_metric - 1)
            print("Confusion matrix (rows=true, cols=pred):")
            print(cm)
        except Exception as e:
            print("\n⚠️ Could not compute metrics from 'label' column:", e)

    # Save predictions
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"\nSaved predictions to: {output_csv}")
    print("\nPreview (first 10 rows):")
    cols_to_show = ["time_ms"] + FEATURE_COLS + ["pred_label", "pred_name"]
    existing_cols = [c for c in cols_to_show if c in df.columns]
    print(df.head(10)[existing_cols])


# ---------------- SINGLE SAMPLE PREDICTION ----------------
def predict_single_sample(values, model, scaler):
    """
    values: list or tuple in order [ax, ay, az, pitch, roll, dotS, dotL]
    """
    arr = np.array(values, dtype=float).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    probs = model.predict(arr_scaled)
    idx = np.argmax(probs, axis=1)[0]
    label = idx + 1
    name = CLASS_NAMES[idx]
    return label, name, probs[0]


# ---------------- MAIN (CLI) ----------------
def main():
    parser = argparse.ArgumentParser(
        description="Predict posture labels using trained neural network."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to .keras model file (default: {DEFAULT_MODEL_PATH})"
    )
    parser.add_argument(
        "--scaler",
        type=str,
        default=DEFAULT_SCALER_PATH,
        help=f"Path to scaler .pkl file (default: {DEFAULT_SCALER_PATH})"
    )
    parser.add_argument(
        "--csv",
        type=str,
        help="Path to input CSV with columns: time_ms, ax, ay, az, pitch, roll, dotS, dotL[, label]"
    )
    parser.add_argument(
        "--out",
        type=str,
        help="Path to output CSV with predictions. If not set, will save next to input with _predicted suffix."
    )
    parser.add_argument(
        "--sample",
        nargs=7,
        metavar=("ax", "ay", "az", "pitch", "roll", "dotS", "dotL"),
        help="Optional: predict a single sample by passing 7 numeric values directly."
    )

    args = parser.parse_args()

    # Load model + scaler
    model, scaler = load_model_and_scaler(args.model, args.scaler)

    # If single sample provided
    if args.sample is not None:
        print("\nPredicting for single sample:")
        print("  [ax, ay, az, pitch, roll, dotS, dotL] =", args.sample)
        label, name, probs = predict_single_sample(args.sample, model, scaler)
        print(f"\nPredicted label: {label} ({name})")
        print("Class probabilities (for classes 1..5):")
        for i, p in enumerate(probs, start=1):
            print(f"  {i} ({CLASS_NAMES[i-1]}): {p:.4f}")
        return

    # Otherwise, require CSV
    if not args.csv:
        print("\nERROR: You must provide either --csv or --sample.")
        print("Example for CSV:")
        print("  python predict_posture.py --csv ../data/test_trial.csv")
        print("\nExample for single sample:")
        print("  python predict_posture.py --sample 0.01 0.02 0.98 -1.2 3.4 0.95 0.10")
        sys.exit(1)

    input_csv = args.csv

    if args.out:
        output_csv = args.out
    else:
        base, ext = os.path.splitext(input_csv)
        output_csv = base + "_predicted" + ext

    predict_on_csv(input_csv, output_csv, model, scaler)


if __name__ == "__main__":
    main()