import glob
import os

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

import joblib  # 👈 for saving the scaler


# -------------------- CONFIG --------------------
DATA_DIR = "../data"          # folder where your CSVs live
REPORT_DIR = "../report"      # folder to save plots + models

os.makedirs(REPORT_DIR, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

NOISE_STD = 0.02              # std dev for Gaussian noise (tune this)

# FINAL LABEL MEANING (from Arduino):
# 1 = Supine
# 2 = Prone
# 3 = Side (either left/right)
# 4 = Standing
# 5 = Unknown
NUM_CLASSES = 5

# Human-readable class names (index 0 -> label 1, etc.)
CLASS_NAMES = ["Supine", "Prone", "Side", "Standing", "Unknown"]

EPOCHS = 100                  # EarlyStopping will stop earlier if needed
BATCH_SIZE = 128
ACTIVATIONS = ["sigmoid", "tanh", "relu"]


# -------------------- LOAD DATA --------------------
def load_all_data(data_dir):
    pattern = os.path.join(data_dir, "*.csv")
    files = glob.glob(pattern)
    if not files:
        raise RuntimeError(f"No CSV files found in {data_dir}. Did you save your IMU logs there?")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        required_cols = ["time_ms", "ax", "ay", "az", "pitch", "roll", "dotS", "dotL", "label"]
        for c in required_cols:
            if c not in df.columns:
                raise RuntimeError(f"File {f} is missing column '{c}'")
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)
    data = data.dropna()

    # Ensure labels are integers and in 1..5
    data["label"] = data["label"].astype(int)
    data = data[(data["label"] >= 1) & (data["label"] <= NUM_CLASSES)]

    return data


print("Loading data...")
data = load_all_data(DATA_DIR)
print(f"Loaded {len(data)} samples.")

# Show label distribution
print("\nLabel distribution (raw):")
print(data["label"].value_counts().sort_index())

# Features
feature_cols = ["ax", "ay", "az", "pitch", "roll", "dotS", "dotL"]
X = data[feature_cols].values
y_raw = data["label"].values.astype(int)  # 1..5

# Convert 1..5 -> 0..4
y = y_raw - 1

print("\nUnique label values after shift (0-based):", np.unique(y))

# -------------------- TRAIN / VAL / TEST SPLIT --------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=RANDOM_STATE, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
)

print("\nSplit sizes:")
print("  Train:", X_train.shape[0])
print("  Val  :", X_val.shape[0])
print("  Test :", X_test.shape[0])

# -------------------- SCALING --------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# ✅ Save the fitted scaler for later inference (predict_posture.py)
scaler_path = os.path.join(REPORT_DIR, "scaler.pkl")
joblib.dump(scaler, scaler_path)
print(f"Saved scaler to: {scaler_path}")

# -------------------- TRAIN NOISE AUGMENTATION --------------------
rng = np.random.default_rng(RANDOM_STATE)
noise = rng.normal(0.0, NOISE_STD, X_train.shape)
X_train_noisy = X_train + noise
X_train_use   = X_train_noisy   # you can switch to X_train if you don't want noise

# -------------------- ONE-HOT LABELS --------------------
y_train_cat = to_categorical(y_train, NUM_CLASSES)
y_val_cat   = to_categorical(y_val,   NUM_CLASSES)
y_test_cat  = to_categorical(y_test,  NUM_CLASSES)

# -------------------- MODEL DEFINITION --------------------
def build_model(hidden_activation="relu"):
    model = models.Sequential()
    model.add(layers.Input(shape=(len(feature_cols),)))

    # Dense stack: Dense + BatchNorm + Activation + Dropout
    model.add(layers.Dense(128))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation(hidden_activation))
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(64))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation(hidden_activation))
    model.add(layers.Dropout(0.3))

    # Output layer
    model.add(layers.Dense(NUM_CLASSES, activation="softmax"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train_and_evaluate(activation_name):
    print(f"\n=== Training with activation: {activation_name} ===")
    model = build_model(hidden_activation=activation_name)

    ckpt_path = os.path.join(REPORT_DIR, f"best_model_{activation_name}.keras")

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=1
        ),
        ModelCheckpoint(
            ckpt_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]

    history = model.fit(
        X_train_use, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    val_loss, val_acc = model.evaluate(X_val, y_val_cat, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)

    print(f"\nFinal Validation accuracy ({activation_name}): {val_acc:.4f}")
    print(f"Final Test accuracy       ({activation_name}): {test_acc:.4f}")

    return model, history, (val_acc, test_acc), ckpt_path


# -------------------- TRAIN FOR EACH ACTIVATION --------------------
models_by_act = {}
histories_by_act = {}
results_by_act = {}
ckpt_paths = {}

for act in ACTIVATIONS:
    model, history, accs, ckpt_path = train_and_evaluate(act)
    models_by_act[act] = model
    histories_by_act[act] = history
    results_by_act[act] = accs
    ckpt_paths[act] = ckpt_path

# -------------------- PLOT TRAINING CURVES --------------------
for act, h in histories_by_act.items():
    # Accuracy curve
    plt.figure()
    plt.plot(h.history["accuracy"], label="train_acc")
    plt.plot(h.history["val_accuracy"], label="val_acc")
    plt.title(f"Accuracy vs Epochs ({act})")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    acc_path = os.path.join(REPORT_DIR, f"acc_{act}.png")
    plt.savefig(acc_path, dpi=300)
    plt.close()

    # Loss curve
    plt.figure()
    plt.plot(h.history["loss"], label="train_loss")
    plt.plot(h.history["val_loss"], label="val_loss")
    plt.title(f"Loss vs Epochs ({act})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    loss_path = os.path.join(REPORT_DIR, f"loss_{act}.png")
    plt.savefig(loss_path, dpi=300)
    plt.close()

print("\nActivation results (val_acc, test_acc):")
metrics_summary = []
for act, (val_acc, test_acc) in results_by_act.items():
    print(f"  {act:7s} -> val: {val_acc:.4f}, test: {test_acc:.4f}")
    metrics_summary.append({"activation": act, "val_acc": val_acc, "test_acc": test_acc})

# Save metrics summary to CSV
metrics_df = pd.DataFrame(metrics_summary)
metrics_csv_path = os.path.join(REPORT_DIR, "activation_results.csv")
metrics_df.to_csv(metrics_csv_path, index=False)
print(f"\nSaved activation metrics to: {metrics_csv_path}")

# -------------------- CONFUSION MATRIX FOR BEST ACTIVATION --------------------
# Choose best by validation accuracy
best_act = max(results_by_act.items(), key=lambda kv: kv[1][0])[0]
print(f"\nBest activation based on validation accuracy: {best_act}")

best_model_path = ckpt_paths[best_act]
print(f"Loading best model weights from: {best_model_path}")
best_model = tf.keras.models.load_model(best_model_path)

y_pred_prob = best_model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

cm = confusion_matrix(y_test, y_pred)
print("\nClassification report (test set):")
print(classification_report(y_test, y_pred, digits=4, target_names=CLASS_NAMES))

# Per-class accuracy
per_class_acc = cm.diagonal() / cm.sum(axis=1)
print("Per-class accuracy:")
for idx, acc in enumerate(per_class_acc):
    name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"Class {idx+1}"
    print(f"  {idx+1} ({name}): {acc:.4f}")

# Plot confusion matrix (absolute counts)
plt.figure()
im = plt.imshow(cm, interpolation="nearest")
plt.title(f"Confusion Matrix (Counts) - {best_act}")
plt.colorbar(im)

tick_marks = np.arange(NUM_CLASSES)
class_labels = [f"{i+1}\n{CLASS_NAMES[i]}" for i in range(NUM_CLASSES)]
plt.xticks(tick_marks, class_labels)
plt.yticks(tick_marks, class_labels)
plt.xlabel("Predicted label")
plt.ylabel("True label")

thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j, i, format(cm[i, j], "d"),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black"
        )

plt.tight_layout()
cm_path = os.path.join(REPORT_DIR, f"confusion_matrix_counts_{best_act}.png")
plt.savefig(cm_path, dpi=300)
plt.close()

# Normalized confusion matrix
cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
plt.figure()
im2 = plt.imshow(cm_norm, interpolation="nearest")
plt.title(f"Confusion Matrix (Normalized) - {best_act}")
plt.colorbar(im2)

plt.xticks(tick_marks, class_labels)
plt.yticks(tick_marks, class_labels)
plt.xlabel("Predicted label")
plt.ylabel("True label")

thresh2 = cm_norm.max() / 2.0
for i in range(cm_norm.shape[0]):
    for j in range(cm_norm.shape[1]):
        plt.text(
            j, i, f"{cm_norm[i, j]:.2f}",
            horizontalalignment="center",
            color="white" if cm_norm[i, j] > thresh2 else "black"
        )

plt.tight_layout()
cm_norm_path = os.path.join(REPORT_DIR, f"confusion_matrix_normalized_{best_act}.png")
plt.savefig(cm_norm_path, dpi=300)
plt.close()

print(f"\nSaved training plots, confusion matrices, and metrics in: {REPORT_DIR}")
print(f"Best model ({best_act}) saved at: {best_model_path}")