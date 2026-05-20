#!/usr/bin/env python3
"""
Build 3 sensor-specific datasets from CSV files in data_raw/

Raw CSV format (each row from logger):
    t, ax, ay, az, gx, gy, gz, mx, my, mz

We will:
- Infer the posture label from the filename:
      supine, prone, side, sit/sitting, unknown/unk
- For EACH file:
    * create sliding windows for:
        - accel: (ax, ay, az)
        - gyro:  (gx, gy, gz)
        - mag:   (mx, my, mz)
- Use sliding windows with:
    WINDOW_SIZE = 100 samples  (~2 seconds at 50 Hz)
    STRIDE      = 50 samples   (50% overlap)
- Save:
    data_processed/dataset_accel.npz
    data_processed/dataset_gyro.npz
    data_processed/dataset_mag.npz

Each .npz contains:
    X: (N, 100, 3)
    y: (N,)   labels in {1..5}
"""

from pathlib import Path
import numpy as np
import csv

SCRIPT_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT    = SCRIPT_DIR.parent
DATA_RAW        = PROJECT_ROOT / "data_raw"
DATA_PROCESSED  = PROJECT_ROOT / "data_processed"

WINDOW_SIZE = 100
STRIDE      = 50   # overlap; you can change if you want

# Map keywords in filenames to numeric labels 1..5
POSTURE_LABELS = {
    "supine":   1,
    "prone":    2,
    "side":     3,   # includes left/right side
    "sitting":  4,
    "sit":      4,
    "unknown":  5,
    "unk":      5,
}


def infer_label_from_filename(path: Path) -> int:
    """
    Infer posture label from filename based on substrings.

    Examples:
      supine_trial1.csv  -> 1
      prone_accel.csv    -> 2
      side_left_1.csv    -> 3
      sitting_gyro.csv   -> 4
      unknown_3.csv      -> 5
    """
    name = path.name.lower()
    for key, lab in POSTURE_LABELS.items():
        if key in name:
            return lab
    raise ValueError(
        f"Could not infer label from filename: {path.name}. "
        f"Expected one of {list(POSTURE_LABELS.keys())} in the name."
    )


def load_all_channels(path: Path) -> np.ndarray:
    """
    Load all 10 columns from a CSV file with rows:

        t, ax, ay, az, gx, gy, gz, mx, my, mz

    Returns:
        data: np.ndarray of shape (T, 10)
    """
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 10:
                continue
            try:
                vals = [float(x) for x in row[:10]]
            except ValueError:
                # header or bad row
                continue
            rows.append(vals)

    if not rows:
        raise ValueError(f"No valid data rows in {path}")

    return np.array(rows, dtype=np.float32)  # (T, 10)


def build_dataset():
    DATA_PROCESSED.mkdir(exist_ok=True, parents=True)

    X_accel, y_accel = [], []
    X_gyro,  y_gyro  = [], []
    X_mag,   y_mag   = [], []

    SENSOR_SLICES = {
        "accel": (1, 2, 3),  # ax, ay, az
        "gyro":  (4, 5, 6),  # gx, gy, gz
        "mag":   (7, 8, 9),  # mx, my, mz
    }

    for csv_path in sorted(DATA_RAW.glob("*.csv")):
        print(f"Reading {csv_path}")
        lab = infer_label_from_filename(csv_path)
        data = load_all_channels(csv_path)   # (T, 10)
        n = data.shape[0]

        if n < WINDOW_SIZE:
            print(f"  Skipping (only {n} samples, need >= {WINDOW_SIZE})")
            continue

        for sensor_name, (i1, i2, i3) in SENSOR_SLICES.items():
            samples = data[:, [i1, i2, i3]]  # (T, 3)

            # sliding windows with stride
            for start in range(0, n - WINDOW_SIZE + 1, STRIDE):
                end = start + WINDOW_SIZE
                window_x = samples[start:end]   # (100,3)

                if sensor_name == "accel":
                    X_accel.append(window_x)
                    y_accel.append(lab)
                elif sensor_name == "gyro":
                    X_gyro.append(window_x)
                    y_gyro.append(lab)
                else:  # "mag"
                    X_mag.append(window_x)
                    y_mag.append(lab)

    def save_npz(name: str, X_list, y_list):
        if not X_list:
            print(f"\nNo windows collected for {name}! Check your data.")
            return
        X = np.stack(X_list, axis=0)
        y = np.array(y_list, dtype=np.int32)
        out_path = DATA_PROCESSED / f"dataset_{name}.npz"
        np.savez(out_path, X=X, y=y)
        print(f"\nSaved {name} dataset to: {out_path}")
        print("  X shape:", X.shape, "y shape:", y.shape)
        labs, counts = np.unique(y, return_counts=True)
        print("  Label distribution:")
        for lab, cnt in zip(labs, counts):
            print(f"    label {lab}: {cnt} windows")

    save_npz("accel", X_accel, y_accel)
    save_npz("gyro",  X_gyro,  y_gyro)
    save_npz("mag",   X_mag,   y_mag)


if __name__ == "__main__":
    build_dataset()