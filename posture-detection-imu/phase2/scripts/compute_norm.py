#!/usr/bin/env python3
"""
Print normalization stats for a given sensor.

Usage:
    python3 compute_norm.py accel
    python3 compute_norm.py gyro
    python3 compute_norm.py mag

Reads:
    model/norm_stats_{sensor}.npz
"""

from pathlib import Path
import sys
import numpy as np

VALID_SENSORS = {"accel", "gyro", "mag"}

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODEL_BASE   = PROJECT_ROOT / "model"


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 compute_norm.py [accel|gyro|mag]")
        sys.exit(1)

    sensor = sys.argv[1].lower()
    if sensor not in VALID_SENSORS:
        print(f"Invalid sensor '{sensor}'. Use one of {VALID_SENSORS}.")
        sys.exit(1)

    norm_path = MODEL_BASE / f"norm_stats_{sensor}.npz"
    if not norm_path.exists():
        raise FileNotFoundError(
            f"Could not find {norm_path}. "
            f"Run train_model.py {sensor} first."
        )

    stats = np.load(norm_path)
    mean = stats["mean"]
    std  = stats["std"]

    print(f"[{sensor}] Channel-wise mean:", mean)
    print(f"[{sensor}] Channel-wise std: ", std)

    mean_str = ", ".join(f"{v:.8f}f" for v in mean)
    std_str  = ", ".join(f"{v:.8f}f" for v in std)
    print("\nC-style arrays:\n")
    print(f"float kMean_{sensor}[3] = {{{mean_str}}};")
    print(f"float kStd_{sensor}[3]  = {{{std_str}}};")


if __name__ == "__main__":
    main()