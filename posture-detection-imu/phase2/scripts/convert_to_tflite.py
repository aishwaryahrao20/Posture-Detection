#!/usr/bin/env python3
"""
Convert a sensor-specific Keras model to TFLite and generate a C header.

Usage:
    python3 convert_to_tflite.py accel
    python3 convert_to_tflite.py gyro
    python3 convert_to_tflite.py mag

Assumed structure:

  project_root/
    model/
      posture_model_accel.keras
      posture_model_gyro.keras
      posture_model_mag.keras

Outputs for each sensor:
    model/posture_model_{sensor}.tflite
    model/posture_model_{sensor}_data.h
"""

from pathlib import Path
import sys
import textwrap
import tensorflow as tf

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODEL_DIR    = PROJECT_ROOT / "model"


def to_c_array(data: bytes) -> str:
    # 12 bytes per line for readability
    hex_bytes = [f"0x{b:02x}" for b in data]
    lines = []
    line = []
    for i, hb in enumerate(hex_bytes):
        line.append(hb)
        if (i + 1) % 12 == 0:
            lines.append(", ".join(line))
            line = []
    if line:
        lines.append(", ".join(line))
    return ",\n  ".join(lines)


def convert_one(sensor: str):
    if sensor not in {"accel", "gyro", "mag"}:
        raise SystemExit("Sensor must be accel, gyro, or mag")

    keras_path = MODEL_DIR / f"posture_model_{sensor}.keras"
    if not keras_path.exists():
        raise FileNotFoundError(
            f"Keras model not found: {keras_path}\n"
            f"Did you run: python3 train_model.py {sensor} ?"
        )

    print(f"[{sensor}] Loading Keras model from: {keras_path}")
    model = tf.keras.models.load_model(keras_path)

    # Get input shape: (None, window_size, n_channels)
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        # Just in case, pick first input
        input_shape = input_shape[0]

    # input_shape is like (None, 100, 3)
    if len(input_shape) != 3:
        raise RuntimeError(f"Unexpected input_shape {input_shape}, expected (None, T, C)")

    _, window_size, n_channels = input_shape
    print(f"[{sensor}] Model input shape: (None, {window_size}, {n_channels})")

    # Build a concrete function for TFLite from a tf.function wrapper
    @tf.function
    def model_fn(x):
        return model(x)

    concrete_fn = model_fn.get_concrete_function(
        tf.TensorSpec(
            shape=(1, window_size, n_channels),
            dtype=tf.float32,
            name="input"
        )
    )

    print(f"[{sensor}] Converting to TFLite from concrete function...")
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn])

    # Keep it pure float to avoid hybrid kernels on TFLite Micro
    converter.optimizations = []
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

    tflite_model = converter.convert()

    tfl_path = MODEL_DIR / f"posture_model_{sensor}.tflite"
    tfl_path.write_bytes(tflite_model)
    print(f"[{sensor}] Wrote TFLite model to: {tfl_path}")

    # ---- Generate C header ----
    array_name = f"posture_model_{sensor}_tflite"
    len_name   = f"posture_model_{sensor}_tflite_len"
    guard      = f"POSTURE_MODEL_{sensor.upper()}_DATA_H"

    array_body = to_c_array(tflite_model)
    array_len  = len(tflite_model)

    header_text = textwrap.dedent(f"""\
        #ifndef {guard}
        #define {guard}

        // Auto-generated from {tfl_path.name}
        const unsigned char {array_name}[] = {{
          {array_body}
        }};

        const unsigned int {len_name} = {array_len};

        #endif  // {guard}
        """)

    header_path = MODEL_DIR / f"posture_model_{sensor}_data.h"
    header_path.write_text(header_text)
    print(f"[{sensor}] Header written to: {header_path}")
    print(f"[{sensor}] Array name: {array_name}, length variable: {len_name} = {array_len}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 convert_to_tflite.py [accel|gyro|mag]")
        raise SystemExit(1)
    sensor = sys.argv[1]
    convert_one(sensor)


if __name__ == "__main__":
    main()