# IMU-Based Posture Detection System

An end-to-end embedded posture classification system using an **Arduino Nano 33 BLE Sense Rev2** and neural networks trained in TensorFlow/Keras. The system detects five static human postures — **Supine, Prone, Side, Sitting/Standing, and Unknown** — from real-time IMU sensor data.

The project was developed in two phases: an offline classification pipeline (Phase 1) and a fully on-device TinyML deployment with sensor-agnostic inference (Phase 2).

> **Course:** Embedded Systems — Arizona State University  
> **Author:** [Aishwarya Hareesh Rao](https://aishwaryarao-portfolio.vercel.app/)  
> **Date:** Fall 2025

---

## System Overview

```
┌──────────────────────┐      Serial / USB        ┌──────────────────────────┐
│  Arduino Nano 33 BLE │ ◄──────────────────────► │   Python Base Station    │
│  Sense Rev2          │                          │                          │
│                      │   Phase 1: raw CSV log   │   • Serial logger        │
│  • BMI270 accel/gyro │ ───────────────────────► │   • NN training (Keras)  │
│  • BMM150 mag        │                          │   • Offline prediction   │
│                      │   Phase 2: PRED,label,p  │                          │
│  • TFLite Micro      │ ───────────────────────► │   • Sensor selector UI   │
│    on-device infer.  │                          │   • Real-time display    │
└──────────────────────┘                          └──────────────────────────┘
```

---

## Phase 1 — Offline Neural Network Classification

**Goal:** Collect IMU data via serial logging, train a feedforward neural network in Python, and evaluate posture classification accuracy across multiple activation functions.

### How It Works

1. The Arduino firmware reads accelerometer data at ~50 Hz and computes derived features: `pitch`, `roll`, `dotS` (Z-axis alignment), and `dotL` (X-axis alignment).
2. A rule-based label (1–5) is assigned on-board for ground-truth annotation.
3. Data is logged over serial to CSV files — one per posture, multiple trials each.
4. A Python training script (`posture_nn.py`) loads all CSVs, applies z-score normalization + Gaussian noise augmentation (σ = 0.02), and trains a two-hidden-layer dense network (128 → 64 neurons) with three activation function variants: **Sigmoid**, **Tanh**, and **ReLU**.
5. Each model is evaluated with accuracy curves, loss curves, confusion matrices, and per-class metrics.

### Results

| Activation | Validation Accuracy | Test Accuracy |
|------------|-------------------- |---------------|
| Sigmoid    | 99.97 %             | ~99.97 %      |
| Tanh       | 99.98 %             | ~99.98 %      |
| ReLU       | 99.99 %             | ~99.99 %      |

All five classes achieved near-perfect precision and recall. ReLU converged fastest and was selected as the best model.

### Key Files

```
phase1/
├── arduino/
│   └── posture_logger.ino        # IMU data logger + rule-based labeling
├── python/
│   ├── posture_nn.py             # Train NN (Sigmoid / Tanh / ReLU)
│   ├── predict_posture.py        # CLI inference on CSV or single sample
│   └── arduino_logger.py         # Serial data logger (Python side)
└── report/
    ├── Posture_Detection_Report.pdf
    ├── acc_relu.png / loss_relu.png      # Training curves
    ├── confusion_matrix_counts_sigmoid.png
    └── activation_results.csv
```

---

## Phase 2 — On-Device TinyML Inference (Sensor-Agnostic)

**Goal:** Deploy posture classification directly on the microcontroller using TensorFlow Lite for Microcontrollers, with separate models for accelerometer, gyroscope, and magnetometer — selectable at runtime.

### How It Works

1. A 9-axis IMU logger streams raw `ax, ay, az, gx, gy, gz, mx, my, mz` at 50 Hz over serial.
2. A Python pipeline (`build_dataset.py`) creates sliding-window datasets per sensor modality: windows of **100 samples × 3 channels** with 50 % overlap.
3. A lightweight 1D-CNN (`train_model.py`) is trained per sensor type:
   - Conv1D(16, 5) → Conv1D(32, 5) → GlobalAveragePooling → Dense(32) → Softmax(5)
4. Models are converted to TFLite and exported as C byte arrays (`convert_to_tflite.py`) for embedding in firmware.
5. The inference sketch (`posture_inference.ino`) loads the selected model, collects a 100-sample window, normalizes it with training-set statistics, runs inference, and returns the predicted label + confidence over serial.
6. A Python base station (`base_station.py`) provides a CLI for sensor selection and displays live predictions.

### Architecture

```
Input (100, 3)
  │
  ├── Conv1D  ─  16 filters, kernel 5, ReLU
  ├── Conv1D  ─  32 filters, kernel 5, ReLU
  ├── GlobalAveragePooling1D
  ├── Dense(32, ReLU)
  └── Dense(5, Softmax)

Parameters: ~6 K  │  TFLite size: ~19–65 KB  │  Arena: 32 KB
Inference latency: < 150 ms per window on Cortex-M4
```

### Results

- Test accuracy across all three sensor modalities exceeded **80 %**.
- Accelerometer models performed strongest; magnetometer models were most orientation-robust.
- "Side" vs. "Unknown" was the hardest boundary due to inherent ambiguity.
- Real-time inference matched offline performance in controlled conditions.

### Key Files

```
phase2/
├── arduino/
│   ├── arduino_imu_logger/
│   │   └── arduino_imu_logger.ino   # 9-axis raw data logger
│   └── posture_inference/
│       └── posture_inference.ino     # On-device TFLite inference
├── scripts/
│   ├── build_dataset.py              # Sliding-window dataset builder
│   ├── train_model.py                # Train 1D-CNN per sensor
│   ├── convert_to_tflite.py          # Keras → TFLite → C header
│   ├── compute_norm.py               # Print normalization stats
│   ├── logger_stop_on_enter.py       # Serial CSV logger
│   └── base_station.py               # Real-time prediction UI
└── report/
    ├── Project4.pdf
    └── main.tex
```

---

## Hardware

- **Arduino Nano 33 BLE Sense Rev2**
  - Cortex-M4F @ 64 MHz, 256 KB RAM
  - BMI270 (accelerometer + gyroscope)
  - BMM150 (magnetometer)
- USB serial connection to host PC

## Software Dependencies

| Component | Stack |
|-----------|-------|
| Firmware  | Arduino IDE, `Arduino_BMI270_BMM150`, `ArduTFLite` |
| Training  | Python 3.10+, TensorFlow / Keras, scikit-learn, NumPy, pandas, matplotlib |
| Serial I/O| pyserial |

## Quick Start

```bash
# Phase 1 — Train offline model
cd phase1/python
python posture_nn.py                       # trains all 3 activations
python predict_posture.py --csv ../data/Trial1_Supine.csv

# Phase 2 — Build dataset, train, convert, deploy
cd phase2/scripts
python build_dataset.py                    # creates .npz per sensor
python train_model.py accel                # train accelerometer model
python convert_to_tflite.py accel          # generate .tflite + .h
# Flash posture_inference.ino to the board, then:
python base_station.py                     # live predictions
```

---

## Posture Classes

| Label | Posture  | Description |
|-------|----------|-------------|
| 1     | Supine   | Lying face up |
| 2     | Prone    | Lying face down |
| 3     | Side     | Lying on left or right side |
| 4     | Sitting / Standing | Upright orientation |
| 5     | Unknown  | Transitional or unrecognized movement |

---

## License

This project was developed for academic coursework at Arizona State University.

## Contact

**Aishwarya Hareesh Rao**  
[Portfolio](https://aishwaryarao-portfolio.vercel.app/) · [LinkedIn](https://www.linkedin.com/in/aishwaryahrao) · [GitHub](https://github.com/aishwaryahrao20)
