#include <Arduino.h>
#include <Arduino_BMI270_BMM150.h>
#include <ArduTFLite.h>

#include "posture_model_accel_data.h"
#include "posture_model_gyro_data.h"
#include "posture_model_mag_data.h"

extern const unsigned char posture_model_accel_tflite[];
extern const unsigned int  posture_model_accel_tflite_len;
extern const unsigned char posture_model_gyro_tflite[];
extern const unsigned int  posture_model_gyro_tflite_len;
extern const unsigned char posture_model_mag_tflite[];
extern const unsigned int  posture_model_mag_tflite_len;

// -------------------- MODEL / INPUT SHAPE --------------------
const int kWindowSize  = 100;
const int kNumChannels = 3;
const int kNumClasses  = 5; // 1:Supine, 2:Prone, 3:Side, 4:Sit, 5:Unknown

// Label map: index 0..4 -> 1..5
const int kLabelMap[kNumClasses] = {1, 2, 3, 4, 5};

// -------- Normalization (from norm_stats_*.npz) --------
// These live in FLASH because of const

// accel
const float kMean_accel[3] = {0.10608104f, 0.26659432f, 0.15548389f};
const float kStd_accel[3]  = {2.19351053f, 1.46535981f, 2.25823331f};

// gyro
const float kMean_gyro[3] = {-0.02273244f, 0.10630315f, -0.09019888f};
const float kStd_gyro[3]  = {14.65891838f, 10.85831642f, 11.78242683f};

// mag
const float kMean_mag[3] = {28.47187805f, 6.90082836f, -36.48045349f};
const float kStd_mag[3]  = {29.12936401f, 33.29433823f, 41.29140472f};

// Active pointers to the normalization stats for the current sensor
const float* kMean = kMean_accel;
const float* kStd  = kStd_accel;

// -------------------- BUFFERS --------------------
float window_buf[kWindowSize][kNumChannels];

// Tensor arena for TFLite (float model needs ~47 KB, so we give 48 KB)
constexpr int kTensorArenaSize = 32 * 1024;
alignas(16) uint8_t tensorArena[kTensorArenaSize];

int currentModelSensor = 0;

// -------------------- UTILS --------------------
void printMenu() {
  Serial.println();
  Serial.println("Choose sensor for prediction:");
  Serial.println("  '1' = accelerometer (ax, ay, az)");
  Serial.println("  '2' = gyroscope     (gx, gy, gz)");
  Serial.println("  '3' = magnetometer  (mx, my, mz)");
  Serial.println("  'q' = quit / ignore");
  Serial.print(">> ");
}

// -------------------- IMU SETUP --------------------
void setupIMU() {
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1) {
      delay(1000);
    }
  }
  Serial.println("IMU ready");
}

// -------------------- LOAD MODEL FOR SENSOR --------------------
// 1 = accel, 2 = gyro, 3 = mag
void loadModelForSensor(int sensor_code) {
  if (sensor_code == currentModelSensor) return;

  const unsigned char* modelData = nullptr;

  if (sensor_code == 1) {
    // from posture_model_accel_data.h
    modelData = posture_model_accel_tflite;
    kMean = kMean_accel;
    kStd  = kStd_accel;
  } else if (sensor_code == 2) {
    // from posture_model_gyro_data.h
    modelData = posture_model_gyro_tflite;
    kMean = kMean_gyro;
    kStd  = kStd_gyro;
  } else if (sensor_code == 3) {
    // from posture_model_mag_data.h
    modelData = posture_model_mag_tflite;
    kMean = kMean_mag;
    kStd  = kStd_mag;
  } else {
    Serial.println("loadModelForSensor: invalid sensor code");
    return;
  }

  bool ok = modelInit(modelData, tensorArena, kTensorArenaSize);
  if (!ok) {
    Serial.println("modelInit() failed! (tensor arena may be too small)");
    while (1) {
      delay(1000);
    }
  }

  currentModelSensor = sensor_code;
  Serial.print("Model loaded for sensor ");
  Serial.println(sensor_code);
}

// -------------------- READ WINDOW --------------------
bool readWindow(int sensor_code) {
  int count = 0;
  unsigned long start = millis();

  while (count < kWindowSize) {
    float x = 0.0f, y = 0.0f, z = 0.0f;
    bool got = false;

    if (sensor_code == 1) {
      if (IMU.accelerationAvailable()) {
        IMU.readAcceleration(x, y, z);
        got = true;
      }
    } else if (sensor_code == 2) {
      if (IMU.gyroscopeAvailable()) {
        IMU.readGyroscope(x, y, z);
        got = true;
      }
    } else if (sensor_code == 3) {
      IMU.readMagneticField(x, y, z);  // mag field read is always OK
      got = true;
    }

    if (got) {
      window_buf[count][0] = x;
      window_buf[count][1] = y;
      window_buf[count][2] = z;
      count++;
      delay(20); // ~50 Hz
    } else {
      delay(1);
    }

    if ((count == 0) && (millis() - start > 5000)) {
      Serial.println("No data from selected sensor (timeout).");
      return false;
    }
  }
  return true;
}

// -------------------- COPY + NORMALIZE --------------------
void fillModelInputFromWindow() {
  int idx = 0;
  for (int i = 0; i < kWindowSize; ++i) {
    for (int c = 0; c < kNumChannels; ++c) {
      float v = window_buf[i][c];
      float norm = (v - kMean[c]) / kStd[c];
      modelSetInput(norm, idx);
      idx++;
    }
  }
}

// -------------------- RUN INFERENCE & PRINT --------------------
int runInferenceAndPrint() {
  if (!modelRunInference()) {
    Serial.println("modelRunInference() failed!");
    return -1;
  }

  Serial.print("PROBS,");
  int   best_idx = 0;
  float best_val = modelGetOutput(0);

  for (int i = 0; i < kNumClasses; ++i) {
    float v = modelGetOutput(i);
    Serial.print(v, 4);
    if (i < kNumClasses - 1) Serial.print(",");
    if (v > best_val) {
      best_val = v;
      best_idx = i;
    }
  }
  Serial.println();

  int label = kLabelMap[best_idx];

  Serial.print("PRED,");
  Serial.print(label);
  Serial.print(",");
  Serial.println(best_val, 4);

  return label;
}

// -------------------- SETUP & LOOP --------------------
void setup() {
  Serial.begin(115200);
  while (!Serial) {
    ; // wait for USB Serial to be ready
  }

  Serial.println("Starting posture inference (3 sensor-specific models)...");
  setupIMU();
  printMenu();
}

void loop() {
  if (!Serial.available()) {
    return;
  }

  char cmd = Serial.read();

  // ignore whitespace / newlines
  if (cmd == '\r' || cmd == '\n' || cmd == ' ' || cmd == '\t') {
    return;
  }

  if (cmd == 'q' || cmd == 'Q') {
    Serial.println("Ignoring command. Use 1/2/3 to run prediction.");
    printMenu();
    return;
  }

  int sensor_code = 0;
  if      (cmd == '1') sensor_code = 1;
  else if (cmd == '2') sensor_code = 2;
  else if (cmd == '3') sensor_code = 3;
  else {
    Serial.println("Unknown command. Use '1', '2', or '3'.");
    printMenu();
    return;
  }

  loadModelForSensor(sensor_code);

  Serial.print("Collecting window from sensor ");
  Serial.println(sensor_code);

  if (!readWindow(sensor_code)) {
    Serial.println("Failed to read window.");
    printMenu();
    return;
  }

  fillModelInputFromWindow();
  runInferenceAndPrint();

  // After one prediction, show the menu again (like the Python loop)
  printMenu();
}