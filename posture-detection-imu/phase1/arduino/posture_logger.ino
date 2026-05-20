#include <Arduino.h>
#include <Arduino_BMI270_BMM150.h>

// Sampling at ~50 Hz
const unsigned long SAMPLE_PERIOD_MS = 20;

// Label meanings:
// 1 = Supine (face up)
// 2 = Prone  (face down)
// 3 = Standing (vertical up/down)
// 4 = Side-Right  (horizontal on right edge, X positive)
// 5 = Side-Left   (horizontal on left edge, X negative)

unsigned long lastSampleTime = 0;
bool headerPrinted = false;

// ---------- Math Helpers ----------
void normalize(float &x, float &y, float &z) {
  float n = sqrtf(x*x + y*y + z*z);
  if (n > 0.0f) { x /= n; y /= n; z /= n; }
}

float pitchDeg(float ax, float ay, float az) {
  return atan2f(-ax, sqrtf(ay*ay + az*az)) * 180.0f / PI;
}

float rollDeg(float ax, float ay, float az) {
  return atan2f(ay, az) * 180.0f / PI;
}

void computeDotVectors(float ax, float ay, float az, float &dotS, float &dotL) {
  float nx=ax, ny=ay, nz=az;
  normalize(nx, ny, nz);
  dotS = nz;        // alignment with Z
  dotL = fabsf(nx); // alignment with X
}

// ---------- Classification ----------
uint8_t classifyPosture(float ax, float ay, float az) {
  float nx=ax, ny=ay, nz=az;
  normalize(nx, ny, nz);

  const float HORIZ_THRESH = 0.75f;
  const float VERT_Z_MAX   = 0.5f;
  const float SIDE_THRESH  = 0.75f;

  // SUPINE
  if (nz >  HORIZ_THRESH) return 1;

  // PRONE
  if (nz < -HORIZ_THRESH) return 2;

  // SIDE: horizontal on edge → split by X sign
  if (fabsf(nz) < VERT_Z_MAX && fabsf(nx) > SIDE_THRESH) {
    if (nx > 0) return 4;   // right-side
    else        return 5;   // left-side
  }

  // STANDING (vertical up or down)
  if (fabsf(nz) < VERT_Z_MAX && fabsf(ny) > HORIZ_THRESH) return 3;

  // Default fallback
  return 3;
}

// ---------- Setup ----------
void setup() {
  Serial.begin(115200);
  while (!Serial) {}
  if (!IMU.begin()) {
    Serial.println("ERROR: IMU init failed!");
    while (1) delay(1000);
  }

  Serial.println("# IMU posture logger started");
  Serial.println("# Labels: 1=Supine, 2=Prone, 3=Standing, 4=Side-Right, 5=Side-Left");
  Serial.println("# Format: time_ms, ax, ay, az, pitch, roll, dotS, dotL, label\n");
}

// ---------- Loop ----------
void loop() {
  unsigned long now = millis();
  if (now - lastSampleTime < SAMPLE_PERIOD_MS) return;
  lastSampleTime = now;

  float ax, ay, az;
  if (!IMU.accelerationAvailable()) return;
  IMU.readAcceleration(ax, ay, az);

  float pitch = pitchDeg(ax, ay, az);
  float roll  = rollDeg(ax, ay, az);
  float dotS, dotL;
  computeDotVectors(ax, ay, az, dotS, dotL);

  uint8_t label = classifyPosture(ax, ay, az);

  if (!headerPrinted) {
    Serial.println("time_ms, ax, ay, az, pitch, roll, dotS, dotL, label");
    headerPrinted = true;
  }

  Serial.print(now);      Serial.print(", ");
  Serial.print(ax, 6);    Serial.print(", ");
  Serial.print(ay, 6);    Serial.print(", ");
  Serial.print(az, 6);    Serial.print(", ");
  Serial.print(pitch, 3); Serial.print(", ");
  Serial.print(roll, 3);  Serial.print(", ");
  Serial.print(dotS, 6);  Serial.print(", ");
  Serial.print(dotL, 6);  Serial.print(", ");
  Serial.println(label);
}