#include <Arduino.h>
#include <Arduino_BMI270_BMM150.h>

const unsigned long LOG_INTERVAL_MS = 20; // ~50 Hz
unsigned long lastLog = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial) {}

  if (!IMU.begin()) {
    Serial.println("IMU init failed!");
    while (1);
  }

  Serial.println("t,ax,ay,az,gx,gy,gz,mx,my,mz");
}

void loop() {
  unsigned long now = millis();
  if (now - lastLog < LOG_INTERVAL_MS) return;
  lastLog = now;

  float ax, ay, az, gx, gy, gz, mx, my, mz;

  if (IMU.accelerationAvailable()) IMU.readAcceleration(ax, ay, az);
  if (IMU.gyroscopeAvailable()) IMU.readGyroscope(gx, gy, gz);
  if (IMU.magneticFieldAvailable()) IMU.readMagneticField(mx, my, mz);

  float t = now / 1000.0f;
  Serial.print(t, 3); Serial.print(",");
  Serial.print(ax, 6); Serial.print(","); Serial.print(ay, 6); Serial.print(","); Serial.print(az, 6); Serial.print(",");
  Serial.print(gx, 6); Serial.print(","); Serial.print(gy, 6); Serial.print(","); Serial.print(gz, 6); Serial.print(",");
  Serial.print(mx, 6); Serial.print(","); Serial.print(my, 6); Serial.print(","); Serial.println(mz, 6);
}