import serial
import time
import os
import sys
import select

# ---------------- CONFIG ----------------
PORT = '/dev/tty.usbmodem1101'   # change if needed
BAUD = 115200
SAVE_DIR = os.path.expanduser('/Users/kniks2502/Downloads/Chikku CEN/posture_project/data')
FILENAME = 'imu_log_with_labels.csv'   # change per trial if you want

OUT_PATH = os.path.join(SAVE_DIR, FILENAME)

# Exact header your ML code expects
CANONICAL_HEADER = "time_ms,ax,ay,az,pitch,roll,dotS,dotL,label"

# ---------------- INIT ----------------
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"\nConnecting to {PORT} at {BAUD} baud...")
ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)  # let Arduino reset

print(f"✅ Connected. Logging IMU data to:\n{OUT_PATH}")
print("Press ENTER to stop logging.\n")

# ---------------- MAIN LOOP ----------------
with open(OUT_PATH, "w", encoding="utf-8", buffering=1) as f:
    # Always write our own clean header
    f.write(CANONICAL_HEADER + "\n")
    print(CANONICAL_HEADER)

    while True:
        line = ser.readline()
        if line:
            try:
                s = line.decode("utf-8", errors="replace").strip()
            except Exception as e:
                print("Decode error:", e)
                continue

            if not s:
                continue

            # Show raw line for debugging (optional)
            # print("RAW:", repr(s))

            # Skip comments from Arduino like "# IMU posture logger..."
            if s.startswith("#"):
                print(s)
                continue

            # Skip Arduino's own header if it sends one
            if s.lower().startswith("time_ms"):
                # ignore, we already wrote the header
                print(s)
                continue

            # Try to parse as CSV data line
            parts = [p.strip() for p in s.split(",")]

            # Expect 9 fields: time_ms, ax, ay, az, pitch, roll, dotS, dotL, label
            if len(parts) == 9:
                clean_line = ",".join(parts)
                f.write(clean_line + "\n")
                print(clean_line)
            else:
                # Not a valid data line, ignore
                # Uncomment below if you want to see what gets skipped:
                # print("SKIP:", s)
                continue

        # Check if ENTER pressed to stop
        r, _, _ = select.select([sys.stdin], [], [], 0)
        if r:
            _ = sys.stdin.readline()
            print("\n🛑 Logging stopped by user.")
            break

ser.close()
print("Serial port closed.")
print(f"✅ Data saved to: {OUT_PATH}")
