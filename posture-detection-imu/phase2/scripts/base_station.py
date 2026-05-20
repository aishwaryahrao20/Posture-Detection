#!/usr/bin/env python3
import serial
import serial.tools.list_ports
import time

LABELS = {
    1: "Supine",
    2: "Prone",
    3: "Side",
    4: "Sitting",
    5: "Unknown",
}

def auto_port():
    """
    Pick a reasonable default serial port.
    Prefer something with 'usbmodem' or 'Nano' in the description.
    """
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("No serial ports found.")
        return None

    for p in ports:
        desc = (p.description or "").lower()
        if "usbmodem" in p.device.lower() or "nano" in desc:
            return p.device

    # fallback: first port
    return ports[0].device

def request_prediction(ser, sensor_code: int):
    """
    Send '1', '2', or '3' to the board and wait for a PRED line.
    """
    ser.reset_input_buffer()
    ser.write(str(sensor_code).encode("utf-8"))
    ser.flush()

    deadline = time.time() + 8
    while time.time() < deadline:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not line:
            continue
        print("RAW:", line)
        if line.startswith("PRED"):
            parts = line.split(",")
            if len(parts) >= 3:
                try:
                    label = int(parts[1])
                    prob  = float(parts[2])
                    print(f"Prediction → {LABELS.get(label, label)} "
                          f"(label={label}, prob={prob:.3f})")
                except ValueError:
                    print("Could not parse PRED line.")
            break

def main():
    port = auto_port()
    if not port:
        return
    print("Using port:", port)

    with serial.Serial(port, 115200, timeout=5) as ser:
        time.sleep(2)
        ser.reset_input_buffer()

        while True:
            print("\nChoose sensor for prediction:")
            print(" 1 = Accelerometer (ax, ay, az)")
            print(" 2 = Gyroscope     (gx, gy, gz)")
            print(" 3 = Magnetometer  (mx, my, mz)")
            print(" q = Quit")
            choice = input(">> ").strip().lower()

            if choice == "q":
                print("Exiting.")
                break
            if choice not in {"1", "2", "3"}:
                print("Invalid choice, please enter 1, 2, 3, or q.")
                continue

            sensor_code = int(choice)
            request_prediction(ser, sensor_code)

if __name__ == "__main__":
    main()