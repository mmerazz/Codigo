#The serial library is used for serial communication in Python.
#It allows you to read from and write to serial ports
import serial
import time
import numpy as np
from pyOpenBCI import OpenBCICyton
# Configure OpenBCI and Arduino ports (Update for your system)
CYTON_PORT = "/dev/tty.usbserial-D200QBI7"  # Change based on your system
ARDUINO_PORT = "COM4"  # Change based on your Arduino port (Windows: "COMx", macOS/Linux: "/dev/ttyUSBx")
BAUD_RATE = 115200
ARDUINO_BAUD = 9600  # Arduino baud rate

# Initialize OpenBCI Cyton
board = OpenBCICyton(port=CYTON_PORT)

# Initialize Arduino serial connection
arduino = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=1)
time.sleep(2)  # Allow Arduino to initialize

# Parameters
BUFFER_SIZE = 125  # Samples per second (assuming 125Hz)
EXTRA_THRESHOLD = 100  # Buffer above average for clench detection
REFRACTORY_PERIOD = 1.5  # Time before allowing a new clench (in seconds)
CLENCH_COOLDOWN = 0.5  # Extra buffer to avoid duplicate clenches (seconds)
clench_detected = False
last_clench_time = 0  # Time of last detected clench
emg_window = []  # Rolling window for EMG data

def detect_clench(sample):
    """Adaptive threshold clench detection with cooldown buffer."""
    global clench_detected, last_clench_time

    emg_signal = sample.channels_data;  # Use all chanels

    # Maintain a rolling window of the last second of data
    emg_window.append(emg_signal)
    if len(emg_window) > BUFFER_SIZE:
        emg_window.pop(0)  # Keep window size fixed

    # Compute rolling average of EMG
    avg_signal = np.mean(emg_window) if emg_window else 0
    adaptive_threshold = avg_signal + EXTRA_THRESHOLD  # Dynamic threshold

    # Get current time
    current_time = time.time()

    # Clench detection with cooldown buffer
    if emg_signal > adaptive_threshold and not clench_detected:
        if current_time - last_clench_time > (REFRACTORY_PERIOD + CLENCH_COOLDOWN):
            print("💪 Clench Detected! Toggling LED.")
            arduino.write(b'T')  # Send toggle command to Arduino
            clench_detected = True
            last_clench_time = current_time  # Update last detection time

    # Reset detection only if EMG stays low for a while
    elif emg_signal < adaptive_threshold * 0.8 and current_time - last_clench_time > CLENCH_COOLDOWN:
        clench_detected = False

# Start OpenBCI stream
print("Starting OpenBCI Stream. Clench your fist to toggle LED.")
board.start_stream(detect_clench)