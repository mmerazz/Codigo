import threading
import serial
import time
import keyboard
import numpy as np
from pyOpenBCI import OpenBCICyton
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from tratamientoDatos import *

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

# -------- Marker Variable (Shared Between Threads) --------
markers_list = []
current_marker = 0  # Default: no marker
marker_lock = threading.Lock()

# -------- Function to Listen for Key Presses --------
def key_listener():
    global current_marker
    while True:
        try:
            if keyboard.is_pressed('1'):
                with marker_lock:
                    current_marker = 1
                print("🟡 Marker 1 Set")
                time.sleep(0.3)  # Debounce
            elif keyboard.is_pressed('2'):
                with marker_lock:
                    current_marker = 2
                print("🟢 Marker 2 Set")
                time.sleep(0.3)
            elif keyboard.is_pressed('3'):
                with marker_lock:
                    current_marker = 3
                print("🔵 Marker 3 Set")
                time.sleep(0.3)
            elif keyboard.is_pressed('0'):
                with marker_lock:
                    current_marker = 0
                print("⚪ Marker Cleared")
                time.sleep(0.3)
        except:
            break

# -------- OpenBCI Sample Callback --------
def print_raw(sample):
    global current_marker
    with marker_lock:
        marker = current_marker

    # You can now use 'marker' with your sample
    print(f"Data: {sample.channels_data}, Marker: {marker}")
    markers_list.append(marker)



def detect_paterns(sample):
    num_samples = 1000
    num_channels = 8
    num_trials = 500 
    fs = 256  # Sampling frequency
    labels = np.random.randint(0, 4, num_trials)  # Random labels for four classes (0, 1, 2, 3)

    # Execution of the program
    emg_data = sample.channels_data
    filtered_data = filter_bands(emg_data, fs)
    epoch_length = 1000
    epochs = []
    labels = []

    # Sliding window: extract segments where Marker is stable
    i = 0
    while i + epoch_length <= markers_list.shape[0]:
        segment_marker = markers_list[i:i+epoch_length]
        if np.all(segment_marker == segment_marker[0]) and segment_marker[0] in [0,1,2,3]:
            epoch = filtered_data[:, i:i+epoch_length]
            epochs.append(epoch)
            labels.append(int(segment_marker[0]))
            i += epoch_length  # Move to next segment
        else:
            i += epoch_length  # Skip noisy or mixed segments

    X = np.array(epochs)  # Shape: (n_trials, 8, epoch_length)
    y = np.array(labels)  # Shape: (n_trials,)

    print(f"Data shape: {X.shape}, Labels shape: {y.shape}")

    # ------------------ 4. Train/Test Split ------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # ------------------ 5. CSP + SVM Pipeline ------------------
    csp = CSP(n_components=4, reg=None, log=True, cov_est='epoch')
    svm = SVC(kernel='linear', probability=True)

    pipeline = Pipeline([
        ('csp', csp),
        ('scaler', StandardScaler()),
        ('svm', svm)
    ])

    # ------------------ 6. Train and Evaluate ------------------
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    return y_pred


# -------- Start OpenBCI and Key Listener --------
try:
    print("Starting Key Listener (press 1, 2, 3 to set markers, 0 to clear)...")
    key_thread = threading.Thread(target=key_listener, daemon=True)
    key_thread.start()

    print("Connecting to OpenBCI board...")
    board = OpenBCICyton(port='COM3')  # Change COM port if needed
    print("Streaming EMG data... Press Ctrl+C to stop.")
    board.start_stream(print_raw)

except KeyboardInterrupt:
    print("Stopping stream...")
    board.stop_stream()
    print("Stream stopped.")
