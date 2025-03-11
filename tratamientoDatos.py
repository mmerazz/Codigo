import numpy as np
from scipy.signal import butter, lfilter
from sklearn.feature_selection import mutual_info_classif
from mne.decoding import CSP
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Función para diseñar un filtro Butterworth
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

# Simulación de datos de EEG
def simulate_eeg_data(num_samples, num_channels, num_trials):
    return np.random.randn(num_trials, num_samples, num_channels)

# Filtrado de datos en bandas de frecuencia
def filter_bands(data, fs):
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 12),
        'beta': (12, 30),
        'gamma': (30, 45)
    }
    return {band: bandpass_filter(data, lowcut, highcut, fs) for band, (lowcut, highcut) in bands.items()}

# Aplicar CSP y calcular características
def apply_csp(filtered_data, labels):
    combined_data = np.concatenate(list(filtered_data.values()), axis=0)
    
    # Create combined_labels based on the number of trials in each frequency band
    combined_labels = np.concatenate([np.full(filtered_data[key].shape[0], label) for key, label in zip(filtered_data.keys(), labels)])

    # Verify the shape of combined_data and combined_labels
    print("Forma de combined_data:", combined_data.shape)  # Should be (n_trials, n_channels, n_samples)
    print("Número de etiquetas:", len(combined_labels))

    # Check that the number of trials matches
    if combined_data.shape[0] != len(combined_labels):
        raise ValueError("El número de ensayos en combined_data debe coincidir con el número de etiquetas.")

    csp = CSP(n_components=2)
    csp.fit(combined_data, combined_labels)
    return csp.transform(combined_data)

def calculate_mutual_information(csp_features, labels):
    # Ensure labels are in the correct shape
    combined_labels = np.concatenate([np.full(filtered_data[key].shape[0], label) for key, label in zip(filtered_data.keys(), labels)])


    # Calculate mutual information
    mi_score = mutual_info_classif(csp_features, combined_labels, discrete_features=True)
    
    return mi_score

# Clasificación
def classify_features(csp_features, labels):
    combined_labels = np.concatenate([np.full(filtered_data[key].shape[0], label) for key, label in zip(filtered_data.keys(), labels)])

    X_train, X_test, y_train, y_test = train_test_split(csp_features, combined_labels, test_size=0.2, random_state=42)
    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)

# Parámetros
num_samples = 1000
num_channels = 8
num_trials = 500 
fs = 256  # Frecuencia de muestreo
labels = np.random.randint(0, 2, num_trials)  # Etiquetas aleatorias para dos clases

# Ejecución del programa
eeg_data = simulate_eeg_data(num_samples, num_channels, num_trials)
filtered_data = filter_bands(eeg_data, fs)
csp_features = apply_csp(filtered_data, labels)

# Verifica el número de ensayos en csp_features
print("Número de ensayos en csp_features:", csp_features.shape[0])

mi_scores = calculate_mutual_information(csp_features, labels)
accuracy = classify_features(csp_features, labels)

# Resultados
print("Scores de Información Mutua:", mi_scores)
print("Precisión de la Clasificación:", accuracy)