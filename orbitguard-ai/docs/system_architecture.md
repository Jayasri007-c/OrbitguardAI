# OrbitGuard AI: System Architecture & Flow

OrbitGuard AI is a dual-model, hybrid anomaly detection and early-warning system built specifically for CubeSat telemetry. It leverages unsupervised machine learning to detect both immediate point-in-time anomalies and progressive, temporal degradations in spacecraft health without relying on pre-labeled or synthetic anomaly data.

## 1. System Architecture

The architecture consists of a 5-stage pipeline:

### Stage 1: Data Ingestion & Preprocessing
* **Technology**: `pandas`, `scikit-learn`
* **Flow**: Telemetry data is ingested as a time-series stream. Missing values are interpolated, and the data is scaled using `StandardScaler` (fitted purely on normal training data to prevent data leakage). For temporal analysis, the continuous stream is segmented into overlapping sliding windows (e.g., length=20).

### Stage 2: Point Anomaly Detection (Isolation Forest)
* **Technology**: `scikit-learn` (IsolationForest)
* **Flow**: The Isolation Forest analyzes individual, instantaneous telemetry snapshots. It isolates observations by randomly selecting a feature and randomly selecting a split value. If an observation requires very few splits to be isolated, it is flagged as highly anomalous.
* **Feature**: Ideal for detecting sudden spikes, sensor glitches, or catastrophic immediate failures.

### Stage 3: Temporal Anomaly Detection (LSTM Autoencoder)
* **Technology**: `TensorFlow`, `Keras`
* **Flow**: The Long Short-Term Memory (LSTM) Autoencoder processes the sliding windows. The encoder compresses the sequence into a lower-dimensional latent space, and the decoder attempts to reconstruct the original sequence. The model is trained purely on normal data; therefore, when it encounters abnormal sequences (e.g., a slow thermal leak), it fails to reconstruct them accurately.
* **Feature**: The Mean Squared Error (MSE) between the input and the reconstruction serves as the anomaly score.

### Stage 4: Fusion & Attribution Layer
* **Technology**: `numpy`
* **Flow**: The normalized scores from the Isolation Forest and the LSTM Autoencoder are combined using a weighted fusion algorithm (e.g., `alpha * IF_score + (1-alpha) * LSTM_score`). 
* **Feature**: 
  * **Severity**: The fused score maps to dynamic severity levels (Warning, High, Critical).
  * **Attribution**: By calculating the per-feature reconstruction error from the LSTM, the system mathematically identifies *which* exact sensor (e.g., Battery Voltage, Gyro X) contributed the most to the anomaly.

### Stage 5: Real-Time Dashboard & Alerting
* **Technology**: `Streamlit`, `Plotly`
* **Flow**: A live interactive dashboard replays telemetry data, passing it through the trained pipeline. It maps anomalous features to physical spacecraft subsystems (e.g., Thermal, Power, Attitude) using a configurable YAML file, and visually alerts operators while plotting trends.

## 2. Core Technologies
* **Python 3.11/3.12**: Core runtime.
* **TensorFlow 2.17 & Keras 3**: Deep learning framework for temporal sequence modeling.
* **Scikit-Learn**: Machine learning utilities and tree-based models.
* **Streamlit**: Web-based frontend framework for the operator dashboard.
* **Plotly**: Interactive, hardware-accelerated time-series visualization.
* **YAML**: Configuration management for subsystems and hyperparameters.

## 3. Evaluation & Optimization Strategy
* **Metrics**: Evaluated using standard Unsupervised statistical thresholds and Supervised metrics (Accuracy, Precision, Recall, F1, ROC-AUC) when ground-truth labels are available in the test set.
* **Optimization**: The `alpha` fusion parameter and the LSTM `window_size` are the primary tunable hyperparameters for optimizing the Recall vs. False Positive Rate trade-off.
