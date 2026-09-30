# ORBITGUARD AI

**CubeSat Telemetry Anomaly Detection and Early-Warning System**

OrbitGuard AI is an early-warning system designed to monitor CubeSat health by analyzing telemetry data. It employs a hybrid ML architecture to detect both point-in-time and temporal anomalies without relying on synthetic data.

**IMPORTANT: The repository contains the complete pipeline but does not contain models trained on simulated data. Users must train the models using their own real telemetry dataset.**

## Architecture

OrbitGuard AI implements two complementary anomaly detection models:
1. **Isolation Forest**: Detects point anomalies (individual telemetry snapshots that deviate from normal).
2. **LSTM Autoencoder**: Detects temporal anomalies (abnormal sequences of telemetry readings over a sliding window).

The system fuses the scores from these two models and calculates per-feature attribution to explain anomalies.

## Setup Instructions

### 1. Prerequisites
- Python 3.12 (Windows)

### 2. Installation
Extract the project and install the dependencies:
```bash
pip install -r requirements.txt
```

### 3. Dataset Configuration
1. Place your real CubeSat telemetry dataset in the `data/` folder (e.g., `data/telemetry.csv`).
2. Edit `config/config.yaml` to specify the exact column names for your timestamp and telemetry features.
3. Edit `config/subsystems.yaml` to map your specific telemetry features to their corresponding CubeSat subsystems.

### 4. Training
**OrbitGuard AI MUST be trained on your real, normal telemetry data.**
Do not use fake or simulated data.

Train the models explicitly:
```bash
# 1. Train the Isolation Forest
python src/train_isolation_forest.py

# 2. Train the LSTM Autoencoder
python src/train_lstm_autoencoder.py
```
This will fit the models and scalers, saving them to the `models/` directory.

### 5. Evaluation
To evaluate the models on your dataset's test split:
```bash
python src/evaluate.py
```
The evaluation results will be saved to `outputs/metrics.csv`.

### 6. Dashboard & Live Replay
After training, launch the interactive dashboard:
```bash
streamlit run app.py
```

The dashboard includes a "Live Simulation" feature. **This feature does NOT generate synthetic data.** It replays rows from your actual `data/telemetry.csv` file sequentially to simulate live data ingestion, applying the trained anomaly detection models in real-time.

## Limitations
- Performance depends heavily on the quality and volume of the normal training data provided.
- Ensure the training data does not contain undetected anomalies, as they will be learned as "normal" behavior.
