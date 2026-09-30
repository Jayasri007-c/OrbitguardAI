# OrbitGuard AI Architecture

OrbitGuard AI uses a hybrid anomaly detection pipeline consisting of an Isolation Forest for point-anomaly detection and an LSTM Autoencoder for temporal-anomaly detection.

## Pipeline
1. **Data Ingestion**: Raw telemetry loaded from CSV, timestamps aligned, missing values interpolated.
2. **Preprocessing**: Features scaled using a standard scaler fitted only on the training split to avoid data leakage. Windows created for the LSTM model.
3. **Model 1 (Isolation Forest)**: Learns point-in-time correlations between telemetry parameters.
4. **Model 2 (LSTM Autoencoder)**: Reconstructs time-series windows, flagging high reconstruction error as anomalous.
5. **Fusion**: Linearly interpolates the scores.
6. **Dashboard**: Presents a live, replayed view of the evaluation dataset.
