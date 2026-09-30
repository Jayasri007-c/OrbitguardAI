import os
import joblib
import pandas as pd
import numpy as np
from keras.models import load_model

from src.data_loader import load_config, load_data, get_train_val_test_splits
from src.preprocessing import create_windows, load_scaler
from src.anomaly_detection import score_isolation_forest, score_lstm_autoencoder
from src.fusion import fuse_scores
from src.severity import calculate_severity

print("Loading configs...")
config = load_config()
feature_cols = config.get("feature_columns", [])
window_size = config.get("window_size", 20)

print("Loading data...")
df = load_data(config)
train_df, val_df, test_df = get_train_val_test_splits(df, config)

print("Loading models...")
if_scaler = load_scaler("models/if_scaler.joblib")
lstm_scaler = load_scaler("models/lstm_scaler.joblib")
clf = joblib.load("models/isolation_forest.joblib")
model = load_model("models/lstm_autoencoder.keras")
max_mse = joblib.load("models/lstm_max_mse.joblib")

print("Simulating first 3 steps...")
for step in range(window_size, window_size + 3):
    current_window_df = test_df.iloc[step - window_size : step]
    current_row = test_df.iloc[step-1]
    
    current_channel = current_row.get("channel", "Unknown Channel")
    
    scaled_if_window = if_scaler.transform(current_window_df[feature_cols])
    scaled_lstm_window = lstm_scaler.transform(current_window_df[feature_cols])
    
    current_if_point = scaled_if_window[-1].reshape(1, -1)
    if_score = float(score_isolation_forest(clf, current_if_point)[0])
    
    lstm_window_reshaped = np.expand_dims(scaled_lstm_window, axis=0)
    lstm_score = float(score_lstm_autoencoder(model, lstm_window_reshaped, max_mse=max_mse)[0])
    
    fused_score = fuse_scores(if_score, lstm_score, alpha=config.get("fusion_alpha", 0.5))
    severity = calculate_severity(fused_score)
    
    print(f"Step {step}: Channel {current_channel}, IF: {if_score:.4f}, LSTM: {lstm_score:.4f}, Fused: {fused_score:.4f} -> {severity}")

print("Test complete. Models and logic run successfully!")
