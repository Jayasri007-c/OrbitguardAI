import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

def create_windows(data, window_size):
    """Create sliding windows for time-series data."""
    windows = []
    for i in range(len(data) - window_size + 1):
        windows.append(data[i:i + window_size])
    return np.array(windows)

def scale_data(train_df, val_df, test_df, feature_cols, scaler_path="models/scaler.joblib"):
    """
    Scale data fitting only on the training set to prevent data leakage.
    """
    scaler = StandardScaler()
    
    train_scaled = scaler.fit_transform(train_df[feature_cols])
    val_scaled = scaler.transform(val_df[feature_cols]) if not val_df.empty else np.array([])
    test_scaled = scaler.transform(test_df[feature_cols]) if not test_df.empty else np.array([])
    
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    return train_scaled, val_scaled, test_scaled, scaler

def load_scaler(scaler_path="models/scaler.joblib"):
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file missing at {scaler_path}. Please train a model first.")
    return joblib.load(scaler_path)
