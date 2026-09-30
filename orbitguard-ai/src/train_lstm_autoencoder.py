import os
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Input
from data_loader import load_config, load_data, get_train_val_test_splits
from preprocessing import scale_data, create_windows

def build_lstm_autoencoder(window_size, num_features):
    model = Sequential([
        Input(shape=(window_size, num_features)),
        LSTM(32, activation='relu', return_sequences=False),
        RepeatVector(window_size),
        LSTM(32, activation='relu', return_sequences=True),
        TimeDistributed(Dense(num_features))
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_lstm_autoencoder():
    config = load_config()
    print("Loading data for LSTM Autoencoder training...")
    try:
        df = load_data(config)
    except FileNotFoundError as e:
        print(e)
        return
        
    feature_cols = config.get("feature_columns", [])
    window_size = config.get("window_size", 20)
    
    train_df, val_df, test_df = get_train_val_test_splits(df, config)
    
    print("Preprocessing and scaling data...")
    train_scaled, val_scaled, _, _ = scale_data(train_df, val_df, test_df, feature_cols, scaler_path="models/lstm_scaler.joblib")
    
    train_windows = create_windows(train_scaled, window_size)
    val_windows = create_windows(val_scaled, window_size) if len(val_scaled) >= window_size else None
    
    print(f"Created {len(train_windows)} training windows of size {window_size}.")
    
    model = build_lstm_autoencoder(window_size, len(feature_cols))
    
    print("Training LSTM Autoencoder on real normal data...")
    # Add early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss' if val_windows is not None else 'loss', patience=3, restore_best_weights=True)
    
    model.fit(
        train_windows, train_windows,
        epochs=20,
        batch_size=32,
        validation_data=(val_windows, val_windows) if val_windows is not None else None,
        callbacks=[early_stopping],
        verbose=2
    )
    
    model_path = "models/lstm_autoencoder.keras"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"LSTM Autoencoder trained and saved to {model_path}")
    
    # Calculate and save max_mse for normalization
    print("Calculating max_train_mse for normalization...")
    reconstructions = model.predict(train_windows, verbose=0)
    mse = np.mean(np.power(train_windows - reconstructions, 2), axis=(1, 2))
    max_mse = float(np.max(mse))
    
    import joblib
    joblib.dump(max_mse, "models/lstm_max_mse.joblib")
    print(f"Max train MSE ({max_mse:.6f}) saved to models/lstm_max_mse.joblib")

if __name__ == "__main__":
    train_lstm_autoencoder()
