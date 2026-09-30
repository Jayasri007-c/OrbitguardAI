import os
import joblib
import pandas as pd
import numpy as np
from keras.models import load_model
from data_loader import load_config, load_data, get_train_val_test_splits
from preprocessing import scale_data, create_windows, load_scaler
from anomaly_detection import score_isolation_forest, score_lstm_autoencoder
from fusion import fuse_scores

def evaluate_models():
    config = load_config()
    print("Loading data for Evaluation...")
    try:
        df = load_data(config)
    except FileNotFoundError as e:
        print(e)
        return
        
    feature_cols = config.get("feature_columns", [])
    window_size = config.get("window_size", 20)
    
    _, _, test_df = get_train_val_test_splits(df, config)
    
    if test_df.empty:
        print("Test dataset is empty. Check split ratios.")
        return
        
    try:
        if_scaler = load_scaler("models/if_scaler.joblib")
        lstm_scaler = load_scaler("models/lstm_scaler.joblib")
        
        clf = joblib.load("models/isolation_forest.joblib")
        model = load_model("models/lstm_autoencoder.keras")
    except Exception as e:
        print(f"Error loading models or scalers: {e}")
        print("Please train the models first.")
        return

    print("Evaluating on unsupervised metrics (Test Data)...")
    
    test_scaled_if = if_scaler.transform(test_df[feature_cols])
    test_scaled_lstm = lstm_scaler.transform(test_df[feature_cols])
    
    # ISOLATION FOREST
    if_scores = score_isolation_forest(clf, test_scaled_if)
    
    # LSTM AUTOENCODER
    test_windows = create_windows(test_scaled_lstm, window_size)
    
    try:
        max_mse = joblib.load("models/lstm_max_mse.joblib")
    except FileNotFoundError:
        max_mse = None
        
    lstm_scores = score_lstm_autoencoder(model, test_windows, max_mse=max_mse)
    
    # Align scores (IF scores are per point, LSTM scores are per window)
    # We will evaluate on the windows to have comparable lengths
    aligned_if_scores = if_scores[window_size-1:]
    
    if len(aligned_if_scores) != len(lstm_scores):
        print("Warning: Score lengths do not match.")
        
    # FUSION
    fused_scores = fuse_scores(aligned_if_scores, lstm_scores, alpha=config.get("fusion_alpha", 0.5))
    
    # Output metrics
    results = pd.DataFrame({
        "IF_Score": aligned_if_scores,
        "LSTM_Score": lstm_scores,
        "Fused_Score": fused_scores
    })
    
    print("\n--- Unsupervised Evaluation Summary ---")
    print(results.describe())
    
    if "anomaly" in test_df.columns:
        print("\nGround truth 'anomaly' column found. Calculating supervised metrics...")
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
        y_true = test_df["anomaly"].values[window_size-1:]
        
        # Use a threshold to convert fused_scores (0-1) to binary predictions (e.g., > 0.5)
        threshold = 0.5
        y_pred = (fused_scores > threshold).astype(int)
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        
        try:
            auc = roc_auc_score(y_true, fused_scores)
        except ValueError:
            auc = np.nan
            
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"ROC AUC:   {auc:.4f}")
        
        results["Ground_Truth"] = y_true
        results["Prediction"] = y_pred
    else:
        print("\nNo 'anomaly' ground truth labels found.")
        print("Supervised metrics (Precision, Recall, F1) cannot be calculated.")
        
    os.makedirs("outputs", exist_ok=True)
    results.to_csv("outputs/metrics.csv", index=False)
    print("Evaluation results saved to outputs/metrics.csv")

if __name__ == "__main__":
    evaluate_models()
