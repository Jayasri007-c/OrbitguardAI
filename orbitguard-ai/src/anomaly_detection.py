import numpy as np

def score_isolation_forest(clf, data_scaled):
    # scikit-learn IsolationForest returns -1 for anomaly, 1 for normal
    # decision_function returns negative for anomaly, positive for normal
    # We invert it so higher score = more anomalous
    scores = -clf.decision_function(data_scaled)
    # Normalize to 0-1 range roughly based on typical IF output
    # Note: These min/max values should ideally be calibrated from training data
    min_score, max_score = -0.5, 0.5 
    normalized_scores = (scores - min_score) / (max_score - min_score)
    return np.clip(normalized_scores, 0, 1)

def score_lstm_autoencoder(model, windows, max_mse=None):
    if len(windows) == 0:
        return np.array([])
    reconstructions = model.predict(windows, verbose=0)
    # Calculate MSE per window
    mse = np.mean(np.power(windows - reconstructions, 2), axis=(1, 2))
    
    if max_mse is not None and max_mse > 0:
        # Scale by the training maximum to prevent false warnings on small batches
        normalized_scores = np.clip(mse / max_mse, 0, 1)
    else:
        # Fallback if max_mse is not provided (legacy behavior)
        normalized_scores = np.clip(mse / (np.max(mse) + 1e-10), 0, 1) if len(mse) > 0 else np.array([])
        
    return normalized_scores
