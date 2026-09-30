import numpy as np

def calculate_feature_attribution(model, window, feature_names):
    """
    Calculate the reconstruction error per feature for a single window.
    window: array of shape (1, window_size, num_features)
    Returns: list of (feature_name, error) sorted by error descending.
    """
    if len(window.shape) == 2:
        window = np.expand_dims(window, axis=0)
        
    reconstruction = model.predict(window, verbose=0)
    
    # Calculate MSE per feature over the window
    # window shape: (1, window_size, num_features)
    mse_per_feature = np.mean(np.power(window[0] - reconstruction[0], 2), axis=0)
    
    # Normalize for display purposes
    total_error = np.sum(mse_per_feature) + 1e-10
    normalized_error = mse_per_feature / total_error
    
    attributions = [(feature_names[i], float(normalized_error[i])) for i in range(len(feature_names))]
    attributions.sort(key=lambda x: x[1], reverse=True)
    
    return attributions
