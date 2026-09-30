import numpy as np

def detect_trend(values):
    """
    Detect the trend of a sequence of values using a simple linear fit.
    """
    if len(values) < 2:
        return "Stable"
    
    x = np.arange(len(values))
    y = np.array(values)
    
    # Simple linear regression slope
    slope, _ = np.polyfit(x, y, 1)
    
    # Define a threshold for stability (this might need tuning based on feature scale)
    threshold = 0.01 * (np.max(y) - np.min(y) + 1e-5)
    
    if slope > threshold:
        return "Increasing"
    elif slope < -threshold:
        return "Decreasing"
    else:
        return "Stable"
