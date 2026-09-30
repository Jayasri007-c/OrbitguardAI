def calculate_severity(anomaly_score):
    """
    Calculate severity based on anomaly score thresholds.
    These thresholds should ideally be derived from percentiles of normal validation data.
    """
    if anomaly_score < 0.5:
        return "Normal"
    elif anomaly_score < 0.7:
        return "Warning"
    elif anomaly_score < 0.85:
        return "High"
    else:
        return "Critical"
