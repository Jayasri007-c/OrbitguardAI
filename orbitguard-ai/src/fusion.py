def fuse_scores(if_score, lstm_score, alpha=0.5):
    """
    Combines the Isolation Forest score and LSTM Autoencoder score.
    Alpha determines the weight of the Isolation Forest score.
    """
    return (alpha * if_score) + ((1 - alpha) * lstm_score)
