import os
from sklearn.ensemble import IsolationForest
import joblib
from data_loader import load_config, load_data, get_train_val_test_splits
from preprocessing import scale_data

def train_isolation_forest():
    config = load_config()
    print("Loading data for Isolation Forest training...")
    try:
        df = load_data(config)
    except FileNotFoundError as e:
        print(e)
        return
        
    feature_cols = config.get("feature_columns", [])
    
    train_df, val_df, test_df = get_train_val_test_splits(df, config)
    
    print("Preprocessing and scaling data...")
    train_scaled, _, _, _ = scale_data(train_df, val_df, test_df, feature_cols, scaler_path="models/if_scaler.joblib")
    
    print("Training Isolation Forest on real normal data...")
    clf = IsolationForest(contamination=0.01, random_state=42)
    clf.fit(train_scaled)
    
    model_path = "models/isolation_forest.joblib"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)
    print(f"Isolation Forest trained and saved to {model_path}")

if __name__ == "__main__":
    train_isolation_forest()
