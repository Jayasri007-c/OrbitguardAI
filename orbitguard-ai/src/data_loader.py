import pandas as pd
import yaml
import os

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_data(config):
    dataset_path = config.get("dataset_path", "data/telemetry.csv")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Telemetry dataset not found at {dataset_path}. Please place your real telemetry CSV in data/.")
    
    df = pd.read_csv(dataset_path)
    
    timestamp_col = config.get("timestamp_column", "timestamp")
    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found in the dataset.")
        
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(by=timestamp_col)
    
    # Remove duplicate timestamps
    df = df.drop_duplicates(subset=[timestamp_col])
    
    # Handle missing values (forward fill then backward fill)
    df = df.ffill().bfill()
    
    # Handle invalid numerical values by dropping or interpolating
    # Assuming numerical features for now
    feature_cols = config.get("feature_columns", [])
    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(f"Configured feature '{col}' not found in the dataset.")
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.interpolate(method='linear').ffill().bfill()
    
    return df

def get_train_val_test_splits(df, config):
    if 'train' in df.columns:
        train_df = df[df['train'] == 1].copy()
        test_df = df[df['train'] == 0].copy()
        
        # Split train_df further to get a validation set (last 15% of train)
        val_split = config.get("validation_split", 0.15)
        val_end = int(len(train_df) * (1 - val_split))
        val_df = train_df.iloc[val_end:].copy()
        train_df = train_df.iloc[:val_end].copy()
    else:
        train_split = config.get("train_split", 0.7)
        val_split = config.get("validation_split", 0.15)
        
        n = len(df)
        train_end = int(n * train_split)
        val_end = int(n * (train_split + val_split))
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
    return train_df, val_df, test_df
