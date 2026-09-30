import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
import joblib
import os
from keras.models import load_model

from src.data_loader import load_config, load_data, get_train_val_test_splits
from src.preprocessing import create_windows, load_scaler
from src.anomaly_detection import score_isolation_forest, score_lstm_autoencoder
from src.fusion import fuse_scores
from src.attribution import calculate_feature_attribution
from src.severity import calculate_severity
from src.subsystem_mapping import load_subsystems, get_subsystem

st.set_page_config(page_title="OrbitGuard AI", layout="wide", initial_sidebar_state="expanded")

# --- App State ---
if 'simulation_started' not in st.session_state:
    st.session_state.simulation_started = False
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'active_alert' not in st.session_state:
    st.session_state.active_alert = None
if 'alert_counter' not in st.session_state:
    st.session_state.alert_counter = 1
if 'channel_status' not in st.session_state:
    st.session_state.channel_status = {}
if 'recent_fused_scores' not in st.session_state:
    st.session_state.recent_fused_scores = []

# --- Load Configuration ---
@st.cache_resource
def load_all_configs():
    return load_config(), load_subsystems()

config, subsystems_config = load_all_configs()

# --- Title ---
st.title("OrbitGuard AI — CubeSat Early-Warning System")

# --- Load Models & Data ---
@st.cache_resource
def load_models_and_data():
    try:
        df = load_data(config)
    except FileNotFoundError:
        return None, "Telemetry dataset not found. Please place your real telemetry CSV in data/."
    
    _, _, test_df = get_train_val_test_splits(df, config)
    if test_df.empty:
        return None, "Test dataset is empty."

    try:
        if_scaler = load_scaler("models/if_scaler.joblib")
        lstm_scaler = load_scaler("models/lstm_scaler.joblib")
        clf = joblib.load("models/isolation_forest.joblib")
        model = load_model("models/lstm_autoencoder.keras")
    except Exception as e:
        return None, "Models not trained yet or scalers missing. Please run training scripts first."
        
    try:
        max_mse = joblib.load("models/lstm_max_mse.joblib")
    except FileNotFoundError:
        max_mse = None

    return (test_df, if_scaler, lstm_scaler, clf, model, max_mse), "Success"

loaded_assets, status_msg = load_models_and_data()
if loaded_assets is None:
    st.error(status_msg)
    st.stop()
test_df, if_scaler, lstm_scaler, clf, model, max_mse = loaded_assets
feature_cols = config.get("feature_columns", [])
window_size = config.get("window_size", 20)

st.sidebar.header("Controls")
if st.sidebar.button("Run Live Simulation"):
    st.session_state.simulation_started = True
    if st.session_state.current_step < window_size:
        st.session_state.current_step = window_size

if st.sidebar.button("Stop Simulation"):
    st.session_state.simulation_started = False

def calculate_trend_from_scores(scores):
    if len(scores) < 2:
        return "Stable"
    slope = np.polyfit(range(len(scores)), scores, 1)[0]
    if slope > 0.05:
        return "Increasing"
    elif slope < -0.05:
        return "Decreasing"
    return "Stable"

tab1, tab2 = st.tabs(["Live Simulation", "Model Evaluation"])

with tab1:
    if not st.session_state.simulation_started and st.session_state.current_step == 0:
        st.info("Click 'Run Live Simulation' in the sidebar to begin playing back the telemetry.")
    
    placeholder = st.empty()
    
    while st.session_state.simulation_started and st.session_state.current_step < len(test_df):
        step = st.session_state.current_step
        
        current_window_df = test_df.iloc[step - window_size : step]
        current_row = test_df.iloc[step-1]
        
        current_channel = current_row.get("channel", "Unknown Channel")
        
        scaled_if_window = if_scaler.transform(current_window_df[feature_cols])
        scaled_lstm_window = lstm_scaler.transform(current_window_df[feature_cols])
        
        current_if_point = scaled_if_window[-1].reshape(1, -1)
        if_score = float(score_isolation_forest(clf, current_if_point)[0])
        
        lstm_window_reshaped = np.expand_dims(scaled_lstm_window, axis=0)
        lstm_score = float(score_lstm_autoencoder(model, lstm_window_reshaped, max_mse=max_mse)[0])
        
        fused_score = fuse_scores(if_score, lstm_score, alpha=config.get("fusion_alpha", 0.5))
        severity = calculate_severity(fused_score)
        
        # Trend
        st.session_state.recent_fused_scores.append(fused_score)
        if len(st.session_state.recent_fused_scores) > 5:
            st.session_state.recent_fused_scores.pop(0)
        current_trend = calculate_trend_from_scores(st.session_state.recent_fused_scores)
        
        # Channel Status Monitor
        st.session_state.channel_status[current_channel] = {
            "Status": "🔴 Anomaly" if severity in ["Warning", "High", "Critical"] else "🟢 Normal",
            "Score": fused_score,
            "Trend": current_trend
        }
        
        # State Machine for Alerts
        if severity in ["Warning", "High", "Critical"]:
            if st.session_state.active_alert is None:
                new_alert = {
                    "ID": f"#{st.session_state.alert_counter:03d}",
                    "Time": pd.Timestamp.now().strftime("%H:%M:%S"),
                    "Channel": current_channel,
                    "Score": fused_score,
                    "Severity": severity,
                    "Trend": current_trend,
                    "Subsystem": "Not available in dataset",
                    "Status": "ACTIVE"
                }
                st.session_state.active_alert = new_alert
                st.session_state.alerts.insert(0, new_alert)
                st.session_state.alert_counter += 1
            else:
                st.session_state.active_alert["Score"] = fused_score
                st.session_state.active_alert["Severity"] = severity
                st.session_state.active_alert["Trend"] = current_trend
                for a in st.session_state.alerts:
                    if a["ID"] == st.session_state.active_alert["ID"]:
                        a.update(st.session_state.active_alert)
                        break
        else:
            if st.session_state.active_alert is not None:
                st.session_state.active_alert["Status"] = "RESOLVED"
                for a in st.session_state.alerts:
                    if a["ID"] == st.session_state.active_alert["ID"]:
                        a["Status"] = "RESOLVED"
                        break
                st.session_state.active_alert = None

        with placeholder.container():
            st.subheader("Satellite Health")
            health_color = "🟢 NORMAL"
            if severity == "Warning": health_color = "⚠️ WARNING"
            elif severity in ["High", "Critical"]: health_color = "🚨 CRITICAL"
            
            cols = st.columns(3)
            cols[0].metric("Overall Health", health_color)
            cols[1].metric("Fused Score", f"{fused_score:.4f}")
            cols[2].metric("IF Score / LSTM Score", f"{if_score:.4f} / {lstm_score:.4f}")
            
            st.markdown("---")
            
            if st.session_state.active_alert:
                alert = st.session_state.active_alert
                st.error(f"""
                🚨 **ACTIVE ALERT {alert['ID']}**
                
                **Channel**: {alert['Channel']}  
                **Subsystem**: {alert['Subsystem']}  
                **Anomaly Score**: {alert['Score']:.4f}  
                **Severity**: {alert['Severity']}  
                **Trend**: {alert['Trend']}  
                **Status**: {alert['Status']}
                """)
                st.markdown("---")
            
            st.subheader("📡 CHANNEL MONITORING")
            if len(st.session_state.channel_status) > 0:
                channel_df = pd.DataFrame.from_dict(st.session_state.channel_status, orient='index')
                channel_df.index.name = 'Channel'
                channel_df.reset_index(inplace=True)
                st.dataframe(channel_df, hide_index=True)
            
            st.markdown("---")
            st.subheader("📈 LIVE TELEMETRY")
            fig = go.Figure()
            for feature in feature_cols[:3]: 
                fig.add_trace(go.Scatter(
                    x=current_window_df[config.get("timestamp_column", "segment")],
                    y=current_window_df[feature],
                    mode='lines',
                    name=feature
                ))
            fig.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.subheader("🚨 ALERT HISTORY")
            if len(st.session_state.alerts) > 0:
                alerts_df = pd.DataFrame(st.session_state.alerts)
                st.dataframe(alerts_df, hide_index=True)
            else:
                st.info("No alerts generated yet.")
                
            with st.expander("Technical Diagnostics"):
                attributions = calculate_feature_attribution(model, lstm_window_reshaped, feature_cols)
                fig2 = go.Figure(go.Bar(
                    x=[f[1] for f in attributions],
                    y=[f[0] for f in attributions],
                    orientation='h'
                ))
                fig2.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0), yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig2, use_container_width=True)
                
        time.sleep(config.get("playback_speed", 1.0))
        st.session_state.current_step += 1

    if st.session_state.current_step >= len(test_df) and len(test_df) > 0:
        st.info("Simulation complete. Reached end of dataset.")
        st.session_state.simulation_started = False

with tab2:
    st.header("📊 Model Evaluation Results")
    st.write("Evaluating the models on the test dataset using the ground-truth `anomaly` column.")
    
    if st.button("Run Evaluation Now"):
        with st.spinner("Calculating metrics..."):
            test_scaled_if = if_scaler.transform(test_df[feature_cols])
            test_scaled_lstm = lstm_scaler.transform(test_df[feature_cols])
            
            if_scores = score_isolation_forest(clf, test_scaled_if)
            test_windows = create_windows(test_scaled_lstm, window_size)
            lstm_scores = score_lstm_autoencoder(model, test_windows, max_mse=max_mse)
            
            aligned_if_scores = if_scores[window_size-1:]
            fused_scores = fuse_scores(aligned_if_scores, lstm_scores, alpha=config.get("fusion_alpha", 0.5))
            
            if "anomaly" in test_df.columns:
                from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
                y_true = test_df["anomaly"].values[window_size-1:]
                
                def calc_metrics(scores, threshold=0.5):
                    y_pred = (scores > threshold).astype(int)
                    p = precision_score(y_true, y_pred, zero_division=0)
                    r = recall_score(y_true, y_pred, zero_division=0)
                    f1 = f1_score(y_true, y_pred, zero_division=0)
                    try:
                        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
                        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                    except Exception:
                        tn, fp, fn, tp, fpr = 0, 0, 0, 0, 0
                    return p, r, f1, fpr, tn, fp, fn, tp
                    
                metrics = {}
                metrics["Isolation Forest"] = calc_metrics(aligned_if_scores)
                metrics["LSTM Autoencoder"] = calc_metrics(lstm_scores)
                metrics["Fused Model"] = calc_metrics(fused_scores)
                
                for m_name, (p, r, f1, fpr, tn, fp, fn, tp) in metrics.items():
                    st.subheader(m_name)
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Precision", f"{p:.4f}")
                    c2.metric("Recall", f"{r:.4f}")
                    c3.metric("F1 Score", f"{f1:.4f}")
                    c4.metric("False Positive Rate", f"{fpr:.4f}")
                    st.text(f"Confusion Matrix:\nTN: {tn} | FP: {fp}\nFN: {fn} | TP: {tp}")
                    st.markdown("---")
            else:
                st.warning("No 'anomaly' column found in dataset. Cannot compute supervised metrics.")
