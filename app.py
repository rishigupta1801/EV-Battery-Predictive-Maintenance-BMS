import streamlit as st
import pandas as pd
import numpy as np
import serial
import time
import os
import scipy.io
import csv
from datetime import datetime

# ML Imports from your final_bms_lstm.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# --- SETUP VARIABLES ---
SERIAL_PORT = "COM8" 
BAUD_RATE = 115200
CSV_FILE = "bms_log.csv"

st.set_page_config(page_title="Live AI BMS", layout="wide")
st.title("🧠 AI-Powered BMS + NASA Digital Twin")
st.markdown("Live Hardware Feed seamlessly fused with LSTM Deep Learning")

# --- STEP 1: TRAIN THE LSTM ON BOOT (CACHED) ---
@st.cache_resource
def load_and_train_ai():
    # We cache this so the neural network only trains ONCE when you open the dashboard!
    base_dir = r"C:\Users\Rishi Gupta\OneDrive\Documents\PBL"
    
    # Exact path based on your screenshots
    mat_file_path = os.path.join(base_dir, "Battery_Data", "5. Battery Data Set", "1. BatteryAgingARC-FY08Q4", "B0005.mat")
    
    if not os.path.exists(mat_file_path):
        st.error(f"Cannot find NASA dataset exactly at {mat_file_path}")
        return None, None, None
        
    mat = scipy.io.loadmat(mat_file_path)
    data = mat['B0005'][0, 0]['cycle'][0]
    
    capacities = []
    for i in range(data.size):
        row = data[i]
        if row['type'][0] == 'discharge':
            capacities.append(row['data'][0][0]['Capacity'][0][0])
            
    # Prepare data for LSTM
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(np.array(capacities).reshape(-1, 1))
    
    X, y = [], []
    lookback = 5
    for i in range(len(scaled_data) - lookback):
        X.append(scaled_data[i:(i + lookback), 0])
        y.append(scaled_data[i + lookback, 0])
        
    X = np.array(X)
    y = np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Build your exact Keras model
    tf.random.set_seed(42)
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=20, batch_size=8, verbose=0) 
    
    # Grab the last 5 cycles to use as our "starting point" for live prediction
    recent_sequence = scaled_data[-lookback:].reshape(1, lookback, 1)
    
    return model, scaler, recent_sequence

# Show a loading spinner while the Neural Network trains
with st.spinner('Initializing NASA Digital Twin & Training LSTM Neural Network (This takes a few seconds)...'):
    lstm_model, capacity_scaler, live_sequence = load_and_train_ai()

# --- STEP 2: CONNECT TO HARDWARE ---
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    st.sidebar.success(f"✅ Hardware Connected on {SERIAL_PORT}")
except:
    st.sidebar.error(f"⚠️ Could not connect to {SERIAL_PORT}. Ensure it is plugged in and the Arduino Serial Monitor is closed.")
    st.stop()

# --- UI LAYOUT ---
col1, col2, col3, col4 = st.columns(4)
metric_voltage = col1.empty()
metric_temp = col2.empty()
metric_health = col3.empty()
metric_rul = col4.empty()

warning_banner = st.empty()
st.subheader("Live Deep Learning Capacity Forecast")
chart_placeholder = st.empty()

# --- STATE VARIABLES ---
history_temp = []
history_capacity = []
current_capacity = 2.0  # Starting Ah
cycle_count = 0

# --- THE LIVE HARDWARE + INFERENCE LOOP ---
while True:
    try:
        line = ser.readline().decode('utf-8').strip()
        
        if line and lstm_model:
            data = line.split(',')
            
            if len(data) == 3:
                voltage = float(data[0])
                current = float(data[1])
                temp = float(data[2])
                
                # 1. READ PHYSICAL SENSOR TO INFLUENCE AI
                # If physical temperature spikes, penalize the LSTM's memory buffer!
                if temp > 32.0:
                    warning_banner.error(f"⚠️ PHYSICAL THERMAL HAZARD ({temp:.2f}°C): Accelerating LSTM Degradation!")
                    penalty = 0.05  # Artificial capacity drop from heat damage
                    current_capacity = max(1.0, current_capacity - penalty)
                else:
                    warning_banner.success("✅ Battery in safe operating temperature range.")

                # 2. PERFORM LIVE AI INFERENCE
                # Create a fake scaled sequence based on our penalized capacity
                scaled_cap = capacity_scaler.transform(np.array([[current_capacity]]))
                live_sequence[0, :-1, 0] = live_sequence[0, 1:, 0] # Shift window
                live_sequence[0, -1, 0] = scaled_cap[0, 0]         # Append new data
                
                # Ask the LSTM what the NEXT capacity will be
                next_scaled = lstm_model.predict(live_sequence, verbose=0)
                next_capacity = capacity_scaler.inverse_transform(next_scaled)[0][0]
                
                current_capacity = float(next_capacity)
                soh = (current_capacity / 2.0) * 100
                
                # Estimate remaining cycles until failure (1.4 Ah threshold based on your code)
                rul = max(0, int((current_capacity - 1.4) / 0.005))
                
                # 3. DATA LOGGING
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(CSV_FILE, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    # Create headers if file is empty
                    if os.stat(CSV_FILE).st_size == 0:
                        writer.writerow(["Timestamp", "Voltage", "Current", "Temp", "LSTM_Capacity", "SOH", "RUL"])
                    writer.writerow([timestamp, voltage, current, temp, current_capacity, soh, rul])
                
                # 4. UPDATE METRICS
                metric_voltage.metric("Bus Voltage", f"{voltage:.2f} V")
                metric_temp.metric("Cell Temperature", f"{temp:.2f} °C")
                metric_health.metric("LSTM State of Health", f"{soh:.1f} %")
                metric_rul.metric("Predicted RUL", f"{rul} Cycles", delta=f"{temp:.1f}°C Load")
                
                # 5. UPDATE CHART
                history_capacity.append(current_capacity)
                if len(history_capacity) > 50:
                    history_capacity.pop(0)
                    
                df_chart = pd.DataFrame({"LSTM Predicted Capacity (Ah)": history_capacity})
                chart_placeholder.line_chart(df_chart)
                
    except Exception as e:
        pass
        
    time.sleep(0.5)