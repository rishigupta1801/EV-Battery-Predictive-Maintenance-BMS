import streamlit as st
import pandas as pd
import numpy as np
import serial
import time
import os
import scipy.io
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

SERIAL_PORT = "COM8" 
BAUD_RATE = 115200

st.set_page_config(page_title="Digital Twin BMS", layout="wide")
st.title("🧠 AI Digital Twin Battery Management System")
st.markdown("Live Thermal Telemetry + Mathematically Simulated Voltage Feed")

@st.cache_resource
def load_and_train_ai():
    base_dir = r"C:\Users\Rishi Gupta\OneDrive\Documents\PBL"
    mat_file_path = os.path.join(base_dir, "Battery_Data", "5. Battery Data Set", "1. BatteryAgingARC-FY08Q4", "B0005.mat")
    if not os.path.exists(mat_file_path): return None, None, None
    mat = scipy.io.loadmat(mat_file_path)
    data = mat['B0005'][0, 0]['cycle'][0]
    capacities = [row['data'][0][0]['Capacity'][0][0] for row in data if row['type'][0] == 'discharge']
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(np.array(capacities).reshape(-1, 1))
    X, y = [], []
    for i in range(len(scaled_data) - 5):
        X.append(scaled_data[i:(i + 5), 0])
        y.append(scaled_data[i + 5, 0])
        
    X = np.reshape(np.array(X), (len(X), 5, 1))
    tf.random.set_seed(42)
    model = Sequential([LSTM(50, return_sequences=True, input_shape=(5, 1)), Dropout(0.2), LSTM(50), Dropout(0.2), Dense(1)])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, np.array(y), epochs=20, batch_size=8, verbose=0) 
    return model, scaler, scaled_data[-5:].reshape(1, 5, 1)

with st.spinner('Initializing Digital Twin AI...'):
    lstm_model, capacity_scaler, live_sequence = load_and_train_ai()

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
except:
    st.error(f"Could not connect to {SERIAL_PORT}.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
metric_voltage = col1.empty()
metric_current = col2.empty()
metric_temp = col3.empty()
metric_rul = col4.empty()
chart_placeholder = st.empty()

history_capacity = []
current_capacity = 2.0  

while True:
    try:
        line = ser.readline().decode('utf-8').strip()
        if line and lstm_model:
            data = line.split(',')
            if len(data) == 3:
                # We completely ignore data[0] and data[1] (The broken INA219)
                temp = float(data[2])
                
                # --- THE DIGITAL TWIN MATH ---
                # As real temperature goes up, fake voltage drops and current spikes
                simulated_voltage = max(3.0, round(4.2 - ((temp - 25.0) * 0.03), 2))
                simulated_current = max(0, round(500.0 + ((temp - 25.0) * 45), 2))

                if temp > 32.0: current_capacity = max(1.0, current_capacity - 0.05)

                scaled_cap = capacity_scaler.transform(np.array([[current_capacity]]))
                live_sequence[0, :-1, 0] = live_sequence[0, 1:, 0] 
                live_sequence[0, -1, 0] = scaled_cap[0, 0]         
                
                next_scaled = lstm_model.predict(live_sequence, verbose=0)
                current_capacity = float(capacity_scaler.inverse_transform(next_scaled)[0][0])
                rul = max(0, int((current_capacity - 1.4) / 0.005))
                
                metric_voltage.metric("Simulated Bus Voltage", f"{simulated_voltage:.2f} V")
                metric_current.metric("Simulated Load", f"{simulated_current:.2f} mA")
                metric_temp.metric("Real Cell Temperature", f"{temp:.2f} °C")
                metric_rul.metric("LSTM Predicted RUL", f"{rul} Cycles")
                
                history_capacity.append(current_capacity)
                if len(history_capacity) > 50: history_capacity.pop(0)
                chart_placeholder.line_chart(pd.DataFrame({"Capacity (Ah)": history_capacity}))
                
    except Exception:
        pass
    time.sleep(0.5)
    # streamlit run digital_twin_bms.py