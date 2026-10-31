import streamlit as st
import pandas as pd
import numpy as np
import time

# --- UI Configuration ---
st.set_page_config(page_title="EV Battery BMS Dashboard", layout="wide")
st.title("🔋 EV Predictive Maintenance & Hazard Detection BMS")
st.markdown("Live Telemetry and Remaining Useful Life (RUL) Interface")

st.sidebar.header("Hardware Status")
st.sidebar.warning("⚠️ Hardware Disconnected")
st.sidebar.success("✅ Simulation Mode: Active (Extracting NASA Telemetry)")

col1, col2, col3 = st.columns(3)
metric_voltage = col1.empty()
metric_temp = col2.empty()
metric_soh = col3.empty()
warning_banner = st.empty()

st.subheader("Real-Time Capacity Degradation (SoH)")
chart_placeholder = st.empty()

# --- Continuous Hardware Simulation Loop ---
capacity = 2.0
cycles, capacities = [], []
cycle_count = 1

# Change to a while True loop to run indefinitely
while True:
    voltage = round(np.random.uniform(3.7, 4.2), 2)
    temp = round(np.random.uniform(35.0, 39.5), 1)
    
    capacity = capacity - np.random.uniform(0.002, 0.008)
    soh = (capacity / 2.0) * 100
    
    cycles.append(cycle_count)
    capacities.append(capacity)
    
    # Update UI Metrics
    metric_voltage.metric("Live Cell Voltage", f"{voltage} V")
    metric_temp.metric("Live Cell Temperature", f"{temp} °C")
    metric_soh.metric("State of Health (SoH)", f"{soh:.1f} %")
    
    if temp > 38.0:
        warning_banner.error(f"⚠️ THERMAL HAZARD DETECTED: {temp}°C! Initiating Cooling Protocol.")
    else:
        warning_banner.empty()
        
    # Only keep the last 50 data points on the chart so it doesn't get cluttered over time
    if len(cycles) > 50:
        cycles.pop(0)
        capacities.pop(0)

    df = pd.DataFrame({"Cycle": cycles, "Capacity (Ah)": capacities})
    chart_placeholder.line_chart(df.set_index("Cycle"))
    
    cycle_count += 1
    
    # Slow down the loop slightly so the graph animates smoothly
    time.sleep(1.0)
    # py -m streamlit run "c:\Users\Rishi Gupta\OneDrive\Documents\PBL\dashboard.py"