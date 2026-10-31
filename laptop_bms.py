import streamlit as st
import pandas as pd
import time
import psutil
import re

st.set_page_config(page_title="BMS Dashboard", layout="wide")

# Hide Streamlit developer menus safely
hide_elements_style = """
    <style>
        [data-testid="stStatusWidget"] { display: none !important; }
        [data-testid="stAppDeployButton"] { display: none !important; }
        [data-testid="stToolbar"] { display: none !important; }
    </style>
"""
st.markdown(hide_elements_style, unsafe_allow_html=True)

st.title("💻 Battery Management System (BMS) Dashboard")
st.markdown("---")

@st.cache_data
def get_historical_data():
    try:
        with open("battery-report.html", "r", encoding="utf-8") as file:
            html = file.read()
            
        def extract_val(label):
            idx = html.find(label)
            if idx != -1:
                snippet = html[idx:idx+200]
                match = re.search(r'<td.*?>\s*(.*?)\s*</td>', snippet)
                if match:
                    return match.group(1).replace(',', '')
            return None

        c = extract_val("CYCLE COUNT")
        cycles = c if c and c != "-" else "N/A"
        
        d = extract_val("DESIGN CAPACITY")
        d_cap = int(d.split()[0]) if d else 1
        
        f = extract_val("FULL CHARGE CAPACITY")
        f_cap = int(f.split()[0]) if f else 1
        
        soh = (f_cap / d_cap) * 100 if d_cap > 1 else 0
        return cycles, d_cap, f_cap, soh
    except Exception:
        return "N/A", 0, 0, 0.0

cycles, d_cap, f_cap, soh = get_historical_data()

# --- Card Grid Layout ---
left_card, right_card = st.columns(2)

# Card 1: Easy-to-explain historical data
with left_card:
    with st.container(border=True):
        st.subheader("Historical Battery Health")
        st.markdown("<br>", unsafe_allow_html=True)
        
        hc1, hc2 = st.columns(2)
        hc1.metric("Total Cycles", cycles)
        hc2.metric("Original Capacity", f"{d_cap} mWh")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        hc3, hc4 = st.columns(2)
        hc3.metric("Current Capacity", f"{f_cap} mWh")
        hc4.metric("Battery Health (SoH)", f"{soh:.1f} %")
        st.markdown("<br>", unsafe_allow_html=True)

# Card 2: Easy-to-explain live data
with right_card:
    with st.container(border=True):
        st.subheader("Live Battery Status")
        st.markdown("<br>", unsafe_allow_html=True)
        
        lc1, lc2, lc3 = st.columns(3)
        charge_metric = lc1.empty()
        status_metric = lc2.empty()
        temp_metric = lc3.empty()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        health_metric = lc1.empty()
        st.markdown("**Visual Charge Level**")
        charge_bar = st.empty()

st.markdown("---")
st.subheader("Live Battery Drain Graph")
graph_placeholder = st.empty()

# --- The Hardware Loop ---
times = []
charges = []

while True:
    # --- Existing Battery Logic ---
    try:
        battery = psutil.sensors_battery()
        charge = battery.percent if battery else 0
        plugged = battery.power_plugged if battery else False
    except Exception:
        charge = 0
        plugged = False

    # --- NEW Temperature Logic ---
    try:
        # Attempt to fetch hardware temps (Often blocked by Windows)
        hw_temps = psutil.sensors_temperatures()
        
        if not hw_temps:
            temp_display = "OS Restricted"
        else:
            # If it actually finds thermal zones, grab the first available one
            first_zone = list(hw_temps.keys())[0]
            current_temp = hw_temps[first_zone][0].current
            temp_display = f"{current_temp} °C"
    except AttributeError:
        # Windows completely lacks the temperature attribute in psutil
        temp_display = "Not Supported"

    # --- Update the UI ---
    charge_metric.metric("Charge Percentage", f"{charge} %")
    status_metric.metric("Power Source", "Charging" if plugged else "Discharging")
    temp_metric.metric("Thermal Status", temp_display) # <--- Pushes data to screen
    
    health_metric.metric("System Status", "Healthy" if charge > 20 else "Warning: Connect Charger")
    charge_bar.progress(charge / 100.0)

    current_time = time.strftime("%H:%M:%S")
    times.append(current_time)
    charges.append(charge)

    if len(times) > 30:
        times.pop(0)
        charges.pop(0)

    df = pd.DataFrame({"Time": times, "Charge %": charges})
    graph_placeholder.area_chart(df.set_index("Time"))

    time.sleep(2)
    # streamlit run laptop_bms.py