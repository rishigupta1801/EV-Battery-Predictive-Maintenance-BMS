# EV-Battery-Predictive-Maintenance-BMS
# Predictive Maintenance and Hazard Detection in EV Batteries (AI Digital Twin & BMS)
**Second Year B.Tech (AI & DS) PBL Project — PVG's COETM, Pune (AY 2026-27)**
* **Student:** Rishi Jitendra Gupta 
* **Guide / Faculty:** Prof. Minakshi Atre

## Project Overview
This project implements a Hardware-in-the-Loop (HITL) Digital Twin and Deep Learning Battery Management System (BMS) for Lithium-Ion EV cells (18650). It combines real-time temperature telemetry from an **ESP32 + DS18B20** sensor node (`COM8`) with a **2-Layer Keras LSTM Neural Network** trained on the **NASA Ames PCoE Battery Aging Dataset (`B0005.mat`)** to forecast State of Health (SoH %) and Remaining Useful Life (RUL) cycles.

## Repository File Structure
* `baseline_model.py` — Extracts NASA `B0005.mat` (168 discharge cycles) and benchmarks Linear Regression vs. XGBoost Regressor.
* `final_bms_lstm.py` — Trains the 2-layer Keras LSTM network (`lookback=5`, `Dropout=0.2`, `1.40 Ah EOL`) and runs 38.0 °C thermal diagnostics.
* `app.py` — **Live Hardware + AI Baseline Dashboard:** Streams live ESP32 serial telemetry (`COM8`), runs real-time LSTM inference, and logs data to `bms_log.csv`.
* `digital_twin_bms.py` — **Electro-Thermal Digital Twin Dashboard:** Mathematically simulates bus voltage sag and load current spikes from live DS18B20 temperature readings.
* `thermal_bms.py` — **Thermal Isolation Dashboard:** 3-column BMS interface isolating physical thermal runaway stress (`>32.0 °C`) against LSTM capacity degradation.
* `laptop_bms.py` & `battery-report.html` — Scrapes local Windows host battery health metrics and demonstrates OS-level thermal sensor locks (`psutil`).
* `bms_log.csv` — Timestamped black-box telemetry log generated during live hardware testing.
