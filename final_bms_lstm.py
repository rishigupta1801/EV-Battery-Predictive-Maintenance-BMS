import os
import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# --- Step 1: File Paths & Parse Data ---
base_dir = r"C:\Users\Rishi Gupta\OneDrive\Documents\PBL"
extract_path = os.path.join(base_dir, "Battery_Data")

mat_file_path = None
for root, dirs, files in os.walk(extract_path):
    if 'B0005.mat' in files:
        mat_file_path = os.path.join(root, 'B0005.mat')
        break

print(f"Loading data from: {mat_file_path}")
mat = scipy.io.loadmat(mat_file_path)

data = mat['B0005'][0, 0]['cycle'][0]
cycles, capacities, temperatures = [], [], []
cycle_count = 1

for i in range(data.size):
    row = data[i]
    if row['type'][0] == 'discharge':
        cap = row['data'][0][0]['Capacity'][0][0]
        temp = np.max(row['data'][0][0]['Temperature_measured'][0]) # Simulating thermal sensor
        
        cycles.append(cycle_count)
        capacities.append(cap)
        temperatures.append(temp)
        cycle_count += 1

df = pd.DataFrame({'Cycle': cycles, 'Capacity': capacities, 'Temperature': temperatures})

# --- Step 2: Thermal Hazard Check (Software Simulation) ---
print("\n--- Running Thermal Diagnostics ---")
HAZARD_THRESHOLD = 38.0
for index, row in df.iterrows():
    if row['Temperature'] > HAZARD_THRESHOLD:
        print(f"⚠️ HAZARD: Cycle {row['Cycle']} reached {row['Temperature']:.2f}°C")

# --- Step 3: Chronological Splitting for Deep Learning ---
print("\n--- Training Deep Learning LSTM Model ---")
# Train on the first 70% of the battery's life, test on the last 30%
train_size = int(len(df) * 0.7)
train_data = df['Capacity'].values[:train_size].reshape(-1, 1)
test_data = df['Capacity'].values[train_size:].reshape(-1, 1)

# Neural networks require data to be scaled between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train = scaler.fit_transform(train_data)
scaled_test = scaler.transform(test_data)

# Create sequences (Look at the past 5 cycles to predict the 6th)
def create_sequences(data, lookback=5):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:(i + lookback), 0])
        y.append(data[i + lookback, 0])
    return np.array(X), np.array(y)

lookback = 5
X_train, y_train = create_sequences(scaled_train, lookback)
X_test, y_test = create_sequences(scaled_test, lookback)

# Reshape for LSTM [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# --- Step 4: Build the Keras LSTM Network ---
tf.random.set_seed(42)
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
# Train the model silently (verbose=0) so it doesn't flood your terminal
model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0)

# --- Step 5: Predict and Evaluate ---
predictions_scaled = model.predict(X_test)
predictions = scaler.inverse_transform(predictions_scaled)
actual = scaler.inverse_transform(y_test.reshape(-1, 1))

lstm_rmse = mean_squared_error(actual, predictions) ** 0.5
lstm_r2 = r2_score(actual, predictions)

print(f"\nFinal LSTM RMSE: {lstm_rmse:.4f}")
print(f"Final LSTM R2 Score: {lstm_r2:.4f}")

# --- Step 6: Plot the Future Prediction ---
plt.figure(figsize=(10, 6))
plt.plot(df['Cycle'], df['Capacity'], label='Actual Capacity Fade', color='blue', alpha=0.5)

# Plot the AI predicting the final 30% of the battery's life
test_cycles = df['Cycle'].values[train_size + lookback:]
plt.plot(test_cycles, predictions, label='LSTM RUL Forecast', color='purple', linewidth=2)

plt.axhline(y=1.4, color='black', linestyle=':', label='End of Life (1.4 Ah)')
plt.axvline(x=train_size, color='gray', linestyle='--', label='<- Training Phase | Forecasting Phase ->')
plt.xlabel('Discharge Cycles')
plt.ylabel('Capacity (Ah)')
plt.title('Deep Learning: LSTM Neural Network Forecasting')
plt.legend()
plt.grid(True)
plt.show()
# py "c:\Users\Rishi Gupta\OneDrive\Documents\PBL\final_bms_lstm.py"