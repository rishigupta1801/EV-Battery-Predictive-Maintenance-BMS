import os
import zipfile
import scipy.io
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# --- NEW IMPORT ---
from xgboost import XGBRegressor

# --- Step 1: File Paths & Extraction ---
base_dir = r"C:\Users\Rishi Gupta\OneDrive\Documents\PBL"
zip_path = os.path.join(base_dir, "5.+Battery+Data+Set.zip")
extract_path = os.path.join(base_dir, "Battery_Data")

print("Checking and extracting datasets...")
if not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

for root, dirs, files in os.walk(extract_path):
    for file in files:
        if file.endswith('.zip'):
            nested_zip_path = os.path.join(root, file)
            inner_extract_path = os.path.join(root, file.replace('.zip', ''))
            if not os.path.exists(inner_extract_path):
                print(f"Extracting hidden nested zip: {file}...")
                with zipfile.ZipFile(nested_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(inner_extract_path)

# --- Step 2: Parse NASA MATLAB (.mat) File ---
mat_file_path = None
for root, dirs, files in os.walk(extract_path):
    if 'B0005.mat' in files:
        mat_file_path = os.path.join(root, 'B0005.mat')
        break

if not mat_file_path:
    raise FileNotFoundError("Could not find B0005.mat. Please check if the main zip file is corrupted.")

print(f"Loading data from: {mat_file_path}")
mat = scipy.io.loadmat(mat_file_path)

data = mat['B0005'][0, 0]['cycle'][0]
cycles = []
capacities = []
cycle_count = 1

for i in range(data.size):
    row = data[i]
    if row['type'][0] == 'discharge':
        cap = row['data'][0][0]['Capacity'][0][0]
        cycles.append(cycle_count)
        capacities.append(cap)
        cycle_count += 1

df = pd.DataFrame({'Cycle': cycles, 'Capacity': capacities})

# --- Step 3: Train Both ML Models ---
X = df[['Cycle']]
y = df['Capacity']

# Split into 80% training data and 20% testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: Baseline Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

# Model 2: XGBoost Regressor (The PPT Promise)
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)

# --- Step 4: Validate & Evaluate ---
lr_rmse = mean_squared_error(y_test, lr_predictions) ** 0.5
lr_r2 = r2_score(y_test, lr_predictions)

xgb_rmse = mean_squared_error(y_test, xgb_predictions) ** 0.5
xgb_r2 = r2_score(y_test, xgb_predictions)

print("\n--- Model Performance Comparison ---")
print(f"Linear Regression -> RMSE: {lr_rmse:.4f} | R2 Score: {lr_r2:.4f}")
print(f"XGBoost Ensemble  -> RMSE: {xgb_rmse:.4f} | R2 Score: {xgb_r2:.4f}")

# --- Step 5: Visualize Degradation ---
plt.figure(figsize=(10, 6))
plt.plot(df['Cycle'], df['Capacity'], label='Actual Capacity Fade (B0005)', color='blue', alpha=0.5)

# Plot both trendlines for comparison
plt.plot(df['Cycle'], lr_model.predict(X), label='Linear Regression (Baseline)', color='red', linestyle='--')
plt.plot(df['Cycle'], xgb_model.predict(X), label='XGBoost (Ensemble)', color='green', linewidth=2)

plt.axhline(y=1.4, color='black', linestyle=':', label='End of Life (Failure Threshold)')
plt.xlabel('Discharge Cycles')
plt.ylabel('Capacity (Ah)')
plt.title('RUL Prediction: Linear Regression vs XGBoost')
plt.legend()
plt.grid(True)
plt.show()
# py "c:\Users\Rishi Gupta\OneDrive\Documents\PBL\baseline_model.py"