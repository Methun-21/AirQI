import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("--- AIRAWARE: ML ACCURACY EVALUATION ---")

# 1. Load Data
print("Loading latest data for testing...")
df = pd.read_csv("delhi_aqi_data_waqi.csv")
df['time'] = pd.to_datetime(df['time'], errors='coerce')
df.dropna(subset=['time'], inplace=True)

target = 'pm2_5'

# Re-create features needed for evaluation
# (Note: In a production pipeline, this feature engineering would be in a shared module)
df['hour'] = df['time'].dt.hour
df['month'] = df['time'].dt.month
df['day_of_week'] = df['time'].dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['is_rush_hour'] = df['hour'].apply(lambda x: 1 if (8 <= x <= 11) or (17 <= x <= 20) else 0)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)

# Simplistic distance mock for evaluation to match training
major_roads = [(28.6473, 77.3155), (28.6307, 77.2479), (28.5372, 77.2882), (28.7011, 77.1611), (28.5932, 77.1636), (28.5485, 77.2520)]
from geopy.distance import geodesic
def get_dist(lat, lon):
    return min(geodesic((lat, lon), r).kilometers for r in major_roads)
df['distance_to_major_road'] = df.apply(lambda row: get_dist(row['lat'], row['lon']), axis=1)

df = df.sort_values(['location', 'time'])
df['pm2_5_lag1'] = df.groupby('location')[target].shift(1)
df['pm2_5_lag3'] = df.groupby('location')[target].shift(3)
df['pm2_5_lag24'] = df.groupby('location')[target].shift(24)
df['rolling_6h'] = df.groupby('location')[target].transform(lambda x: x.rolling(window=6, min_periods=1).mean())
df['rolling_std_6h'] = df.groupby('location')[target].transform(lambda x: x.rolling(window=6, min_periods=1).std().fillna(0))
df['temp_hum'] = df['temp'] * df['humidity']
df['wind_temp'] = df['wind'] * df['temp']

try:
    features = joblib.load('models/features_list.pkl')
    stacked_model = joblib.load('models/stacked_model.pkl')
except:
    print("Models not found! Please run train_model.py first.")
    exit(1)

df.dropna(subset=features + [target], inplace=True)

# 2. Get the Testing Split (Last 15% of data like in training)
df = df.sort_values('time')
split = int(len(df) * 0.85)
X_test = df[features].iloc[split:]
y_test_actual = df[target].iloc[split:].values

print(f"Evaluating on {len(X_test)} unseen test samples...")

# 3. Predict
log_preds = stacked_model.predict(X_test)
y_pred = np.expm1(log_preds)

# 4. Calculate Metrics
mae = mean_absolute_error(y_test_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
r2 = r2_score(y_test_actual, y_pred)

print("\n--- RESULTS ---")
print(f"MAE (Mean Absolute Error): {mae:.2f} ug/m3")
print(f"RMSE (Root Mean Squared Error): {rmse:.2f} ug/m3")
print(f"R^2 Score: {r2:.3f}")

# Generate a report file
with open("accuracy_report.txt", "w") as f:
    f.write("AIRAWARE - Model Accuracy Report\n")
    f.write("================================\n")
    f.write(f"Test Samples: {len(X_test)}\n")
    f.write(f"MAE (Mean Absolute Error): {mae:.2f} ug/m3\n")
    f.write(f"RMSE (Root Mean Squared Error): {rmse:.2f} ug/m3\n")
    f.write(f"R^2 Score: {r2:.3f}\n\n")
    f.write("Sample Predictions (Actual vs Predicted):\n")
    for i in range(10):
        f.write(f"Actual: {y_test_actual[i]:.1f} | Predicted: {y_pred[i]:.1f} (Diff: {abs(y_test_actual[i]-y_pred[i]):.1f})\n")

# 5. Plotting Actual vs Predicted
plt.figure(figsize=(10, 6))
sns.set_style("darkgrid")

# We plot the first 100 test samples to see the tracking capability
plot_limit = min(100, len(y_test_actual))
plt.plot(y_test_actual[:plot_limit], label='Actual PM2.5 (Ground Truth)', color='#ef4444', linewidth=2)
plt.plot(y_pred[:plot_limit], label='Predicted PM2.5 (Ensemble)', color='#0ea5e9', linestyle='--', linewidth=2)

plt.title('Predictive Accuracy: Actual vs Forecasted PM2.5 Levels', fontsize=14, pad=15)
plt.ylabel('PM2.5 Concentration (ug/m3)', fontsize=12)
plt.xlabel('Time (Test Samples)', fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig('accuracy_plot.png', dpi=300)
print("\n--- Saved accuracy visualization to 'accuracy_plot.png' ---")
print("--- Saved detailed metrics to 'accuracy_report.txt' ---")
