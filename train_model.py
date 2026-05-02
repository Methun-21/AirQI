import pandas as pd
import numpy as np
import joblib
from geopy.distance import geodesic
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

print("--- 1. LOADING & CLEANING ---")
df = pd.read_csv("delhi_aqi_data_waqi.csv")
df['time'] = pd.to_datetime(df['time'], errors='coerce')
df.dropna(subset=['time'], inplace=True)

# Geographic Features
major_roads = [(28.6473, 77.3155), (28.6307, 77.2479), (28.5372, 77.2882), (28.7011, 77.1611), (28.5932, 77.1636), (28.5485, 77.2520), (28.6315, 77.2167), (28.6517, 77.1907)]
def get_dist(lat, lon):
    return min(geodesic((lat, lon), r).kilometers for r in major_roads)
df['distance_to_major_road'] = df.apply(lambda row: get_dist(row['lat'], row['lon']), axis=1)

print("--- 2. ADVANCED FEATURE ENGINEERING ---")
target = 'pm2_5'
df['hour'] = df['time'].dt.hour
df['month'] = df['time'].dt.month
df['day_of_week'] = df['time'].dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['is_rush_hour'] = df['hour'].apply(lambda x: 1 if (8 <= x <= 11) or (17 <= x <= 20) else 0)

# Cyclic Encoding
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)

# DEEP Spatiotemporal Lags (per location)
df = df.sort_values(['location', 'time'])
df['pm2_5_lag1'] = df.groupby('location')[target].shift(1)
df['pm2_5_lag3'] = df.groupby('location')[target].shift(3)
df['pm2_5_lag24'] = df.groupby('location')[target].shift(24) # 24h memory
df['rolling_6h'] = df.groupby('location')[target].transform(lambda x: x.rolling(window=6, min_periods=1).mean())
df['rolling_std_6h'] = df.groupby('location')[target].transform(lambda x: x.rolling(window=6, min_periods=1).std().fillna(0)) # Volatility

# Multi-Variable Interactions
df['temp_hum'] = df['temp'] * df['humidity']
df['wind_temp'] = df['wind'] * df['temp']

features = [
    'lat', 'lon', 'temp', 'humidity', 'wind', 'hour_sin', 'hour_cos', 
    'distance_to_major_road', 'pm2_5_lag1', 'pm2_5_lag3', 'pm2_5_lag24',
    'rolling_6h', 'rolling_std_6h', 'temp_hum', 'wind_temp', 'month', 
    'is_weekend', 'is_rush_hour'
]
df.dropna(subset=features + [target], inplace=True)

print("--- 3. STRATEGIC DATA SPLIT & TARGET TRANSFORMATION ---")
df = df.sort_values('time')
X = df[features]
# Log1p Transformation to handle extreme PM2.5 spikes safely
y = np.log1p(df[target]) 

split = int(len(df) * 0.85)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]
actual_y_test = np.expm1(y_test) # Keep actual values for final evaluation

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

print("--- 4. DEEP HYPERPARAMETER OPTIMIZATION (TSS) ---")
tss = TimeSeriesSplit(n_splits=3)
n_iter_search = 15 # Deep search

# 1. Random Forest Tuning
print("Tuning RandomForest...")
rf_grid = {'n_estimators': [100, 300], 'max_depth': [20, 30, None], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2]}
rf_cv = RandomizedSearchCV(RandomForestRegressor(random_state=42, n_jobs=-1), rf_grid, cv=tss, n_iter=n_iter_search, random_state=42)
rf_cv.fit(X_train, y_train)
best_rf = rf_cv.best_estimator_

# 2. XGBoost Tuning
print("Tuning XGBoost...")
xgb_grid = {'n_estimators': [200, 500], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [5, 7, 9], 'subsample': [0.8, 1.0], 'colsample_bytree': [0.8, 1.0]}
xgb_cv = RandomizedSearchCV(XGBRegressor(random_state=42, n_jobs=-1), xgb_grid, cv=tss, n_iter=n_iter_search, random_state=42)
xgb_cv.fit(X_train, y_train)
best_xgb = xgb_cv.best_estimator_

# 3. LightGBM Tuning
print("Tuning LightGBM...")
lgb_grid = {'n_estimators': [200, 500], 'learning_rate': [0.01, 0.05, 0.1], 'num_leaves': [31, 63], 'max_depth': [-1, 10]}
lgb_cv = RandomizedSearchCV(LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1), lgb_grid, cv=tss, n_iter=n_iter_search, random_state=42)
lgb_cv.fit(X_train, y_train)
best_lgb = lgb_cv.best_estimator_

# 4. CatBoost (No deep tuning needed, usually great out-of-box with fast learning)
print("Training CatBoost...")
best_cat = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, random_state=42, verbose=0, thread_count=-1)
best_cat.fit(X_train, y_train)

print("--- 5. ADVANCED STACKING ENSEMBLE ---")
estimators = [
    ('rf', best_rf),
    ('xgb', best_xgb),
    ('lgb', best_lgb),
    ('cat', best_cat)
]
# Ridge Meta-Learner over 4 State-of-the-Art models
stack = StackingRegressor(estimators=estimators, final_estimator=Ridge(), n_jobs=-1)
stack.fit(X_train, y_train)

# EVALUATION (Convert back from Log-Scale)
log_preds = stack.predict(X_test)
actual_preds = np.expm1(log_preds)
mae = mean_absolute_error(actual_y_test, actual_preds)

joblib.dump(best_rf, 'models/rf_model.pkl')
joblib.dump(best_xgb, 'models/xgb_model.pkl')
joblib.dump(best_lgb, 'models/lgb_model.pkl')
joblib.dump(best_cat, 'models/cat_model.pkl')
joblib.dump(stack, 'models/stacked_model.pkl')
joblib.dump(features, 'models/features_list.pkl')

print(f"\n🚀 SUPERCHARGED Model Saved. Final MAE: {mae:.2f}")
print(f"Features Used: {len(features)}")
