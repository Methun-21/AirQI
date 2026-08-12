"""
AIRAWARE ML Training Pipeline
Loads training data from SQLite database (airaware.db), trains base gradient boosting models and Ridge Stacking Ensemble,
evaluates performance, and logs run metrics to SQLite.
"""

import os
import joblib
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from features import engineer_dataframe_features, FEATURE_COLUMNS
from database import load_telemetry_df, log_model_run

warnings.filterwarnings('ignore')


def train_pipeline(data_path="delhi_aqi_data_waqi.csv", output_dir="models", db_path="airaware.db"):
    print("[INFO] Loading training dataset from SQLite database / CSV...")
    raw_df = load_telemetry_df(db_path=db_path, csv_fallback=data_path)
    
    if raw_df.empty:
        raise ValueError("No telemetry data found in SQLite database or CSV.")
        
    print(f"[OK] Loaded {len(raw_df)} total records for feature engineering.")

    print("[INFO] Running feature engineering pipeline...")
    target = 'pm2_5'
    df = engineer_dataframe_features(raw_df, target_col=target)
    
    features = FEATURE_COLUMNS
    df.dropna(subset=features + [target], inplace=True)
    df = df.sort_values('time')
    
    X = df[features]
    y = np.log1p(df[target])  # Log1p transformation for volatility handling

    split = int(len(df) * 0.85)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    actual_y_test = np.expm1(y_test)

    print(f"[OK] Training set: {len(X_train)} samples | Validation set: {len(X_test)} samples")

    print("[INFO] Training Tier-1 Base Estimators...")
    
    # 1. Random Forest Regressor
    print(" -> Training RandomForest...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=3,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    # 2. XGBoost Regressor
    print(" -> Training XGBoost...")
    xgb_model = XGBRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)

    # 3. LightGBM Regressor
    print(" -> Training LightGBM...")
    lgb_model = LGBMRegressor(
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=7,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)

    # 4. CatBoost Regressor
    print(" -> Training CatBoost...")
    cb_temp_dir = os.path.join(output_dir, "catboost_temp")
    os.makedirs(cb_temp_dir, exist_ok=True)
    cat_model = CatBoostRegressor(
        iterations=250,
        learning_rate=0.05,
        depth=6,
        random_state=42,
        verbose=0,
        train_dir=cb_temp_dir
    )
    cat_model.fit(X_train, y_train)

    print("[INFO] Training Tier-2 Stacking Ensemble Regressor...")
    estimators = [
        ('rf', rf_model),
        ('xgb', xgb_model),
        ('lgb', lgb_model),
        ('cat', cat_model)
    ]
    
    stack = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=5,
        n_jobs=1
    )
    stack.fit(X_train, y_train)

    # Evaluate predictions
    log_preds = stack.predict(X_test)
    actual_preds = np.expm1(log_preds)
    
    mae = float(mean_absolute_error(actual_y_test, actual_preds))
    rmse = float(np.sqrt(mean_squared_error(actual_y_test, actual_preds)))
    r2 = float(r2_score(actual_y_test, actual_preds))

    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(rf_model, os.path.join(output_dir, 'rf_model.pkl'))
    joblib.dump(xgb_model, os.path.join(output_dir, 'xgb_model.pkl'))
    joblib.dump(lgb_model, os.path.join(output_dir, 'lgb_model.pkl'))
    joblib.dump(cat_model, os.path.join(output_dir, 'cat_model.pkl'))
    joblib.dump(stack, os.path.join(output_dir, 'stacked_model.pkl'))
    joblib.dump(features, os.path.join(output_dir, 'features_list.pkl'))

    # Log training metrics to SQLite DB
    try:
        log_model_run(mae=mae, rmse=rmse, r2=r2, sample_count=len(df), status="SUCCESS", db_path=db_path)
        print(f"[OK] Logged model run to database: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.3f}")
    except Exception as e:
        print(f"[WARNING] Could not log model run to database: {e}")

    print(f"[SUCCESS] Stacking Ensemble Trained & Saved to '{output_dir}/'. MAE: {mae:.2f}")
    return mae


if __name__ == "__main__":
    train_pipeline()
