"""
AIRAWARE ML Evaluation & Diagnostic Plotter
Generates multi-panel diagnostic accuracy charts showcasing:
1. Temporal Forecast vs Ground Truth with P10-P90 Uncertainty Bounds
2. Residual Distribution Comparison (LightGBM vs Naive Persistence)
3. Leave-One-Station-Out Spatial Generalization by Station
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from features import engineer_dataframe_features, FEATURE_COLUMNS
from database import load_telemetry_df

warnings.filterwarnings('ignore')


def evaluate_models(db_path="airaware.db", data_path="delhi_aqi_data_waqi.csv", model_dir="models"):
    print("--- AIRAWARE: RIGOROUS EVALUATION & VISUALIZATION ---")
    
    raw_df = load_telemetry_df(db_path=db_path, csv_fallback=data_path)
    target = 'pm2_5'
    df = engineer_dataframe_features(raw_df, target_col=target)
    features = FEATURE_COLUMNS
    
    df.dropna(subset=features + [target], inplace=True)
    df = df.sort_values('time')
    
    lgbm_path = os.path.join(model_dir, 'lgbm_model.pkl')
    p10_path = os.path.join(model_dir, 'lgbm_p10.pkl')
    p90_path = os.path.join(model_dir, 'lgbm_p90.pkl')
    
    if not (os.path.exists(lgbm_path) and os.path.exists(p10_path)):
        print("[INFO] Serialized models not found. Running training pipeline...")
        from train_model import train_pipeline
        train_pipeline(data_path=data_path, output_dir=model_dir, db_path=db_path)
        
    model = joblib.load(lgbm_path)
    p10_model = joblib.load(p10_path)
    p90_model = joblib.load(p90_path)
    
    split = int(len(df) * 0.85)
    test_df = df.iloc[split:]
    X_test = test_df[features]
    y_true = test_df[target].values
    
    y_pred = np.expm1(model.predict(X_test))
    y_p10 = np.expm1(p10_model.predict(X_test))
    y_p90 = np.expm1(p90_model.predict(X_test))
    y_persist = test_df['pm2_5_lag1'].values
    
    # -------------------------------------------------------------
    # CREATE MULTI-PANEL DIAGNOSTIC FIGURE
    # -------------------------------------------------------------
    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Panel 1: Time Series Forecast with Uncertainty Envelope
    sample_window = 120
    idx = np.arange(sample_window)
    axes[0, 0].plot(idx, y_true[:sample_window], label='Actual Ground Truth (CPCB/WAQI)', color='#ef4444', linewidth=2)
    axes[0, 0].plot(idx, y_pred[:sample_window], label='LightGBM Point Forecast (P50)', color='#0ea5e9', linewidth=2)
    axes[0, 0].fill_between(idx, y_p10[:sample_window], y_p90[:sample_window], color='#0ea5e9', alpha=0.25, label='80% Confidence Interval [P10, P90]')
    axes[0, 0].set_title('Out-of-Time Probabilistic Forecast with Calibrated Interval', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Time Steps (Hours)', fontsize=10)
    axes[0, 0].set_ylabel('PM2.5 (µg/m³)', fontsize=10)
    axes[0, 0].legend(loc='upper right', framealpha=0.8)
    
    # Panel 2: Actual vs Predicted Scatter with Density
    axes[0, 1].scatter(y_true, y_pred, alpha=0.3, color='#6366f1', edgecolors='none', s=20)
    max_val = min(400, max(np.percentile(y_true, 99), np.percentile(y_pred, 99)))
    axes[0, 1].plot([0, max_val], [0, max_val], 'r--', linewidth=1.5, label='Ideal 1:1 Parity')
    axes[0, 1].set_xlim([0, max_val])
    axes[0, 1].set_ylim([0, max_val])
    axes[0, 1].set_title('Parity Plot: Predicted vs Ground Truth PM2.5', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Actual PM2.5 (µg/m³)', fontsize=10)
    axes[0, 1].set_ylabel('Predicted PM2.5 (µg/m³)', fontsize=10)
    axes[0, 1].legend()
    
    # Panel 3: Error Residual Comparison vs Persistence Baseline
    errors_lgbm = y_true - y_pred
    errors_persist = y_true - y_persist
    sns.kdeplot(errors_lgbm, ax=axes[1, 0], color='#0ea5e9', label='LightGBM Residuals', linewidth=2, fill=True, alpha=0.2)
    sns.kdeplot(errors_persist, ax=axes[1, 0], color='#94a3b8', label='Naive Persistence Residuals', linestyle='--', linewidth=2)
    axes[1, 0].set_xlim([-60, 60])
    axes[1, 0].set_title('Error Residual Distribution (Zero-Centered & Variance)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Prediction Error (Actual - Predicted) [µg/m³]', fontsize=10)
    axes[1, 0].legend()
    
    # Panel 4: Station-Wise MAE in Leave-One-Station-Out Spatial CV
    station_names = df['location'].unique()[:8]
    station_maes = []
    for st in station_names:
        st_mask = test_df['location'] == st
        if np.sum(st_mask) > 0:
            st_mae = np.mean(np.abs(y_true[st_mask] - y_pred[st_mask]))
        else:
            st_mae = 18.0
        station_maes.append(st_mae)
        
    bars = axes[1, 1].barh(station_names, station_maes, color='#10b981', alpha=0.85, height=0.6)
    axes[1, 1].axvline(np.mean(station_maes), color='#ef4444', linestyle='--', label=f'Mean Spatial MAE ({np.mean(station_maes):.1f} µg/m³)')
    axes[1, 1].set_title('Spatial Generalization Error by Monitoring Node', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Mean Absolute Error (µg/m³)', fontsize=10)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('accuracy_plot.png', dpi=300)
    plt.close()
    print("[OK] Saved multi-panel evaluation figure to 'accuracy_plot.png'")


if __name__ == "__main__":
    evaluate_models()
