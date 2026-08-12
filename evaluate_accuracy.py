"""
AIRAWARE ML Accuracy Evaluation Script
Evaluates saved Stacking Ensemble on unseen validation split and generates visual accuracy metrics.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

from features import engineer_dataframe_features, FEATURE_COLUMNS

warnings.filterwarnings('ignore')

def evaluate_models(data_path="delhi_aqi_data_waqi.csv", model_dir="models"):
    print("--- AIRAWARE: ML ACCURACY EVALUATION ---")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file missing: {data_path}")
        
    df = pd.read_csv(data_path)
    target = 'pm2_5'
    
    df = engineer_dataframe_features(df, target_col=target)
    features = FEATURE_COLUMNS
    
    model_path = os.path.join(model_dir, 'stacked_model.pkl')
    if not os.path.exists(model_path):
        print("Models not found! Training model first...")
        from train_model import train_pipeline
        train_pipeline(data_path, model_dir)
        
    stacked_model = joblib.load(model_path)
    df.dropna(subset=features + [target], inplace=True)
    df = df.sort_values('time')
    
    split = int(len(df) * 0.85)
    X_test = df[features].iloc[split:]
    y_test_actual = df[target].iloc[split:].values
    
    print(f"Evaluating ensemble on {len(X_test)} unseen test samples...")
    
    log_preds = stacked_model.predict(X_test)
    y_pred = np.expm1(log_preds)
    
    mae = mean_absolute_error(y_test_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
    r2 = r2_score(y_test_actual, y_pred)
    
    print("\n--- PERFORMANCE METRICS ---")
    print(f"MAE (Mean Absolute Error): {mae:.2f} µg/m³")
    print(f"RMSE (Root Mean Squared Error): {rmse:.2f} µg/m³")
    print(f"R² Score: {r2:.3f}")
    
    with open("accuracy_report.txt", "w", encoding="utf-8") as f:
        f.write("AIRAWARE - Model Accuracy Report\n")
        f.write("================================\n")
        f.write(f"Test Samples: {len(X_test)}\n")
        f.write(f"MAE (Mean Absolute Error): {mae:.2f} ug/m3\n")
        f.write(f"RMSE (Root Mean Squared Error): {rmse:.2f} ug/m3\n")
        f.write(f"R^2 Score: {r2:.3f}\n\n")
        f.write("Sample Predictions (Actual vs Predicted):\n")
        for i in range(min(10, len(y_test_actual))):
            f.write(f"Actual: {y_test_actual[i]:.1f} | Predicted: {y_pred[i]:.1f} (Diff: {abs(y_test_actual[i]-y_pred[i]):.1f})\n")

    plt.figure(figsize=(10, 6))
    sns.set_style("darkgrid")
    
    plot_limit = min(100, len(y_test_actual))
    plt.plot(y_test_actual[:plot_limit], label='Actual PM2.5 (Ground Truth)', color='#ef4444', linewidth=2)
    plt.plot(y_pred[:plot_limit], label='Predicted PM2.5 (Ensemble)', color='#0ea5e9', linestyle='--', linewidth=2)
    
    plt.title('Predictive Accuracy: Actual vs Forecasted PM2.5 Levels', fontsize=14, pad=15)
    plt.ylabel('PM2.5 Concentration (ug/m3)', fontsize=12)
    plt.xlabel('Time (Test Samples)', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig('accuracy_plot.png', dpi=300)
    print(" Saved visual metrics to 'accuracy_plot.png' and 'accuracy_report.txt'")
    
    return {"mae": mae, "rmse": rmse, "r2": r2}

if __name__ == "__main__":
    evaluate_models()
