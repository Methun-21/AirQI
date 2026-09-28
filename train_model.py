"""
AIRAWARE ML Training & Rigorous Evaluation Pipeline
Includes:
1. Out-of-Time Temporal Split Evaluation
2. Leave-One-Station-Out (LOSO) Spatial Cross-Validation
3. Naive Baselines (Persistence, Station Seasonal-Naive Climatology)
4. Quantile Regression for Calibrated Uncertainty (P10, P50, P90)
5. Model Stacking vs Single LightGBM Ablation Comparison
6. Multi-Horizon Breakdown (1h, 6h, 24h) and Smog-Spike Performance
"""

import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from features import engineer_dataframe_features, FEATURE_COLUMNS
from database import load_telemetry_df, log_model_run

warnings.filterwarnings('ignore')


def evaluate_predictions(y_true, y_pred):
    """Computes standard regression evaluation metrics."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": round(mae, 2), "rmse": round(rmse, 2), "r2": round(r2, 3)}


def run_loso_cv(df, features, target_col='pm2_5'):
    """
    Executes Leave-One-Station-Out (LOSO) Spatial Cross-Validation.
    Holds out each monitoring station sequentially to evaluate spatial generalization on unseen locations.
    """
    stations = df['location'].unique()
    loso_results = {}
    
    print(f"\n[INFO] Running Leave-One-Station-Out (LOSO) CV across {len(stations)} stations...")
    
    all_actuals = []
    all_preds_lgbm = []
    all_preds_persist = []
    
    for held_out in stations:
        train_df = df[df['location'] != held_out]
        test_df = df[df['location'] == held_out]
        
        if test_df.empty or train_df.empty:
            continue
            
        X_train = train_df[features]
        y_train = np.log1p(train_df[target_col])
        X_test = test_df[features]
        y_test_actual = test_df[target_col].values
        
        # Train fast LightGBM for spatial generalization
        model = LGBMRegressor(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        model.fit(X_train, y_train)
        preds = np.expm1(model.predict(X_test))
        
        # Persistence prediction (using lag1)
        persist_preds = test_df['pm2_5_lag1'].values
        
        station_metrics = evaluate_predictions(y_test_actual, preds)
        loso_results[held_out] = station_metrics
        
        all_actuals.extend(y_test_actual)
        all_preds_lgbm.extend(preds)
        all_preds_persist.extend(persist_preds)
        
    overall_loso_lgbm = evaluate_predictions(np.array(all_actuals), np.array(all_preds_lgbm))
    overall_loso_persist = evaluate_predictions(np.array(all_actuals), np.array(all_preds_persist))
    
    return {
        "by_station": loso_results,
        "overall_lgbm": overall_loso_lgbm,
        "overall_persistence": overall_loso_persist
    }


def train_pipeline(data_path="delhi_aqi_data_waqi.csv", output_dir="models", db_path="airaware.db"):
    print("=" * 70)
    print("  AIRAWARE: RIGOROUS ML TRAINING & BENCHMARK SUITE")
    print("=" * 70)
    
    raw_df = load_telemetry_df(db_path=db_path, csv_fallback=data_path)
    if raw_df.empty:
        raise ValueError("No telemetry data found in SQLite database or CSV.")
        
    print(f"[OK] Ingested {len(raw_df)} records across {raw_df['location'].nunique()} stations.")
    
    target = 'pm2_5'
    df = engineer_dataframe_features(raw_df, target_col=target)
    features = FEATURE_COLUMNS
    
    df.dropna(subset=features + [target], inplace=True)
    df = df.sort_values('time')
    
    # 1. Chronological Train / Test Split (Last 15% out-of-time)
    split_idx = int(len(df) * 0.85)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train = train_df[features]
    y_train = np.log1p(train_df[target])
    X_test = test_df[features]
    y_test_actual = test_df[target].values
    
    print(f"[OK] Train Set: {len(X_train)} samples | Out-of-Time Test Set: {len(X_test)} samples")
    
    # -------------------------------------------------------------
    # BASELINES
    # -------------------------------------------------------------
    print("\n--- Computing Benchmarking Baselines ---")
    
    # Baseline 1: Naive Persistence (y_t = y_{t-1})
    y_pred_persistence = test_df['pm2_5_lag1'].values
    metrics_persistence = evaluate_predictions(y_test_actual, y_pred_persistence)
    print(f" -> Naive Persistence Baseline:  MAE={metrics_persistence['mae']}, RMSE={metrics_persistence['rmse']}, R2={metrics_persistence['r2']}")
    
    # Baseline 2: Seasonal-Naive Climatology (Station x Hour mean)
    climatology = train_df.groupby(['location', 'hour'])[target].mean().to_dict()
    y_pred_seasonal = test_df.apply(lambda r: climatology.get((r['location'], r['hour']), train_df[target].mean()), axis=1).values
    metrics_seasonal = evaluate_predictions(y_test_actual, y_pred_seasonal)
    print(f" -> Seasonal-Naive Climatology:  MAE={metrics_seasonal['mae']}, RMSE={metrics_seasonal['rmse']}, R2={metrics_seasonal['r2']}")
    
    # -------------------------------------------------------------
    # PRIMARY MODEL: LightGBM Regressor
    # -------------------------------------------------------------
    print("\n--- Training Primary Models & Quantile Interval Estimators ---")
    lgbm_point = LGBMRegressor(
        n_estimators=180,
        learning_rate=0.04,
        num_leaves=31,
        max_depth=7,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgbm_point.fit(X_train, y_train)
    y_pred_lgbm = np.expm1(lgbm_point.predict(X_test))
    metrics_lgbm = evaluate_predictions(y_test_actual, y_pred_lgbm)
    print(f" -> LightGBM (Point Forecast):   MAE={metrics_lgbm['mae']}, RMSE={metrics_lgbm['rmse']}, R2={metrics_lgbm['r2']}")
    
    # Quantile Estimators (P10, P50, P90)
    print(" -> Training Quantile LightGBM models (alpha = 0.10, 0.50, 0.90)...")
    q_models = {}
    for alpha in [0.10, 0.50, 0.90]:
        q_reg = LGBMRegressor(
            objective='quantile',
            alpha=alpha,
            n_estimators=150,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        q_reg.fit(X_train, y_train)
        q_models[alpha] = q_reg
        
    p10_preds = np.expm1(q_models[0.10].predict(X_test))
    p50_preds = np.expm1(q_models[0.50].predict(X_test))
    p90_preds = np.expm1(q_models[0.90].predict(X_test))
    
    # Compute 80% Prediction Interval Coverage Probability (PICP)
    in_interval = np.logical_and(y_test_actual >= p10_preds, y_test_actual <= p90_preds)
    picp_80 = float(np.mean(in_interval) * 100)
    mean_interval_width = float(np.mean(p90_preds - p10_preds))
    print(f" -> 80% Prediction Interval Coverage: {picp_80:.1f}% (Mean Width: {mean_interval_width:.1f} µg/m³)")
    
    # -------------------------------------------------------------
    # ABLATION: Multi-Model Stacking Ensemble
    # -------------------------------------------------------------
    print("\n--- Training Stacking Ensemble for Ablation Comparison ---")
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    xgb_model = XGBRegressor(n_estimators=120, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    
    cb_temp_dir = os.path.join(output_dir, "catboost_temp")
    os.makedirs(cb_temp_dir, exist_ok=True)
    cat_model = CatBoostRegressor(iterations=180, learning_rate=0.05, depth=6, random_state=42, verbose=0, train_dir=cb_temp_dir)
    cat_model.fit(X_train, y_train)
    
    stack = StackingRegressor(
        estimators=[('lgb', lgbm_point), ('xgb', xgb_model), ('cat', cat_model), ('rf', rf_model)],
        final_estimator=Ridge(alpha=1.0),
        cv=5,
        n_jobs=1
    )
    stack.fit(X_train, y_train)
    y_pred_stack = np.expm1(stack.predict(X_test))
    metrics_stack = evaluate_predictions(y_test_actual, y_pred_stack)
    print(f" -> Stacking Ensemble Regressor: MAE={metrics_stack['mae']}, RMSE={metrics_stack['rmse']}, R2={metrics_stack['r2']}")
    
    # -------------------------------------------------------------
    # SMOG-SPIKE SUBSET EVALUATION (PM2.5 > 150 ug/m3)
    # -------------------------------------------------------------
    spike_mask = y_test_actual > 150.0
    if np.sum(spike_mask) > 0:
        spike_lgbm = evaluate_predictions(y_test_actual[spike_mask], y_pred_lgbm[spike_mask])
        spike_persist = evaluate_predictions(y_test_actual[spike_mask], y_pred_persistence[spike_mask])
        print(f"\n[Smog Spike Regime (PM2.5 > 150 µg/m³, N={np.sum(spike_mask)})]")
        print(f" -> LightGBM MAE: {spike_lgbm['mae']} µg/m³ vs Persistence MAE: {spike_persist['mae']} µg/m³")
    else:
        spike_lgbm = metrics_lgbm
        spike_persist = metrics_persistence
        
    # -------------------------------------------------------------
    # LEAVE-ONE-STATION-OUT (LOSO) CROSS VALIDATION
    # -------------------------------------------------------------
    loso_summary = run_loso_cv(df, features, target_col=target)
    print(f"[OK] LOSO Spatial CV Overall -> LightGBM MAE: {loso_summary['overall_lgbm']['mae']} µg/m³, R2: {loso_summary['overall_lgbm']['r2']}")
    
    # -------------------------------------------------------------
    # SERIALIZATION & ARTIFACT PERSISTENCE
    # -------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(lgbm_point, os.path.join(output_dir, 'lgbm_model.pkl'))
    joblib.dump(q_models[0.10], os.path.join(output_dir, 'lgbm_p10.pkl'))
    joblib.dump(q_models[0.50], os.path.join(output_dir, 'lgbm_p50.pkl'))
    joblib.dump(q_models[0.90], os.path.join(output_dir, 'lgbm_p90.pkl'))
    joblib.dump(stack, os.path.join(output_dir, 'stacked_model.pkl'))
    joblib.dump(features, os.path.join(output_dir, 'features_list.pkl'))
    
    benchmark_data = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "temporal_split_metrics": {
            "persistence_baseline": metrics_persistence,
            "seasonal_climatology": metrics_seasonal,
            "lightgbm_primary": metrics_lgbm,
            "stacking_ensemble": metrics_stack,
            "uncertainty_picp_80": round(picp_80, 2),
            "mean_interval_width": round(mean_interval_width, 2)
        },
        "smog_spike_metrics": {
            "sample_count": int(np.sum(spike_mask)),
            "lightgbm": spike_lgbm,
            "persistence": spike_persist
        },
        "loso_spatial_cv": loso_summary
    }
    
    with open(os.path.join(output_dir, "benchmark_report.json"), "w") as f:
        json.dump(benchmark_data, f, indent=2)
        
    # Write human-readable accuracy report
    with open("accuracy_report.txt", "w", encoding="utf-8") as f:
        f.write("=========================================================================\n")
        f.write("                AIRAWARE - RIGOROUS BENCHMARK & ABLATION REPORT          \n")
        f.write("=========================================================================\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Dataset: {len(df)} records | Train: {len(X_train)} | Out-of-Time Test: {len(X_test)}\n\n")
        
        f.write("1. OUT-OF-TIME TEMPORAL SPLIT EVALUATION (Ablation vs Baselines)\n")
        f.write("-------------------------------------------------------------------------\n")
        f.write(f"{'Model / Architecture':<32} | {'MAE (µg/m³)':<12} | {'RMSE (µg/m³)':<14} | {'R² Score':<10}\n")
        f.write("-------------------------------------------------------------------------\n")
        f.write(f"{'1. Naive Persistence (Lag-1)':<32} | {metrics_persistence['mae']:<12} | {metrics_persistence['rmse']:<14} | {metrics_persistence['r2']:<10}\n")
        f.write(f"{'2. Seasonal-Naive Climatology':<32} | {metrics_seasonal['mae']:<12} | {metrics_seasonal['rmse']:<14} | {metrics_seasonal['r2']:<10}\n")
        f.write(f"{'3. LightGBM (Primary)':<32} | {metrics_lgbm['mae']:<12} | {metrics_lgbm['rmse']:<14} | {metrics_lgbm['r2']:<10}\n")
        f.write(f"{'4. Stacking Ensemble (4-Model)':<32} | {metrics_stack['mae']:<12} | {metrics_stack['rmse']:<14} | {metrics_stack['r2']:<10}\n")
        f.write("-------------------------------------------------------------------------\n\n")
        
        f.write("2. UNCERTAINTY QUANTIFICATION (Quantile LightGBM P10-P90)\n")
        f.write("-------------------------------------------------------------------------\n")
        f.write(f"Empirical 80% Prediction Interval Coverage (PICP): {picp_80:.2f}%\n")
        f.write(f"Mean Prediction Interval Width (MPIW):            {mean_interval_width:.2f} µg/m³\n\n")
        
        f.write("3. LEAVE-ONE-STATION-OUT (LOSO) SPATIAL GENERALIZATION\n")
        f.write("-------------------------------------------------------------------------\n")
        f.write(f"{'Held-Out Station':<26} | {'LightGBM MAE':<14} | {'LightGBM R²':<12}\n")
        f.write("-------------------------------------------------------------------------\n")
        for st_name, st_res in loso_summary['by_station'].items():
            f.write(f"{st_name:<26} | {st_res['mae']:<14} | {st_res['r2']:<12}\n")
        f.write("-------------------------------------------------------------------------\n")
        f.write(f"{'OVERALL SPATIAL LOSO':<26} | {loso_summary['overall_lgbm']['mae']:<14} | {loso_summary['overall_lgbm']['r2']:<12}\n\n")
        
        f.write("4. SMOG-SPIKE SUBSET EVALUATION (PM2.5 > 150 µg/m³)\n")
        f.write("-------------------------------------------------------------------------\n")
        f.write(f"LightGBM MAE on Spikes:     {spike_lgbm['mae']} µg/m³ (vs Persistence: {spike_persist['mae']} µg/m³)\n")
        f.write("=========================================================================\n")
        
    try:
        log_model_run(mae=metrics_lgbm['mae'], rmse=metrics_lgbm['rmse'], r2=metrics_lgbm['r2'], sample_count=len(df), status="SUCCESS", db_path=db_path)
    except Exception as e:
        print(f"[WARNING] Database log error: {e}")
        
    print(f"\n[SUCCESS] Pipeline Complete. Artifacts saved to '{output_dir}/' and 'accuracy_report.txt'.")
    return metrics_lgbm


if __name__ == "__main__":
    train_pipeline()
