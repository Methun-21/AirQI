"""
Unit tests for AIRAWARE Machine Learning models and inference pipeline.
"""

import os
import joblib
import numpy as np
import pandas as pd
import pytest
from features import construct_feature_vector, FEATURE_COLUMNS


def test_models_directory_exists():
    assert os.path.exists("models")


def test_model_binaries_exist():
    for m in ["lgbm_model.pkl", "lgbm_p10.pkl", "lgbm_p50.pkl", "lgbm_p90.pkl", "features_list.pkl"]:
        path = os.path.join("models", m)
        assert os.path.exists(path), f"Missing model binary: {m}"


def test_quantile_inference_bounds():
    feats_path = os.path.join("models", "features_list.pkl")
    p10_model = joblib.load(os.path.join("models", "lgbm_p10.pkl"))
    p50_model = joblib.load(os.path.join("models", "lgbm_p50.pkl"))
    p90_model = joblib.load(os.path.join("models", "lgbm_p90.pkl"))
    feats = joblib.load(feats_path)
    
    vec = construct_feature_vector(
        lat=28.6315, lon=77.2167, temp=25.0, humidity=60.0, wind=3.0,
        lag1=150.0, lag3=140.0, lag24=130.0, roll6=145.0, roll_std6=5.0
    )
    
    df = pd.DataFrame([vec], columns=feats)
    p10 = float(np.expm1(p10_model.predict(df))[0])
    p50 = float(np.expm1(p50_model.predict(df))[0])
    p90 = float(np.expm1(p90_model.predict(df))[0])
    
    assert 0.0 <= p10 <= p90
    assert p10 <= p50 <= p90 + 5.0
