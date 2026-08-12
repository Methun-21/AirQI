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


def test_stacked_model_file_exists():
    model_path = os.path.join("models", "stacked_model.pkl")
    assert os.path.exists(model_path), "stacked_model.pkl missing. Run train_model.py to build model binary."


def test_model_inference_output():
    model_path = os.path.join("models", "stacked_model.pkl")
    feats_path = os.path.join("models", "features_list.pkl")
    
    model = joblib.load(model_path)
    feats = joblib.load(feats_path)
    
    vec = construct_feature_vector(
        lat=28.6315, lon=77.2167, temp=25.0, humidity=60.0, wind=3.0,
        lag1=150.0, lag3=140.0, lag24=130.0, roll6=145.0, roll_std6=5.0
    )
    
    df = pd.DataFrame([vec], columns=feats)
    log_pred = model.predict(df)
    pred_pm25 = float(np.expm1(log_pred)[0])
    
    assert isinstance(pred_pm25, float)
    assert 0.0 <= pred_pm25 <= 1000.0
