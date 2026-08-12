"""
Unit tests for AIRAWARE feature engineering module.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from features import (
    min_distance_to_road,
    compute_time_features,
    construct_feature_vector,
    engineer_dataframe_features,
    FEATURE_COLUMNS,
    MAJOR_ROADS_AND_JUNCTIONS
)


def test_major_roads_list_non_empty():
    assert len(MAJOR_ROADS_AND_JUNCTIONS) > 0
    for lat, lon in MAJOR_ROADS_AND_JUNCTIONS:
        assert 28.0 <= lat <= 29.5
        assert 76.5 <= lon <= 78.0


def test_min_distance_to_road():
    # Test Connaught Place distance to roads (should be small > 0)
    cp_lat, cp_lon = 28.6315, 77.2167
    dist = min_distance_to_road(cp_lat, cp_lon)
    assert isinstance(dist, float)
    assert 0.0 <= dist <= 50.0


def test_compute_time_features():
    dt = datetime(2026, 8, 11, 14, 30)  # 2:30 PM (not rush hour, weekday)
    feats = compute_time_features(dt)
    
    assert "hour_sin" in feats
    assert "hour_cos" in feats
    assert feats["month"] == 8
    assert feats["is_weekend"] == 0
    assert feats["is_rush_hour"] == 0


def test_construct_feature_vector_length():
    vec = construct_feature_vector(
        lat=28.6315, lon=77.2167, temp=30.0, humidity=50.0, wind=4.0,
        lag1=150.0, lag3=140.0, lag24=160.0, roll6=145.0, roll_std6=10.0
    )
    assert len(vec) == len(FEATURE_COLUMNS)
    assert len(vec) == 18


def test_engineer_dataframe_features():
    sample_data = {
        'time': ['2026-08-11 10:00:00', '2026-08-11 11:00:00', '2026-08-11 12:00:00'],
        'location': ['Connaught Place', 'Connaught Place', 'Connaught Place'],
        'lat': [28.6315, 28.6315, 28.6315],
        'lon': [77.2167, 77.2167, 77.2167],
        'temp': [30.0, 31.0, 32.0],
        'humidity': [60.0, 58.0, 55.0],
        'wind': [3.5, 4.0, 4.2],
        'pm2_5': [120.0, 130.0, 125.0]
    }
    df = pd.DataFrame(sample_data)
    processed_df = engineer_dataframe_features(df, target_col='pm2_5')
    
    for col in FEATURE_COLUMNS:
        assert col in processed_df.columns
