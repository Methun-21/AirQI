"""
AIRAWARE Feature Engineering Engine
Centralized, production-grade feature transformations shared across training, evaluation, and real-time inference.
"""

import numpy as np
import pandas as pd
from geopy.distance import geodesic
from datetime import datetime

# Major Delhi traffic corridors & high-density transport nodes
MAJOR_ROADS_AND_JUNCTIONS = [
    (28.6473, 77.3155),  # Anand Vihar Hub
    (28.6307, 77.2479),  # ITO Crossing
    (28.5372, 77.2882),  # Ashram Chowk
    (28.7011, 77.1611),  # Azadpur Mandi
    (28.5932, 77.1636),  # Dhaula Kuan Flyover
    (28.5485, 77.2520),  # Nehru Place Junction
    (28.6315, 77.2167),  # Connaught Place Outer Ring
    (28.6517, 77.1907)   # Karol Bagh Metro Corridor
]

FEATURE_COLUMNS = [
    'lat', 'lon', 'temp', 'humidity', 'wind', 'hour_sin', 'hour_cos',
    'distance_to_major_road', 'pm2_5_lag1', 'pm2_5_lag3', 'pm2_5_lag24',
    'rolling_6h', 'rolling_std_6h', 'temp_hum', 'wind_temp', 'month',
    'is_weekend', 'is_rush_hour'
]


def min_distance_to_road(lat: float, lon: float) -> float:
    """Calculates geodesic distance in kilometers to the nearest major traffic arterial."""
    return min(geodesic((lat, lon), r).kilometers for r in MAJOR_ROADS_AND_JUNCTIONS)


def compute_time_features(dt: datetime = None):
    """Generates temporal features including sin/cos cyclic hour encoding and rush-hour indicators."""
    if dt is None:
        dt = datetime.now()
    
    hour = dt.hour
    month = dt.month
    day_of_week = dt.weekday()
    
    is_weekend = 1 if day_of_week >= 5 else 0
    is_rush_hour = 1 if (8 <= hour <= 11) or (17 <= hour <= 20) else 0
    hour_sin = float(np.sin(2 * np.pi * hour / 24.0))
    hour_cos = float(np.cos(2 * np.pi * hour / 24.0))
    
    return {
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'month': month,
        'is_weekend': is_weekend,
        'is_rush_hour': is_rush_hour
    }


def construct_feature_vector(
    lat: float,
    lon: float,
    temp: float,
    humidity: float,
    wind: float,
    lag1: float,
    lag3: float,
    lag24: float,
    roll6: float,
    roll_std6: float,
    dt: datetime = None,
    dist_road_override: float = None
) -> list:
    """Constructs an 18-element feature list for model prediction matching exact training schema."""
    t_feats = compute_time_features(dt)
    dist_road = dist_road_override if dist_road_override is not None else min_distance_to_road(lat, lon)
    
    temp_hum = temp * humidity
    wind_temp = wind * temp
    
    return [
        lat, lon, temp, humidity, wind,
        t_feats['hour_sin'], t_feats['hour_cos'],
        dist_road, lag1, lag3, lag24, roll6, roll_std6,
        temp_hum, wind_temp, t_feats['month'],
        t_feats['is_weekend'], t_feats['is_rush_hour']
    ]


def engineer_dataframe_features(df: pd.DataFrame, target_col: str = 'pm2_5') -> pd.DataFrame:
    """Processes historical DataFrame for model training and evaluation with lag and rolling statistics."""
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df = df.dropna(subset=['time'])
    
    # Distance feature
    df['distance_to_major_road'] = df.apply(lambda row: min_distance_to_road(row['lat'], row['lon']), axis=1)
    
    # Time features
    df['hour'] = df['time'].dt.hour
    df['month'] = df['time'].dt.month
    df['day_of_week'] = df['time'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_rush_hour'] = df['hour'].apply(lambda x: 1 if (8 <= x <= 11) or (17 <= x <= 20) else 0)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    
    # Spatiotemporal Lags & Rolling Statistics
    df = df.sort_values(['location', 'time'])
    df['pm2_5_lag1'] = df.groupby('location')[target_col].shift(1)
    df['pm2_5_lag3'] = df.groupby('location')[target_col].shift(3)
    df['pm2_5_lag24'] = df.groupby('location')[target_col].shift(24)
    df['rolling_6h'] = df.groupby('location')[target_col].transform(lambda x: x.rolling(window=6, min_periods=1).mean())
    df['rolling_std_6h'] = df.groupby('location')[target_col].transform(lambda x: x.rolling(window=6, min_periods=1).std().fillna(0))
    
    # Interaction terms
    df['temp_hum'] = df['temp'] * df['humidity']
    df['wind_temp'] = df['wind'] * df['temp']
    
    return df
