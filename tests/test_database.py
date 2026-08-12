"""
Unit tests for AIRAWARE SQLite database module.
"""

import os
import gc
import time
import pytest
import sqlite3
import pandas as pd
from database import (
    init_db,
    insert_telemetry,
    load_telemetry_df,
    log_model_run,
    get_connection
)

TEST_DB_PATH = "test_airaware.db"


def remove_file_safely(filepath):
    gc.collect()
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
        except PermissionError:
            time.sleep(0.1)
            try:
                os.remove(filepath)
            except Exception:
                pass


@pytest.fixture(autouse=True)
def cleanup_test_db():
    remove_file_safely(TEST_DB_PATH)
    yield
    remove_file_safely(TEST_DB_PATH)


def test_init_db_creates_tables():
    init_db(TEST_DB_PATH)
    assert os.path.exists(TEST_DB_PATH)
    
    conn = get_connection(TEST_DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        assert "telemetry" in tables
        assert "model_logs" in tables
    finally:
        conn.close()


def test_insert_and_load_telemetry():
    sample_records = [
        {
            'timestamp': '2026-08-11 12:00:00',
            'location': 'Connaught Place',
            'lat': 28.6315,
            'lon': 77.2167,
            'aqi': 150,
            'pm2_5': 120.5,
            'pm10': 180.0,
            'no2': 40.0,
            'o3': 20.0,
            'so2': 15.0,
            'nh3': 10.0,
            'temp': 29.5,
            'humidity': 60.0,
            'wind': 3.5
        }
    ]
    insert_telemetry(sample_records, TEST_DB_PATH)
    df = load_telemetry_df(TEST_DB_PATH)
    
    assert not df.empty
    assert len(df) == 1
    assert df.iloc[0]['location'] == 'Connaught Place'
    assert float(df.iloc[0]['pm2_5']) == 120.5


def test_log_model_run():
    log_model_run(mae=15.5, rmse=25.2, r2=0.72, sample_count=1000, status="SUCCESS", db_path=TEST_DB_PATH)
    
    conn = get_connection(TEST_DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM model_logs")
        rows = cursor.fetchall()
        assert len(rows) == 1
        assert float(rows[0]['mae']) == 15.5
        assert rows[0]['status'] == "SUCCESS"
    finally:
        conn.close()
