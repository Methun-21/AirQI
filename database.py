"""
AIRAWARE Database Module
Handles SQLite storage (airaware.db) for sensor telemetry, historical lags, and MLOps model training logs.
"""

import os
import sqlite3
import pandas as pd
from datetime import datetime

DEFAULT_DB_PATH = "airaware.db"


def get_connection(db_path=DEFAULT_DB_PATH):
    """Establishes and returns a connection to the SQLite database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path=DEFAULT_DB_PATH):
    """Initializes the database schema if tables do not exist."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        
        # Telemetry table storing WAQI + OpenWeather telemetry
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS telemetry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                location TEXT NOT NULL,
                lat REAL NOT NULL,
                lon REAL NOT NULL,
                aqi INTEGER,
                pm2_5 REAL NOT NULL,
                pm10 REAL,
                no2 REAL,
                o3 REAL,
                so2 REAL,
                nh3 REAL,
                temp REAL,
                humidity REAL,
                wind REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Model training execution logs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trained_at DATETIME NOT NULL,
                mae REAL NOT NULL,
                rmse REAL,
                r2 REAL,
                sample_count INTEGER,
                status TEXT DEFAULT 'SUCCESS'
            )
        """)
        
        # Create index on location and timestamp for fast lag querying
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_loc_time ON telemetry(location, timestamp)")
        conn.commit()
    finally:
        conn.close()


def insert_telemetry(records, db_path=DEFAULT_DB_PATH):
    """
    Inserts a list of dictionary records into the telemetry table.
    """
    init_db(db_path)
    query = """
        INSERT INTO telemetry 
        (timestamp, location, lat, lon, aqi, pm2_5, pm10, no2, o3, so2, nh3, temp, humidity, wind)
        VALUES (:timestamp, :location, :lat, :lon, :aqi, :pm2_5, :pm10, :no2, :o3, :so2, :nh3, :temp, :humidity, :wind)
    """
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.executemany(query, records)
        conn.commit()
    finally:
        conn.close()


def load_telemetry_df(db_path=DEFAULT_DB_PATH, csv_fallback="delhi_aqi_data_waqi.csv"):
    """
    Loads all telemetry data from SQLite into a pandas DataFrame.
    If database is empty or missing, migrates CSV data first.
    """
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM telemetry ORDER BY timestamp ASC", conn)
    finally:
        conn.close()
        
    if df.empty:
        if os.path.exists(csv_fallback):
            print(f"[INFO] Database empty. Migrating CSV '{csv_fallback}' to SQLite...")
            migrate_csv_to_db(csv_fallback, db_path)
            conn = get_connection(db_path)
            try:
                df = pd.read_sql_query("SELECT * FROM telemetry ORDER BY timestamp ASC", conn)
            finally:
                conn.close()
        else:
            print("[WARNING] Database and CSV fallback are both empty.")
            return pd.DataFrame()
            
    if 'time' not in df.columns and 'timestamp' in df.columns:
        df['time'] = df['timestamp']
        
    return df


def log_model_run(mae, rmse=None, r2=None, sample_count=None, status="SUCCESS", db_path=DEFAULT_DB_PATH):
    """Logs model training run results into the model_logs table."""
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO model_logs (trained_at, mae, rmse, r2, sample_count, status)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (datetime.now().isoformat(), mae, rmse, r2, sample_count, status))
        conn.commit()
    finally:
        conn.close()


def migrate_csv_to_db(csv_path="delhi_aqi_data_waqi.csv", db_path=DEFAULT_DB_PATH):
    """Migrates historical CSV data into SQLite database."""
    if not os.path.exists(csv_path):
        print(f"[WARNING] CSV path {csv_path} does not exist. Skipping migration.")
        return 0
        
    init_db(db_path)
    df = pd.read_csv(csv_path)
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df = df.dropna(subset=['time', 'pm2_5'])
    
    records = []
    for _, row in df.iterrows():
        records.append({
            'timestamp': str(row['time']),
            'location': str(row['location']),
            'lat': float(row['lat']),
            'lon': float(row['lon']),
            'aqi': int(row['aqi']) if pd.notnull(row.get('aqi')) else 0,
            'pm2_5': float(row['pm2_5']),
            'pm10': float(row['pm10']) if pd.notnull(row.get('pm10')) else None,
            'no2': float(row['no2']) if pd.notnull(row.get('no2')) else None,
            'o3': float(row['o3']) if pd.notnull(row.get('o3')) else None,
            'so2': float(row['so2']) if pd.notnull(row.get('so2')) else None,
            'nh3': float(row['nh3']) if pd.notnull(row.get('nh3')) else None,
            'temp': float(row['temp']) if pd.notnull(row.get('temp')) else 25.0,
            'humidity': float(row['humidity']) if pd.notnull(row.get('humidity')) else 60.0,
            'wind': float(row['wind']) if pd.notnull(row.get('wind')) else 3.0
        })
        
    insert_telemetry(records, db_path)
    print(f"[OK] Migrated {len(records)} records from {csv_path} to {db_path}")
    return len(records)


if __name__ == "__main__":
    init_db()
    count = migrate_csv_to_db()
    print(f"Database initialization and migration complete. Total migrated: {count}")
