# collect_data_waqi.py (GitHub Version - uses environment variables for keys)
import os
import requests
import csv
from datetime import datetime

# --- CONFIGURATION ---
# This version reads keys securely from environment variables (GitHub Secrets)
OW_API_KEY = os.environ.get("OW_API_KEY")
WAQI_TOKEN = os.environ.get("WAQI_TOKEN")

if not all([OW_API_KEY, WAQI_TOKEN]):
    raise ValueError("One or more API key environment variables are not set.")

DELHI_LOCATIONS = {
    "Connaught Place": (28.6315, 77.2167), "Karol Bagh": (28.6517, 77.1907),
    "Chandni Chowk": (28.6562, 77.2300), "Dwarka": (28.5921, 77.0460),
    "Saket": (28.5245, 77.2066), "Rohini": (28.7360, 77.1200),
    "Lajpat Nagar": (28.5672, 77.2433), "Mayur Vihar": (28.6034, 77.2900),
    "Vasant Kunj": (28.5270, 77.1500), "Delhi University": (28.6863, 77.2090),
}
FILENAME = "delhi_aqi_data_waqi.csv"

# --- Main Script ---
print("Starting data collection with high-accuracy WAQI data...")
for loc, (lat, lon) in DELHI_LOCATIONS.items():
    try:
        # Step 1: Call WAQI API for accurate pollution data
        waqi_url = f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={WAQI_TOKEN}"
        waqi_res = requests.get(waqi_url)
        waqi_res.raise_for_status()
        waqi_data = waqi_res.json()

        if waqi_data.get('status') != 'ok':
            print(f"⚠️ WAQI API returned an error for {loc}: {waqi_data.get('data')}")
            continue

        aqi = waqi_data['data'].get('aqi', 0)
        iaqi = waqi_data['data'].get('iaqi', {})
        pm25 = iaqi.get('pm25', {}).get('v', 0)
        pm10 = iaqi.get('pm10', {}).get('v', 0)
        no2 = iaqi.get('no2', {}).get('v', 0)
        o3 = iaqi.get('o3', {}).get('v', 0)
        so2 = iaqi.get('so2', {}).get('v', 0)
        nh3 = iaqi.get('nh3', {}).get('v', 0)

        # Step 2: Call OpenWeather API for weather data
        weather_url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OW_API_KEY}&units=metric"
        weather_res = requests.get(weather_url)
        weather_res.raise_for_status()
        weather = weather_res.json()
        
        temp = weather.get("main", {}).get("temp")
        humidity = weather.get("main", {}).get("humidity")
        wind = weather.get("wind", {}).get("speed")

        # Step 3: Write the combined data to the new CSV
        with open(FILENAME, "a", newline="", encoding='utf-8') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow([
                    "time", "location", "lat", "lon", "aqi", "pm2_5", "pm10", 
                    "no2", "o3", "so2", "nh3", "temp", "humidity", "wind"
                ])
            
            writer.writerow([
                datetime.now(), loc, lat, lon, aqi, pm25, pm10,
                no2, o3, so2, nh3, temp, humidity, wind
            ])
        print(f"✅ Data saved for {loc} (AQI: {aqi}, PM2.5: {pm25})")

    except Exception as e:
        print(f"❌ Failed to fetch data for {loc}: {e}")

print(f"Data collection finished. Dataset updated: {FILENAME}")
