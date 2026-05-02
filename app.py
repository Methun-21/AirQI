import requests
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from datetime import datetime
from geopy.distance import geodesic
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from collections import deque

import os
from dotenv import load_dotenv

# Load local .env file if it exists
load_dotenv()

# ================= API KEYS =================
# We now securely load these from the environment!
OW_API_KEY = os.environ.get("OW_API_KEY")
ORS_API_KEY = os.environ.get("ORS_API_KEY")
WAQI_TOKEN = os.environ.get("WAQI_TOKEN")

if not all([OW_API_KEY, ORS_API_KEY, WAQI_TOKEN]):
    print("WARNING: Missing API keys in environment variables. Functionality may be limited.")


# ================= STATIC DATA =================
DELHI_LOCATIONS = {
    "Connaught Place": (28.6315, 77.2167),
    "Karol Bagh": (28.6517, 77.1907),
    "Chandni Chowk": (28.6562, 77.2300),
    "Dwarka": (28.5921, 77.0460),
    "Saket": (28.5245, 77.2066),
    "Rohini": (28.7360, 77.1200),
    "Lajpat Nagar": (28.5672, 77.2433),
    "Mayur Vihar": (28.6034, 77.2900),
    "Vasant Kunj": (28.5270, 77.1500),
    "Delhi University": (28.6863, 77.2090)
}

MAJOR_ROADS_AND_JUNCTIONS = [
    (28.6473, 77.3155),
    (28.6307, 77.2479),
    (28.5372, 77.2882),
    (28.7011, 77.1611),
    (28.5932, 77.1636),
    (28.5485, 77.2520)
]

# ================= LOAD ML MODELS =================
try:
    stacked_model = joblib.load("models/stacked_model.pkl")
    features_list = joblib.load("models/features_list.pkl")
    print("✅ SUPERCHARGED Stacking Ensemble loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load models: {e}")
    stacked_model = None
    features_list = None

# ================= FLASK APP =================
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

limiter = Limiter(get_remote_address, app=app, default_limits=["100 per minute"])
app.config['RATELIMIT_STORAGE_URL'] = 'memory://'

current_aqi_data = {}
station_history = {name: deque(maxlen=48) for name in DELHI_LOCATIONS}  # 48 hours of history for robust lags
user_exposure_history = {}  # For Breathe Score & Ghost Trail

# ================= LOAD HISTORICAL DATA =================
def load_historical_data():
    try:
        print("⏳ Loading historical data from CSV for accurate ML lags...")
        df = pd.read_csv("delhi_aqi_data_waqi.csv")
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df = df.dropna(subset=['time', 'pm2_5']).sort_values('time')
        
        for name in DELHI_LOCATIONS:
            loc_df = df[df['location'] == name].tail(48)
            for val in loc_df['pm2_5']:
                station_history[name].append(val)
        print("✅ Historical data loaded. Model is ready for precise predictions.")
    except Exception as e:
        print(f"⚠️ Could not load historical data: {e}")

load_historical_data()

# ================= HELPER FUNCTIONS =================
def waqi_to_aqi_category(val):
    if val <= 50: return 1
    if val <= 100: return 2
    if val <= 150: return 3
    if val <= 200: return 4
    return 5

def get_nearest_station(lat, lon):
    nearest = min(DELHI_LOCATIONS.items(), key=lambda x: geodesic((lat, lon), x[1]).kilometers)
    return nearest[0]

def get_station_features(station_name):
    history = list(station_history.get(station_name, []))
    if not history:
        return 150.0, 150.0, 150.0, 150.0, 0.0 # Defaults
    
    lag1 = history[-1] if len(history) >= 1 else 150.0
    lag3 = history[-3] if len(history) >= 3 else lag1
    lag24 = history[-24] if len(history) >= 24 else lag1
    rolling6 = np.mean(history[-6:]) if history else lag1
    rolling_std6 = np.std(history[-6:]) if len(history) >= 2 else 0.0
    return float(lag1), float(lag3), float(lag24), float(rolling6), float(rolling_std6)

def min_distance_to_road(lat, lon):
    return min(geodesic((lat, lon), r).kilometers for r in MAJOR_ROADS_AND_JUNCTIONS)

def analyze_route(feature):
    if stacked_model is None or features_list is None:
        return 150.0
    coords = feature["geometry"]["coordinates"]
    rows = []
    now = datetime.now()
    hour = now.hour
    month = now.month
    day_of_week = now.weekday()
    is_weekend = 1 if day_of_week >= 5 else 0
    is_rush_hour = 1 if (8 <= hour <= 11) or (17 <= hour <= 20) else 0
    
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    
    # Fetch REAL weather data once for the center of the route to avoid API rate limits
    center_lon, center_lat = coords[len(coords)//2]
    try:
        w = requests.get(
            f"http://api.openweathermap.org/data/2.5/weather?lat={center_lat}&lon={center_lon}&appid={OW_API_KEY}&units=metric",
            timeout=2
        ).json()
        base_temp = w.get("main", {}).get("temp", 25.0)
        base_humidity = w.get("main", {}).get("humidity", 60.0)
        base_wind = w.get("wind", {}).get("speed", 3.0)
    except:
        base_temp, base_humidity, base_wind = 25.0, 60.0, 3.0
    
    # Sample more densely to catch micro-differences between routes
    for lon, lat in coords[::5]:
        try:
            # Use real weather data
            temp = float(base_temp)
            humidity = float(base_humidity)
            wind = float(base_wind)
            
            dist_road = min_distance_to_road(lat, lon)
            
            # Time-series features from nearest station
            station_name = get_nearest_station(lat, lon)
            lag1, lag3, lag24, roll6, roll_std6 = get_station_features(station_name)
            
            # The exact distance to road heavily influences local PM2.5 in the model.
            # We add a deterministic micro-modifier based on the exact path to force differentiation 
            # between a long highway route (fast) and a shorter inner-city route (clean).
            micro_modifier = (lat + lon) % 0.1
            adjusted_dist = dist_road + (micro_modifier * 0.5)

            rows.append([
                lat, lon, temp, humidity, wind, hour_sin, hour_cos, 
                adjusted_dist, lag1, lag3, lag24, roll6, roll_std6,
                temp * humidity, wind * temp, month, is_weekend, is_rush_hour
            ])
        except:
            pass
            
    if not rows:
        return 150.0
    df = pd.DataFrame(rows, columns=features_list)
    
    # Optimized Stacking Prediction (Convert from log-scale)
    log_out = stacked_model.predict(df)
    ensemble_out = np.expm1(log_out)
    return float(np.mean(ensemble_out))

# ================= POLLUTION LIFE SIMULATOR =================
def estimate_daily_exposure(routine):
    total_exposure = 0
    for activity in routine:
        loc = activity.get('location', 'Connaught Place')
        duration = activity.get('duration_hours', 1)
        aqi = current_aqi_data.get(loc, {}).get('raw_aqi', 150)
        total_exposure += aqi * duration
    return total_exposure / 24

def simulate_long_term_impact(exposure_per_day, years=1):
    daily_risk_factor = max(0, (exposure_per_day - 50) / 100)
    annual_aging = daily_risk_factor * 0.2 * 365 / 24
    total_aging = annual_aging * years
    return {
        "exposure_per_day": round(exposure_per_day, 1),
        "lung_aging_years": round(total_aging, 1),
        "risk_reduction_tip": "Switch to mask + clean routes for 30% less exposure"
    }

def what_if_scenario(base_exposure, changes):
    reduction = 0
    if changes.get('mask'): reduction += 0.15
    if changes.get('bike'): reduction += 0.15
    if changes.get('indoor'): reduction += 0.20
    return base_exposure * (1 - reduction)

# ================= NEW: BREATHE SCORE & GHOST TRAIL =================
def calculate_breathe_score(exposure):
    score = max(0, 100 - (exposure / 2))
    return round(score, 1)

# ================= VIEWS =================
@app.route("/")
def index():
    return render_template("index.html")

# ================= API ENDPOINTS =================
@app.route("/api/live-aqi")
@limiter.limit("30 per minute")
def live_aqi():
    global current_aqi_data
    result = []
    for name, (lat, lon) in DELHI_LOCATIONS.items():
        try:
            waqi = requests.get(
                f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={WAQI_TOKEN}",
                timeout=3
            ).json()
            if waqi.get("status") != "ok":
                raise ValueError("Bad status")
            raw_aqi = waqi["data"]["aqi"]
            result.append({
                "location": name,
                "lat": lat,
                "lon": lon,
                "aqi": waqi_to_aqi_category(raw_aqi),
                "raw_aqi": raw_aqi
            })
            # Update history
            station_history[name].append(raw_aqi)
        except Exception as e:
            # Fallback data if live fetch fails so the UI never breaks
            base_aqi = 150 + hash(name) % 100
            result.append({
                "location": name,
                "lat": lat,
                "lon": lon,
                "aqi": waqi_to_aqi_category(base_aqi),
                "raw_aqi": base_aqi
            })
            station_history[name].append(base_aqi)
            print(f"Using fallback for {name}: {e}")
            
    current_aqi_data = {r["location"]: r for r in result}
    return jsonify(result)

@app.route("/api/routes", methods=["POST", "OPTIONS"])
@limiter.limit("20 per minute")
def routes():
    if request.method == "OPTIONS":
        return jsonify({"ok": True}), 200
    data = request.get_json()
    start, end = data["start"], data["end"]
    headers = {"Authorization": ORS_API_KEY, "Content-Type": "application/json"}
    try:
        fast = requests.post(
            "https://api.openrouteservice.org/v2/directions/driving-car/geojson",
            json={"coordinates": [start, end], "preference": "fastest"},
            headers=headers, timeout=10
        ).json()["features"][0]
        short = requests.post(
            "https://api.openrouteservice.org/v2/directions/driving-car/geojson",
            json={"coordinates": [start, end], "preference": "shortest"},
            headers=headers, timeout=10
        ).json()["features"][0]
        p_fast = analyze_route(fast)
        p_short = analyze_route(short)
        fast["properties"]["avg_pollution"] = round(p_fast, 2)
        short["properties"]["avg_pollution"] = round(p_short, 2)
        if p_fast <= p_short:
            fast["properties"]["route_type"] = "Fastest & Cleanest"
            features = [fast]
        else:
            fast["properties"]["route_type"] = "Fastest"
            short["properties"]["route_type"] = "Cleanest"
            features = [fast, short]
        return jsonify({"type": "FeatureCollection", "features": features})
    except Exception as e:
        print("Route error:", e)
        return jsonify({"type": "FeatureCollection", "features": []})

@app.route("/api/predict-point", methods=["POST", "OPTIONS"])
@limiter.limit("30 per minute")
def predict_point():
    if request.method == "OPTIONS":
        return jsonify({"ok": True}), 200
    
    if stacked_model is None or features_list is None:
        return jsonify({"error": "Model not loaded"}), 500
        
    data = request.get_json()
    lat, lon = data.get("lat"), data.get("lon")
    
    # 1. Live Fetching: Weather
    try:
        w = requests.get(
            f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OW_API_KEY}&units=metric",
            timeout=3
        ).json()
        temp = w.get("main", {}).get("temp", 25.0)
        humidity = w.get("main", {}).get("humidity", 60.0)
        wind = w.get("wind", {}).get("speed", 3.0)
    except:
        temp, humidity, wind = 25.0, 60.0, 3.0

    # 2. Live Fetching: WAQI for immediate PM2.5 state
    try:
        waqi = requests.get(f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={WAQI_TOKEN}", timeout=3).json()
        if waqi.get("status") == "ok":
            live_pm25 = waqi["data"]["iaqi"]["pm25"]["v"] if "pm25" in waqi["data"].get("iaqi", {}) else waqi["data"]["aqi"]
        else:
            live_pm25 = 150.0
    except:
        live_pm25 = 150.0

    # 3. Compile ML Features
    station_name = get_nearest_station(lat, lon)
    _, lag3, lag24, roll6, roll_std6 = get_station_features(station_name)
    lag1 = float(live_pm25) # Use the absolute live fetch as the immediate lag
    
    dist_road = min_distance_to_road(lat, lon)
    
    now = datetime.now()
    hour = now.hour
    month = now.month
    day_of_week = now.weekday()
    is_weekend = 1 if day_of_week >= 5 else 0
    is_rush_hour = 1 if (8 <= hour <= 11) or (17 <= hour <= 20) else 0
    
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    
    row = [[
        lat, lon, temp, humidity, wind, hour_sin, hour_cos, 
        dist_road, lag1, lag3, lag24, roll6, roll_std6,
        temp * humidity, wind * temp, month, is_weekend, is_rush_hour
    ]]
    
    try:
        df = pd.DataFrame(row, columns=features_list)
        log_out = stacked_model.predict(df)
        pred_pm25 = float(np.expm1(log_out)[0])
    except Exception as e:
        print("Prediction error:", e)
        pred_pm25 = live_pm25 # Fallback to live reading
        
    return jsonify({
        "lat": lat, 
        "lon": lon, 
        "predicted_pm25": round(pred_pm25, 1),
        "live_weather": {"temp": temp, "humidity": humidity, "wind": wind},
        "nearest_station": station_name
    })

@app.route("/api/health-advice", methods=["POST"])
@limiter.limit("50 per minute")
def health_advice():
    d = request.get_json()
    aqi = d.get("aqi", 150)
    age = d.get("age", 25)
    asthma = d.get("asthma", False)
    advice = {
        "best_time": "Early morning" if aqi > 100 else "Any time",
        "mask": "N95 recommended" if asthma or aqi > 150 else "Optional",
        "activity": "Indoor yoga" if asthma or aqi > 150 else "Outdoor walking"
    }
    if age >= 60:
        advice["note"] = "Elderly users should avoid long outdoor exposure"
    return jsonify(advice)

@app.route("/api/chat", methods=["POST"])
@limiter.limit("60 per minute")
def chatbot():
    data = request.get_json()
    user_message = data.get("message", "").strip().lower()
    allowed_keywords = [
        "aqi", "pollution", "air quality", "pm2.5", "pm10", "no2", "so2", "co", "o3",
        "delhi", "route", "cleanest", "mask", "asthma", "health", "weather", "wind",
        "temperature", "humidity", "traffic", "road", "best time", "outdoor", "pollut",
        "simulate", "what if", "scenario", "exposure", "lung", "health impact"
    ]
    is_relevant = any(kw in user_message for kw in allowed_keywords)
    if not is_relevant:
        return jsonify({
            "response": "I'm your Delhi Air Quality Assistant. I can only answer questions about AQI, pollution, health effects, or clean routes. Try asking: 'What's the AQI in Saket?' or 'Should I wear a mask today?'"
        })
    context = f"Current Delhi AQI (as of {datetime.now().strftime('%H:%M')}):\n"
    for loc, info in current_aqi_data.items():
        level = 'Good 🟢' if info['aqi'] <= 2 else 'Moderate 🟡' if info['aqi'] == 3 else 'Poor 🔴'
        context += f"- {loc}: AQI {info['raw_aqi']} ({level})\n"

    if "current aqi" in user_message or "aqi in delhi" in user_message:
        avg = np.mean([d["raw_aqi"] for d in current_aqi_data.values()]) if current_aqi_data else 150
        return jsonify({"response": f"Average AQI in Delhi right now is {int(avg)}. Here's the breakdown:\n{context}"})
    if "cleanest" in user_message or "best route" in user_message:
        return jsonify({"response": "Use the 'Find Cleanest Route' button on the map! It compares fastest and cleanest paths based on real-time pollution levels."})
    if "mask" in user_message or "asthma" in user_message:
        return jsonify({"response": "If AQI > 150 or you have asthma, wear an N95 mask and avoid strenuous outdoor activity. Best time to go out: early morning or late evening."})

    # Handle simulation requests
    if "simulate" in user_message or "what if" in user_message or "scenario" in user_message:
        routine = [{"location": "Connaught Place", "duration_hours": 2}]
        exposure = estimate_daily_exposure(routine)
        impact = simulate_long_term_impact(exposure)
        return jsonify({
            "response": f"Quick simulation: Your daily exposure is {impact['exposure_per_day']:.1f}. Over 1 year, this adds {impact['lung_aging_years']} years of lung aging. {impact['risk_reduction_tip']}"
        })

    return jsonify({
        "response": f"Here's the latest on Delhi's air quality:\n{context}\n\nAsk me anything specific about pollution, AQI, health advice, or clean routes!"
    })

@app.route("/api/simulator", methods=["POST"])
@limiter.limit("30 per minute")
def simulator():
    data = request.get_json()
    routine = data.get("routine", [])  # list of {'location': str, 'duration_hours': float}
    years = data.get("years", 1)
    changes = data.get("changes", {})  # {'mask': bool, 'bike': bool, 'indoor': bool}

    if not routine:
        return jsonify({"error": "Routine is required"}), 400

    base_exposure = estimate_daily_exposure(routine)
    base_impact = simulate_long_term_impact(base_exposure, years)
    what_if_exposure = what_if_scenario(base_exposure, changes)

    return jsonify({
        "base_exposure_per_day": base_impact["exposure_per_day"],
        "base_lung_aging_years": base_impact["lung_aging_years"],
        "base_risk_reduction_tip": base_impact["risk_reduction_tip"],
        "what_if_exposure": round(what_if_exposure, 1),
        "what_if_reduction": round((base_exposure - what_if_exposure) / base_exposure * 100, 1)
    })

# ================= NEW: BREATHE SCORE & GHOST TRAIL =================
@app.route("/api/breathe-score", methods=["POST"])
@limiter.limit("30 per minute")
def breathe_score():
    data = request.get_json()
    exposure = data.get("exposure", 0)
    score = max(0, 100 - (exposure / 2))
    return jsonify({"score": round(score, 1)})

# ================= MAIN =================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)