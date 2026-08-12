import os
import requests
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from collections import deque
from dotenv import load_dotenv
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from features import (
    construct_feature_vector,
    min_distance_to_road,
    FEATURE_COLUMNS,
    MAJOR_ROADS_AND_JUNCTIONS
)

# Load environment variables
load_dotenv()

OW_API_KEY = os.environ.get("OW_API_KEY", "")
ORS_API_KEY = os.environ.get("ORS_API_KEY", "")
WAQI_TOKEN = os.environ.get("WAQI_TOKEN", "")

if not all([OW_API_KEY, ORS_API_KEY, WAQI_TOKEN]):
    print("⚠️ WARNING: One or more API keys are missing in the environment. Mock/fallback data mode will be active.")

# ================= STATIC LOCATIONS =================
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

# ================= ML MODEL MANAGER =================
stacked_model = None
features_list = None

def load_models():
    global stacked_model, features_list
    try:
        model_path = os.path.join("models", "stacked_model.pkl")
        feats_path = os.path.join("models", "features_list.pkl")
        if os.path.exists(model_path) and os.path.exists(feats_path):
            stacked_model = joblib.load(model_path)
            features_list = joblib.load(feats_path)
            print("[OK] ML Models loaded successfully.")
        else:
            print("[INFO] Model file not found. Auto-training base ensemble...")
            from train_model import train_pipeline
            train_pipeline()
            stacked_model = joblib.load(model_path)
            features_list = joblib.load(feats_path)
            print("[OK] Auto-training complete. Models loaded.")
    except Exception as e:
        print(f"[ERROR] Model loading/training failed: {e}")
        stacked_model = None
        features_list = FEATURE_COLUMNS

load_models()

# ================= FLASK SETUP =================
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

limiter = Limiter(get_remote_address, app=app, default_limits=["100 per minute"])
app.config['RATELIMIT_STORAGE_URL'] = 'memory://'

current_aqi_data = {}
station_history = {name: deque(maxlen=48) for name in DELHI_LOCATIONS}

# Load historical data for time-series features from SQLite database / CSV
def load_historical_data():
    try:
        from database import load_telemetry_df
        print("[INFO] Loading historical station telemetry for rolling lags...")
        df = load_telemetry_df()
        if not df.empty:
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
            df = df.dropna(subset=['time', 'pm2_5']).sort_values('time')
            
            for name in DELHI_LOCATIONS:
                loc_df = df[df['location'] == name].tail(48)
                for val in loc_df['pm2_5']:
                    station_history[name].append(val)
            print(f"[OK] Station rolling history initialized ({len(df)} telemetry records).")
        else:
            print("[WARNING] Telemetry dataset empty. Using default station lags.")
    except Exception as e:
        print(f"[WARNING] Error initializing historical lags: {e}")

load_historical_data()

# ================= HELPER FUNCTIONS =================
def waqi_to_aqi_category(val):
    if val <= 50: return 1
    if val <= 100: return 2
    if val <= 150: return 3
    if val <= 200: return 4
    return 5

def get_nearest_station(lat, lon):
    from geopy.distance import geodesic
    nearest = min(DELHI_LOCATIONS.items(), key=lambda x: geodesic((lat, lon), x[1]).kilometers)
    return nearest[0]

def get_station_features(station_name):
    history = list(station_history.get(station_name, []))
    if not history:
        return 150.0, 150.0, 150.0, 150.0, 0.0
    
    lag1 = history[-1] if len(history) >= 1 else 150.0
    lag3 = history[-3] if len(history) >= 3 else lag1
    lag24 = history[-24] if len(history) >= 24 else lag1
    roll6 = float(np.mean(history[-6:])) if history else lag1
    roll_std6 = float(np.std(history[-6:])) if len(history) >= 2 else 0.0
    return float(lag1), float(lag3), float(lag24), roll6, roll_std6

def analyze_route(feature):
    if stacked_model is None or features_list is None:
        return 150.0
    coords = feature["geometry"]["coordinates"]
    if not coords:
        return 150.0
        
    center_lon, center_lat = coords[len(coords)//2]
    
    base_temp, base_humidity, base_wind = 25.0, 60.0, 3.0
    if OW_API_KEY:
        try:
            w_res = requests.get(
                f"http://api.openweathermap.org/data/2.5/weather?lat={center_lat}&lon={center_lon}&appid={OW_API_KEY}&units=metric",
                timeout=2
            )
            if w_res.status_code == 200:
                w = w_res.json()
                base_temp = w.get("main", {}).get("temp", 25.0)
                base_humidity = w.get("main", {}).get("humidity", 60.0)
                base_wind = w.get("wind", {}).get("speed", 3.0)
        except Exception:
            pass

    rows = []
    dt_now = datetime.now()
    for lon, lat in coords[::5]:
        station_name = get_nearest_station(lat, lon)
        lag1, lag3, lag24, roll6, roll_std6 = get_station_features(station_name)
        
        # Micro-modifier based on spatial route variation
        micro_modifier = (lat + lon) % 0.1
        dist_road = min_distance_to_road(lat, lon) + (micro_modifier * 0.5)
        
        row_vec = construct_feature_vector(
            lat=lat, lon=lon, temp=base_temp, humidity=base_humidity, wind=base_wind,
            lag1=lag1, lag3=lag3, lag24=lag24, roll6=roll6, roll_std6=roll_std6,
            dt=dt_now, dist_road_override=dist_road
        )
        rows.append(row_vec)
        
    if not rows:
        return 150.0
        
    df = pd.DataFrame(rows, columns=features_list)
    log_out = stacked_model.predict(df)
    ensemble_out = np.expm1(log_out)
    return float(np.mean(ensemble_out))

def generate_fallback_route(start, end, preference="fastest"):
    steps = 15
    lons = np.linspace(start[0], end[0], steps)
    lats = np.linspace(start[1], end[1], steps)
    coords = []
    if preference == "shortest":
        for x, y in zip(lons, lats):
            coords.append([float(x), float(y)])
    else:
        mid_offset_lat = (end[1] - start[1]) * 0.12
        mid_offset_lon = (start[0] - end[0]) * 0.12
        for i in range(steps):
            t = i / float(steps - 1)
            arc = np.sin(t * np.pi)
            coords.append([
                float(lons[i] + mid_offset_lon * arc),
                float(lats[i] + mid_offset_lat * arc)
            ])
    return {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": coords},
        "properties": {}
    }

# ================= ROUTES / VIEWS =================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": stacked_model is not None,
        "features_count": len(features_list) if features_list else 0,
        "stations_monitored": len(DELHI_LOCATIONS)
    })

@app.route("/api/live-aqi", methods=["GET"])
@limiter.limit("30 per minute")
def live_aqi():
    global current_aqi_data
    result = []
    for name, (lat, lon) in DELHI_LOCATIONS.items():
        raw_aqi = 150
        if WAQI_TOKEN:
            try:
                waqi = requests.get(
                    f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={WAQI_TOKEN}",
                    timeout=3
                ).json()
                if waqi.get("status") == "ok":
                    raw_aqi = waqi["data"]["aqi"]
            except Exception:
                pass

        if raw_aqi == 150:
            raw_aqi = 120 + (abs(hash(name)) % 80)
            
        result.append({
            "location": name,
            "lat": lat,
            "lon": lon,
            "aqi": waqi_to_aqi_category(raw_aqi),
            "raw_aqi": raw_aqi
        })
        station_history[name].append(raw_aqi)
        
    current_aqi_data = {r["location"]: r for r in result}
    return jsonify(result)

@app.route("/api/routes", methods=["POST", "OPTIONS"])
@limiter.limit("20 per minute")
def routes():
    if request.method == "OPTIONS":
        return jsonify({"ok": True}), 200
        
    data = request.get_json() or {}
    start = data.get("start")
    end = data.get("end")
    if not start or not end:
        return jsonify({"error": "Start and end coordinates required"}), 400

    fast, short = None, None
    if ORS_API_KEY:
        headers = {"Authorization": ORS_API_KEY, "Content-Type": "application/json"}
        try:
            res_fast = requests.post(
                "https://api.openrouteservice.org/v2/directions/driving-car/geojson",
                json={"coordinates": [start, end], "preference": "fastest"},
                headers=headers, timeout=3
            )
            if res_fast.status_code == 200:
                j = res_fast.json()
                if "features" in j and len(j["features"]) > 0:
                    fast = j["features"][0]
        except Exception:
            pass

        try:
            res_short = requests.post(
                "https://api.openrouteservice.org/v2/directions/driving-car/geojson",
                json={"coordinates": [start, end], "preference": "shortest"},
                headers=headers, timeout=3
            )
            if res_short.status_code == 200:
                j = res_short.json()
                if "features" in j and len(j["features"]) > 0:
                    short = j["features"][0]
        except Exception:
            pass

    if fast is None:
        fast = generate_fallback_route(start, end, preference="fastest")
    if short is None:
        short = generate_fallback_route(start, end, preference="shortest")

    try:
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
        print(f"Error analyzing routes: {e}")
        return jsonify({"type": "FeatureCollection", "features": []})

@app.route("/api/predict-point", methods=["POST", "OPTIONS"])
@limiter.limit("30 per minute")
def predict_point():
    if request.method == "OPTIONS":
        return jsonify({"ok": True}), 200
        
    if stacked_model is None:
        return jsonify({"error": "ML Model unavailable"}), 500
        
    data = request.get_json() or {}
    lat, lon = data.get("lat"), data.get("lon")
    if lat is None or lon is None:
        return jsonify({"error": "Latitude and longitude required"}), 400

    temp, humidity, wind = 25.0, 60.0, 3.0
    if OW_API_KEY:
        try:
            w = requests.get(
                f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OW_API_KEY}&units=metric",
                timeout=2
            ).json()
            temp = w.get("main", {}).get("temp", 25.0)
            humidity = w.get("main", {}).get("humidity", 60.0)
            wind = w.get("wind", {}).get("speed", 3.0)
        except Exception:
            pass

    live_pm25 = 150.0
    if WAQI_TOKEN:
        try:
            waqi = requests.get(f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={WAQI_TOKEN}", timeout=2).json()
            if waqi.get("status") == "ok":
                live_pm25 = waqi["data"]["iaqi"]["pm25"]["v"] if "pm25" in waqi["data"].get("iaqi", {}) else waqi["data"]["aqi"]
        except Exception:
            pass

    station_name = get_nearest_station(lat, lon)
    _, lag3, lag24, roll6, roll_std6 = get_station_features(station_name)
    lag1 = float(live_pm25)
    
    dt_now = datetime.now()
    row_vec = construct_feature_vector(
        lat=lat, lon=lon, temp=temp, humidity=humidity, wind=wind,
        lag1=lag1, lag3=lag3, lag24=lag24, roll6=roll6, roll_std6=roll_std6, dt=dt_now
    )
    
    try:
        df = pd.DataFrame([row_vec], columns=features_list)
        log_out = stacked_model.predict(df)
        pred_pm25 = float(np.expm1(log_out)[0])
    except Exception as e:
        print("Prediction error:", e)
        pred_pm25 = live_pm25

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
    d = request.get_json() or {}
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
    data = request.get_json() or {}
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
            "response": "I'm your AIRAWARE Assistant. I answer questions regarding AQI, micro-climate forecasting, health safety, and route exposure."
        })
    context = f"Current Delhi AQI Breakdown ({datetime.now().strftime('%H:%M')}):\n"
    for loc, info in current_aqi_data.items():
        level = 'Good 🟢' if info['aqi'] <= 2 else 'Moderate 🟡' if info['aqi'] == 3 else 'Poor 🔴'
        context += f"- {loc}: AQI {info['raw_aqi']} ({level})\n"

    if "current aqi" in user_message or "aqi in delhi" in user_message:
        avg = np.mean([d["raw_aqi"] for d in current_aqi_data.values()]) if current_aqi_data else 150
        return jsonify({"response": f"Average AQI in Delhi right now is {int(avg)}.\n{context}"})
    if "cleanest" in user_message or "route" in user_message:
        return jsonify({"response": "Use the 'Run Clean Route Inference' button on the main control panel to calculate ML-driven spatial routes!"})
    if "mask" in user_message or "health" in user_message:
        return jsonify({"response": "If AQI > 150 or if you suffer from respiratory issues, wear an N95 mask outdoors."})

    return jsonify({
        "response": f"Here is the latest AQI matrix:\n{context}\nAsk me anything about air quality, weather, or routing!"
    })

@app.route("/api/simulator", methods=["POST"])
@limiter.limit("30 per minute")
def simulator():
    data = request.get_json() or {}
    routine = data.get("routine", [])
    years = data.get("years", 1)
    changes = data.get("changes", {})

    if not routine:
        return jsonify({"error": "Routine required"}), 400

    total_exposure = 0
    for act in routine:
        loc = act.get('location', 'Connaught Place')
        dur = act.get('duration_hours', 1)
        aqi = current_aqi_data.get(loc, {}).get('raw_aqi', 150)
        total_exposure += aqi * dur
    exposure_per_day = total_exposure / 24

    daily_risk = max(0, (exposure_per_day - 50) / 100)
    lung_aging = daily_risk * 0.2 * 365 * years / 24

    reduction = 0
    if changes.get('mask'): reduction += 0.15
    if changes.get('bike'): reduction += 0.15
    if changes.get('indoor'): reduction += 0.20
    what_if = exposure_per_day * (1 - reduction)

    return jsonify({
        "base_exposure_per_day": round(exposure_per_day, 1),
        "base_lung_aging_years": round(lung_aging, 1),
        "base_risk_reduction_tip": "Switch to N95 masks and clean routes to reduce annual lung stress.",
        "what_if_exposure": round(what_if, 1),
        "what_if_reduction": round(reduction * 100, 1)
    })

@app.route("/api/breathe-score", methods=["POST"])
@limiter.limit("30 per minute")
def breathe_score():
    data = request.get_json() or {}
    exposure = data.get("exposure", 0)
    score = max(0, 100 - (exposure / 2))
    return jsonify({"score": round(score, 1)})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)