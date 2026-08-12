# AIRAWARE: Predictive Micro-Zoning Engine 🌍💨

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Flask Framework](https://img.shields.io/badge/Framework-Flask-000000?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Machine Learning](https://img.shields.io/badge/ML-Stacking%20Ensemble-10B981?style=flat&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Testing](https://img.shields.io/badge/Tests-Pytest-0A9EDC?style=flat&logo=pytest&logoColor=white)](https://pytest.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**AIRAWARE** is an end-to-end, AI-driven micro-climate forecasting and dynamic traffic management platform built for atmospheric urban pollution control in Delhi. Moving beyond reactive city-wide averages, AIRAWARE uses a multi-stage **Stacking Machine Learning Ensemble** to predict hyper-local $PM_{2.5}$ formation hours in advance.

---

## 🎯 Key Capabilities

- **Hyper-Local $PM_{2.5}$ Micro-Zoning:** Real-time spatial inference based on geodesic distance to major traffic corridors, 48-hour rolling atmospheric lags, and cyclical temporal encodings.
- **Health-Optimized Clean Routing API:** Computes and compares spatial routes between origins and destinations, recommending paths that minimize cumulative $PM_{2.5}$ exposure.
- **Personal Exposure & Lung Impact Simulator:** Quantitative actuarial exposure model projecting annual pulmonary strain based on daily outdoor routines.
- **Dual-Mode System Interface:** Seamless toggle between **Citizen View** (routing & health tips) and **Government/Admin Mode** (hotspot identification for anti-smog deployment).
- **Automated MLOps Pipeline:** Scheduled CI/CD via GitHub Actions for continuous data ingestion, automated retraining, and model evaluation.

---

## 🧠 Machine Learning Architecture

AIRAWARE employs a two-tier **Stacking Ensemble Regressor** designed to model nonlinear spatiotemporal atmospheric volatility.

```
       [ Input Features (18 Spatiotemporal & Meteorological Features) ]
                                       │
         ┌─────────────────────────────┼─────────────────────────────┐
         ▼                             ▼                             ▼
  ┌──────────────┐              ┌──────────────┐              ┌──────────────┐
  ┌───┴──────────┴───┐          ┌───┴──────────┴───┐          ┌───┴──────────┴───┐
  │  Random Forest   │          │     XGBoost      │          │    LightGBM      │
  │ (100 Trees, d=20)│          │ (150 Trees, lr=.05)│        │ (150 Trees, lr=.05)│
  └──────────────────┘          └──────────────────┘          └──────────────────┘
         │                             │                             │
         └─────────────────────────────┼─────────────────────────────┘
                                       ▼
                            ┌─────────────────────┐
                            │  CatBoost Regressor │
                            │ (250 Iterations)    │
                            └──────────┬──────────┘
                                       │
                                       ▼  (Out-of-Fold Predictions)
                            ┌─────────────────────┐
                            │   Ridge Regressor   │  <-- Tier-2 Meta Learner
                            └──────────┬──────────┘
                                       │
                                       ▼
                       [ Predicted Hyper-Local PM2.5 ]
```

### Feature Engineering Pipeline (`features.py`)
- **Geospatial Proximity:** Geodesic distance to 8 major arterial transport corridors in Delhi (`distance_to_major_road`).
- **Cyclical Time Encoding:** Sine/Cosine transform ($\sin(2\pi \cdot \text{hour}/24)$, $\cos(2\pi \cdot \text{hour}/24)$) to model diurnal atmospheric inversion.
- **Spatiotemporal Lags:** 1h, 3h, 24h historical memory lags with 6-hour rolling mean and standard deviation volatility metrics.
- **Thermodynamic Interactions:** Cross-features ($Temp \times Humidity$, $Wind \times Temp$).

### Benchmark Model Evaluation
- **MAE (Mean Absolute Error):** `17.54 µg/m³`
- **RMSE (Root Mean Squared Error):** `29.34 µg/m³`
- **R² Score:** `0.696`

---

## 🛠️ Tech Stack

- **Backend:** Python 3.12, Flask, Geopy, Pandas, NumPy, Scikit-Learn
- **Machine Learning:** XGBoost, LightGBM, CatBoost, Joblib
- **Frontend:** HTML5, Modern Vanilla CSS (Glassmorphism), JavaScript (ES6+), Leaflet.js, Chart.js
- **Testing & Quality:** Pytest, Pytest-Flask
- **DevOps / MLOps:** GitHub Actions (CI/CT Workflows)
- **External Data Providers:** World Air Quality Index (WAQI), OpenWeatherMap, OpenRouteService

---

## 🏗️ Project Structure

```
AIRAWARE/
├── .github/workflows/         # Continuous Integration & Training Workflows
│   ├── main.yml               # Hourly data collection pipeline
│   └── retrain_model.yml      # Scheduled weekly MLOps model retraining
├── models/                    # Serialized Machine Learning artifacts (.pkl)
│   ├── stacked_model.pkl      # Stacking Ensemble meta-model
│   └── features_list.pkl      # Standardized feature schema
├── static/                    # Frontend assets
│   ├── css/style.css          # Glassmorphic UI Design System
│   └── js/main.js             # Map rendering, chart telemetry & API logic
├── templates/
│   └── index.html             # Single-Page Web Application layout
├── tests/                     # Automated Test Suite
│   ├── test_api.py            # Flask API endpoint integration tests
│   ├── test_features.py       # Feature engineering unit tests
│   └── test_ml.py             # ML inference & pipeline unit tests
├── .env.example               # Template environment configuration
├── .gitignore                 # Standard repository ignores
├── app.py                     # Flask Web Server & REST API Gateway
├── collect_data.py            # Automated sensor telemetry collection
├── evaluate_accuracy.py       # Model accuracy benchmark script
├── features.py                # Centralized Feature Engineering Engine
├── train_model.py             # Model training & ensemble optimization
├── requirements.txt           # Project dependencies
└── LICENSE                    # MIT Open Source License
```

---

## ⚙️ Quickstart & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Methun-21/AirQI.git
cd AirQI
```

### 2. Set Up Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / MacOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env` and populate your API credentials (the app includes fallback modes if keys are omitted):
```bash
cp .env.example .env
```

### 5. Run Automated Tests
```bash
python -m pytest tests/ -v
```

### 6. Start the Flask Server
```bash
python app.py
```
Open your browser and navigate to `http://127.0.0.1:5001/`

---

## 🔌 API Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `GET /` | `GET` | Renders main application Web UI |
| `GET /api/health` | `GET` | System health check and model loading status |
| `GET /api/live-aqi` | `GET` | Returns live sensor matrix and AQI readings |
| `POST /api/routes` | `POST` | Calculates fastest vs cleanest spatial route |
| `POST /api/predict-point` | `POST` | Click-to-predict ML inference for latitude/longitude |
| `POST /api/health-advice` | `POST` | Generates tailored medical advice based on user profile |
| `POST /api/simulator` | `POST` | Projections for annual lung exposure & mitigation impact |
| `POST /api/chat` | `POST` | AirBot natural language assistant queries |

---

## 📄 License & Attribution

Distributed under the **MIT License**. Developed by **Methunraj A.**
