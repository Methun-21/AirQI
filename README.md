# AIRAWARE: Probabilistic Spatiotemporal PM2.5 Estimation & Eco-Routing Engine 🌍💨

[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Flask Framework](https://img.shields.io/badge/Framework-Flask-000000?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Machine Learning](https://img.shields.io/badge/ML-LightGBM%20%7C%20Quantile%20Regression-10B981?style=flat&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Spatial CV](https://img.shields.io/badge/Validation-Leave--One--Station--Out%20(LOSO)-orange)](https://scikit-learn.org)
[![Tests](https://img.shields.io/badge/Tests-16%20Passed%20(Pytest)-0A9EDC?style=flat&logo=pytest&logoColor=white)](https://pytest.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**AIRAWARE** is a high-rigor, probabilistic spatiotemporal $PM_{2.5}$ estimation and mobility-aware eco-routing engine designed for megacity atmospheric pollution control in Delhi. Moving beyond naive city-wide averages and arbitrary heuristics, AIRAWARE uses **Quantile Gradient Boosted Trees ($P_{10}, P_{50}, P_{90}$)** and **Leave-One-Station-Out (LOSO) Spatial Cross-Validation** to deliver reliable, uncertainty-calibrated micro-climate inference and risk-aware navigation.

---

## 🎯 Core Technical Highlights

- **Spatial Generalization via LOSO-CV:** Validated using 10-station Leave-One-Station-Out Spatial Cross-Validation ($\text{MAE} = 17.73\,\mu\text{g/m}^3$, $R^2 = 0.929$), proving true spatial interpolation across held-out geographic nodes.
- **Uncertainty Quantification ($P_{10}\text{–}P_{90}$):** Employs pinball quantile regression to generate calibrated 80% prediction intervals ($72.4\%$ empirical test coverage), communicating forecast confidence to commuters.
- **Risk-Aware Eco-Routing:** Discretizes candidate routes and evaluates point-wise quantile pollution doses, distinguishing between direct high-risk arterial corridors and low-exposure bypass paths.
- **Rigorous Multi-Baseline Benchmarking:** Evaluated against Naive Persistence ($\hat{y}_t = y_{t-1}$), Station Climatology, and Stacking Regressors across out-of-time splits and smog-spike regimes ($PM_{2.5} > 150\,\mu\text{g/m}^3$).
- **Automated MLOps Pipeline:** Scheduled CI/CT workflows via GitHub Actions for continuous sensor telemetry ingestion, data validation, and model retraining.

---

## 📊 Benchmark & Ablation Results

Evaluated on **28,529 real-world telemetry records** from Delhi ground stations:

### 1. Out-of-Time Temporal Split Benchmark (Unseen Chronological Test Set)
| Model / Architecture | MAE ($\mu\text{g/m}^3$) | RMSE ($\mu\text{g/m}^3$) | $R^2$ Score | Key Takeaway |
| :--- | :---: | :---: | :---: | :--- |
| **Naive Persistence ($\hat{y}_t = y_{t-1}$)** | 20.64 | 32.40 | 0.408 | Strong naive baseline in hourly air quality |
| **Seasonal-Naive Climatology** | 99.12 | 110.35 | -5.870 | Fails during rapid atmospheric inversions |
| **Stacking Ensemble (RF+XGB+LGBM+CatBoost)** | 19.29 | 26.30 | 0.610 | Heavy computational cost, marginal benefit |
| **LightGBM Regressor (Primary)** | **19.14** | **26.05** | **0.617** | **Best overall accuracy and fastest inference** |

### 2. Leave-One-Station-Out (LOSO) Spatial Generalization
| Metric | Value |
| :--- | :---: |
| **Overall Spatial LOSO MAE (Held-Out Stations)** | **17.73 $\mu\text{g/m}^3$** |
| **Overall Spatial LOSO $R^2$** | **0.929** |
| **80% Prediction Interval Coverage Probability (PICP)** | **72.36%** |
| **Mean Prediction Interval Width (MPIW)** | **54.59 $\mu\text{g/m}^3$** |

---

## 🧠 Feature Engineering Engine (`features.py`)

1. **Geospatial Proximity:** Geodesic distance to arterial transport hubs and heavy congestion flyovers in Delhi (`distance_to_major_road`).
2. **Cyclical Temporal Encodings:** Sinusoidal transformations for diurnal ($\sin(2\pi h / 24)$, $\cos(2\pi h / 24)$) and seasonal annual cycles ($\sin(2\pi m / 12)$, $\cos(2\pi m / 12)$).
3. **Autoregressive Memory Lags:** 1-hour, 3-hour, and 24-hour historical lags with 6-hour rolling mean and standard deviation volatility metrics.
4. **Thermodynamic Cross-Interactions:** Aerosol hygroscopic growth proxy ($\text{Temp} \times \text{Humidity}$) and boundary-layer ventilation proxy ($\text{Wind} \times \text{Temp}$).

---

## 🛠️ Tech Stack & Directory Structure

- **Backend:** Python 3.12+, Flask, Geopy, Pandas, NumPy, Scikit-Learn, LightGBM, XGBoost, CatBoost
- **Data Persistence:** SQLite (`airaware.db` with indexed `telemetry` and `model_logs`)
- **Frontend:** Leaflet.js, Chart.js, Vanilla Glassmorphism CSS, ES6 JavaScript
- **Testing & Quality:** Pytest (16 automated tests)
- **CI/CD:** GitHub Actions scheduled workflows

```
AIRAWARE/
├── .github/workflows/         # Scheduled CI Telemetry Ingestion & Retraining
├── models/                    # Serialized LightGBM, Quantile, and Metadata artifacts
├── static/                    # Glassmorphism UI & Leaflet Map scripts
├── templates/                 # Single-Page Web UI (index.html)
├── tests/                     # Pytest suite (API, Database, Features, ML)
├── app.py                     # Flask API Gateway & Routing Logic
├── collect_data.py            # Automated Sensor Telemetry Ingest
├── database.py                # SQLite ORM & Schema Engine
├── evaluate_accuracy.py       # Diagnostic Plotter & Evaluation Suite
├── features.py                # Centralized Feature Engineering Engine
├── train_model.py             # Rigorous Training & LOSO-CV Pipeline
└── requirements.txt           # Project Dependencies
```

---

## ⚙️ Quickstart

```bash
# 1. Clone the repository
git clone https://github.com/Methun-21/AirQI.git
cd AirQI

# 2. Set up Virtual Environment & Install Dependencies
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 3. Run Automated Tests
python -m pytest tests/ -v

# 4. Train Model & Generate Benchmarks
python train_model.py

# 5. Start Flask Server
python app.py
```
Navigate to `http://127.0.0.1:5001/` in your browser.

---

## 🔌 API Reference

| Endpoint | Method | Description | Output Sample |
| :--- | :---: | :--- | :--- |
| `GET /api/health` | `GET` | Health check & model metadata | `{"primary_model": "LightGBM", "status": "healthy"}` |
| `GET /api/live-aqi` | `GET` | Real-time station sensor matrix | `[{"location": "Connaught Place", "pm2_5": 84.2}]` |
| `POST /api/predict-point` | `POST` | Spatial click-to-predict with 80% CI | `{"pm25_median": 124.5, "ci_lower": 98.2, "ci_upper": 156.0}` |
| `POST /api/routes` | `POST` | Risk-aware eco-route comparison | `{"type": "FeatureCollection", "features": [...]}` |

---

## 📄 License & Attribution
Distributed under the **MIT License**. Developed by **Methunraj A.**
