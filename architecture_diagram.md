# AIRAWARE System Architecture

This architecture outlines the end-to-end data pipeline, demonstrating how raw environmental data and geographical coordinates are transformed into actionable health metrics by the machine learning backend.

```mermaid
graph TD
    %% Define Styles
    classDef user fill:#38bdf8,stroke:#0284c7,stroke-width:2px,color:#fff;
    classDef frontend fill:#1e293b,stroke:#334155,stroke-width:2px,color:#fff;
    classDef api fill:#f59e0b,stroke:#d97706,stroke-width:2px,color:#fff;
    classDef backend fill:#0f172a,stroke:#475569,stroke-width:2px,color:#fff;
    classDef ml fill:#10b981,stroke:#059669,stroke-width:2px,color:#fff;
    classDef database fill:#6366f1,stroke:#4f46e5,stroke-width:2px,color:#fff;

    %% 1. User Interaction Layer
    subgraph UI["Frontend Layer (Glassmorphism Web UI)"]
        User(("👤 Web Client")):::user
        Map["🗺️ Leaflet.js Interactive Map"]:::frontend
        Chat["💬 AirBot Assistant"]:::frontend
        Sim["🫁 Exposure Simulator"]:::frontend
        
        User -- Inputs Start/End --> Map
        User -- Asks Questions --> Chat
        User -- Sets Routine --> Sim
    end

    %% 2. External APIs (Data Ingestion)
    subgraph External["External Data Sources (APIs)"]
        WAQI["☁️ WAQI Sensor API<br>(Live PM2.5)"]:::api
        Weather["🌦️ OpenWeatherMap API<br>(Temp, Humidity, Wind)"]:::api
        ORS["🛣️ OpenRouteService API<br>(Path Coordinates)"]:::api
    end

    %% 3. Backend Logic Layer (Python Flask)
    subgraph Backend["Backend Layer (Flask Server)"]
        Router["🔀 API Gateway (app.py)"]:::backend
        Geospatial["📍 Geospatial Engine<br>(Coordinate Sampling & Geodesic Dist)"]:::backend
        FeatureEng["⚙️ Feature Engineering Pipeline<br>(Lags, Volatility, Cyclic Time)"]:::backend
        
        Router --> Geospatial
        Geospatial --> FeatureEng
    end

    %% 4. Machine Learning Ensemble (The Brain)
    subgraph MLCore["Machine Learning Core (scikit-learn)"]
        direction TB
        subgraph BaseLearners["Tier 1: Base Learners"]
            RF["🌲 Random Forest"]:::ml
            XGB["🚀 XGBoost"]:::ml
            LGBM["⚡ LightGBM"]:::ml
            CAT["🐆 CatBoost"]:::ml
        end
        
        Meta["🧠 Ridge Meta-Regressor<br>(Stacking Ensemble)"]:::ml
        Output["📊 Final Hyper-Local PM2.5 Prediction"]:::ml
        
        RF --> Meta
        XGB --> Meta
        LGBM --> Meta
        CAT --> Meta
        Meta --> Output
    end
    
    %% 5. Databases / Storage
    subgraph Storage["Temporary Memory & Models"]
        ModelCache[("💾 Compiled Models<br>(/models folder)")]:::database
        StationHist[("⏳ Station History<br>(48-hr deque)")]:::database
    end

    %% Connections
    Map -- "Requests Route" --> Router
    Chat -- "Natural Language" --> Router
    Sim -- "Daily Routine" --> Router

    Router -- "Fetches Live AQI" --> WAQI
    Router -- "Fetches Route Path" --> ORS
    Geospatial -- "Fetches Live Weather" --> Weather
    
    WAQI --> StationHist
    StationHist -- "Provides Time-Series Lags" --> FeatureEng
    Weather -- "Provides Thermodynamics" --> FeatureEng
    
    FeatureEng -- "Feeds 18 Features" --> BaseLearners
    ModelCache -. "Loads .pkl Files" .-> MLCore
    
    Output -- "Returns Route Pollution Avgs" --> Router
    Router -- "Renders Fastest & Cleanest" --> Map
    Router -- "Returns Lung Aging Impact" --> Sim
```

### Key Components to describe in your Paper:
1. **Frontend Layer:** The interactive visual interface built on Leaflet.js, managing User Inputs (routines, queries, start/end points).
2. **External Data Ingestion:** Demonstrates real-time reliance on OpenWeatherMap (Thermodynamics) and WAQI (Base PM2.5).
3. **Geospatial & Feature Engineering:** Where the coordinates get sampled every 5th point and converted into the 18 advanced features (Sine/Cosine time, 24h lags, geodesic distance to highways).
4. **The Stacking Ensemble Layer:** Highlights the 4 distinct gradient boosters feeding into the final Ridge regressor, emphasizing the project's deep ML complexity.
