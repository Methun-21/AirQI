const API_BASE = "/api";
const map = L.map('map_container', { zoomControl: false, attributionControl: false }).setView([28.6139, 77.2090], 12);

// Check backend API Health Status
fetch(`${API_BASE}/health`)
    .then(res => res.json())
    .then(data => {
        if (data.status === "healthy") {
            console.log("AIRAWARE Probabilistic Spatial Engine Connected:", data);
        }
    })
    .catch(err => console.warn("Backend health check warning:", err));

// Custom Dark Map Theme
L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: ''
}).addTo(map);

document.getElementById('map_container').classList.add('active');

const locations = {
    "Connaught Place": [77.2167, 28.6315],
    "Karol Bagh": [77.1907, 28.6517],
    "Chandni Chowk": [77.2300, 28.6562],
    "Dwarka": [77.0460, 28.5921],
    "Saket": [77.2066, 28.5245],
    "Rohini": [77.1200, 28.7360],
    "Lajpat Nagar": [77.2433, 28.5672],
    "Mayur Vihar": [77.2900, 28.6034],
    "Vasant Kunj": [77.1500, 28.5270],
    "Delhi University": [77.2090, 28.6863],
    "India Gate": [77.2295, 28.6129],
    "Anand Vihar": [77.3155, 28.6473],
    "ITO Crossing": [77.2479, 28.6307]
};

// Fetch Live Telemetry and Render Markers
fetch(`${API_BASE}/live-aqi`)
    .then(res => res.json())
    .then(data => {
        data.forEach(row => {
            let color = row.pm2_5 <= 60 ? '#10b981' : row.pm2_5 <= 150 ? '#f59e0b' : '#ef4444';
            let markerIcon = L.divIcon({
                className: 'custom-div-icon',
                html: `<div style="background-color: ${color}; width: 14px; height: 14px; border-radius: 50%; border: 3px solid white; box-shadow: 0 0 12px ${color};" class="aqi-marker-active"></div>`,
                iconSize: [14, 14],
                iconAnchor: [7, 7]
            });

            L.marker([row.lat, row.lon], { icon: markerIcon })
                .bindPopup(`<div style="color:#f8fafc; padding:6px; font-family:'Outfit',sans-serif;">
                                <div style="font-weight: 800; font-size: 1.05rem; border-bottom: 1px solid rgba(255,255,255,0.1); margin-bottom: 6px; padding-bottom: 4px;">${row.location}</div>
                                <div style="display:flex; justify-content: space-between; align-items:center; margin-bottom:4px;">
                                    <span style="font-size:0.8rem; color:#94a3b8;">PM2.5 Mass</span>
                                    <strong style="color:${color}; font-size:1.1rem;">${row.pm2_5} µg/m³</strong>
                                </div>
                                <div style="display:flex; justify-content: space-between; align-items:center;">
                                    <span style="font-size:0.8rem; color:#94a3b8;">Observed AQI</span>
                                    <strong style="color:#f8fafc; font-size:0.95rem;">${row.aqi}</strong>
                                </div>
                            </div>`)
                .addTo(map);
        });
        renderCharts(data);
    });

// Sensor Network Bar Chart
function renderCharts(data) {
    const ctx = document.getElementById('aqiBarChart').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.slice(0, 6).map(d => d.location.split(' ')[0]),
            datasets: [{
                label: 'PM2.5 (µg/m³)',
                data: data.slice(0, 6).map(d => d.pm2_5),
                backgroundColor: data.slice(0, 6).map(d => d.pm2_5 > 150 ? '#ef4444' : d.pm2_5 > 60 ? '#f59e0b' : '#10b981'),
                borderRadius: 6,
                barThickness: 20
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
                x: { grid: { display: false }, ticks: { color: '#94a3b8', font: {family: 'Outfit'} } }
            }
        }
    });
}

// Clean Route Optimization with Uncertainty Intervals
let currentRoute = null;
function getRoute() {
    const start = locations[document.getElementById("start").value];
    const end = locations[document.getElementById("end").value];

    if (currentRoute) map.removeLayer(currentRoute);
    const compBox = document.getElementById("route-comparison-box");
    compBox.style.display = "none";
    compBox.innerHTML = ""; 
    
    showToast("Running probabilistic quantile route inference...", "fa-route");

    fetch(`${API_BASE}/routes`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ start, end })
    })
    .then(res => res.json())
    .then(data => {
        if (!data || !data.features || data.features.length === 0) {
            showToast("Could not calculate route for selected points", "fa-exclamation-triangle");
            return;
        }
        currentRoute = L.geoJSON(data, {
            style: (f) => {
                if (f.properties.route_type.includes("Cleanest")) {
                    return { color: '#10b981', weight: 6, opacity: 0.95, lineCap: 'round' }; 
                } else if (f.properties.route_type.includes("Fastest")) {
                    return { color: '#ef4444', weight: 4, opacity: 0.7, dashArray: '6, 8', lineCap: 'round' }; 
                }
                return { color: '#0ea5e9', weight: 5, opacity: 0.9 };
            },
            onEachFeature: (f, l) => {
                let isClean = f.properties.route_type.includes("Cleanest") || f.properties.route_type.includes("Fastest & Cleanest");
                let cardColor = isClean ? "rgba(16, 185, 129, 0.12)" : "rgba(239, 68, 68, 0.12)";
                let iconColor = isClean ? "#10b981" : "#ef4444";
                let iconType = isClean ? "fa-shield-alt" : "fa-tachometer-alt";

                let ci = f.properties.ci_80 || [f.properties.expected_pm25, f.properties.worst_case_pm25];

                compBox.innerHTML += `
                    <div style="background: ${cardColor}; border: 1px solid ${iconColor}40; padding: 12px 14px; border-radius: 14px; display: flex; align-items: center; justify-content: space-between;">
                        <div style="display: flex; align-items: center; gap: 12px;">
                            <div style="width: 34px; height: 34px; border-radius: 17px; background: rgba(0,0,0,0.3); display:flex; align-items:center; justify-content:center;">
                                <i class="fas ${iconType}" style="color: ${iconColor}; font-size: 1.0rem;"></i>
                            </div>
                            <div>
                                <div style="font-weight: 700; font-size: 0.9rem; color: #fff;">${f.properties.route_type}</div>
                                <div style="font-size: 0.75rem; color: var(--text-secondary);">
                                    Expected: <strong style="color: ${iconColor};">${f.properties.expected_pm25} µg/m³</strong> 
                                    <span style="color:#94a3b8; font-size:0.7rem;">[80% CI: ${ci[0]}–${ci[1]}]</span>
                                </div>
                            </div>
                        </div>
                        <div style="background: ${iconColor}; color: ${isClean ? 'black' : 'white'}; padding: 4px 8px; border-radius: 6px; font-size: 0.65rem; font-weight: 800; letter-spacing: 0.05em;">
                            ${isClean ? (f.properties.exposure_reduction_pct ? `-${f.properties.exposure_reduction_pct}% DOSE` : 'OPTIMAL') : 'HIGH RISK'}
                        </div>
                    </div>
                `;
            }
        }).addTo(map);

        compBox.style.display = "flex";
        if (currentRoute.getBounds().isValid()) {
            map.fitBounds(currentRoute.getBounds(), { padding: [50, 50] });
        }
    })
    .catch(() => showToast("Error connecting to routing engine", "fa-exclamation-triangle"));
}

// Toast Notifications
function showToast(message, icon = "fa-info-circle") {
    const container = document.getElementById("toast-container");
    const toast = document.createElement("div");
    toast.className = "toast";
    toast.innerHTML = `<i class="fas ${icon}" style="color: var(--accent); font-size: 1.2rem;"></i> <span>${message}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = "0";
        toast.style.transform = "translateY(-20px) scale(0.95)";
        setTimeout(() => toast.remove(), 400);
    }, 3500);
}

// Live ML Inference on Map Click with Quantile Uncertainty
let livePredictionMarker = null;
map.on('click', function(e) {
    const lat = e.latlng.lat;
    const lon = e.latlng.lng;
    
    showToast("Evaluating micro-climate quantile model...", "fa-crosshairs");
    
    if (livePredictionMarker) map.removeLayer(livePredictionMarker);
    livePredictionMarker = L.marker([lat, lon], {
        icon: L.divIcon({
            className: 'custom-div-icon',
            html: `<div style="background-color: var(--text-secondary); width: 14px; height: 14px; border-radius: 50%; border: 2px solid white; animation: pulse 1s infinite;"></div>`,
            iconSize: [14, 14],
            iconAnchor: [7, 7]
        })
    }).addTo(map);

    fetch(`${API_BASE}/predict-point`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ lat, lon })
    })
    .then(res => res.json())
    .then(data => {
        if (data.error) {
            showToast("Inference failed: " + data.error, "fa-exclamation-triangle");
            return;
        }

        let pm25 = data.pm25_median;
        let color = pm25 <= 60 ? '#10b981' : pm25 <= 150 ? '#f59e0b' : '#ef4444';
        let status = pm25 <= 60 ? 'Moderate Air' : pm25 <= 150 ? 'Unhealthy' : 'Severe Inversion';

        map.removeLayer(livePredictionMarker);
        livePredictionMarker = L.marker([lat, lon], {
            icon: L.divIcon({
                className: 'custom-div-icon',
                html: `<div style="background-color: ${color}; width: 18px; height: 18px; border-radius: 50%; border: 3px solid white; box-shadow: 0 0 15px ${color};" class="aqi-marker-active"></div>`,
                iconSize: [18, 18],
                iconAnchor: [9, 9]
            })
        })
        .bindPopup(`
            <div style="color: #f8fafc; padding: 4px; min-width: 190px; font-family: 'Outfit', sans-serif;">
                <div style="font-weight: 800; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 6px; margin-bottom: 8px; display:flex; align-items:center; gap:6px;">
                    <i class="fas fa-brain" style="color: var(--accent);"></i> Spatial Forecast
                </div>
                <div style="font-size: 1.5rem; font-weight: 800; color: ${color}; line-height:1; margin-bottom:2px;">
                    ${pm25} <span style="font-size:0.75rem; font-weight:600; color:#94a3b8;">µg/m³ (P50)</span>
                </div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-bottom: 8px;">
                    80% Prediction Interval: <strong style="color: #f8fafc;">${data.ci_lower} – ${data.ci_upper} µg/m³</strong>
                </div>
                <div style="font-size: 0.8rem; font-weight: 700; color: #cbd5e1; margin-bottom: 6px;">
                    EPA AQI Equivalent: <strong style="color: ${color};">${data.calculated_aqi}</strong> (${status})
                </div>
                <div style="font-size: 0.7rem; color: #94a3b8; margin-top: 6px; border-top: 1px solid rgba(255,255,255,0.05); padding-top:4px;">
                    Nearest Reference Node: <strong>${data.nearest_station}</strong>
                </div>
            </div>
        `, { closeButton: false, className: 'premium-popup' })
        .addTo(map)
        .openPopup();
        
        showToast("Spatial inference complete", "fa-check-circle");
    })
    .catch(err => {
        showToast("Server unreachable", "fa-times-circle");
        if(livePredictionMarker) map.removeLayer(livePredictionMarker);
    });
});
