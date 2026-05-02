const API_BASE = "http://127.0.0.1:5001/api";
const map = L.map('map_container', { zoomControl: false, attributionControl: false }).setView([28.6139, 77.2090], 12);

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

// Admin Toggle Logic
function setViewMode(mode) {
    document.querySelectorAll('.view-toggle-option').forEach(el => el.classList.remove('active'));
    if(mode === 'admin') {
        document.body.classList.add('admin-mode');
        document.getElementById('toggle-admin').classList.add('active');
        showToast("Admin View Active: Highlighting Critical Hotspots", "fa-shield-alt");
    } else {
        document.body.classList.remove('admin-mode');
        document.getElementById('toggle-citizen').classList.add('active');
        showToast("Citizen View Active: Routine & Routing", "fa-user");
    }
}

// Fetch Live AQI and Render Markers
let markers = [];
fetch(`${API_BASE}/live-aqi`)
    .then(res => res.json())
    .then(data => {
        data.forEach(row => {
            let color = row.aqi <= 2 ? '#10b981' : row.aqi === 3 ? '#f59e0b' : '#ef4444';
            let markerIcon = L.divIcon({
                className: 'custom-div-icon',
                html: `<div style="background-color: ${color}; width: 14px; height: 14px; border-radius: 50%; border: 3px solid white; box-shadow: 0 0 12px ${color};" class="aqi-marker-active"></div>`,
                iconSize: [14, 14],
                iconAnchor: [7, 7]
            });

            L.marker([row.lat, row.lon], { icon: markerIcon })
                .bindPopup(`<div style="color:#0f172a; padding:8px;">
                                <div style="font-weight: 800; font-size: 1.1rem; border-bottom: 1px solid #e2e8f0; margin-bottom: 5px; padding-bottom: 5px;">${row.location}</div>
                                <div style="display:flex; justify-content: space-between; align-items:center;">
                                    <span>AQI</span>
                                    <strong style="color:${color}; font-size:1.2rem;">${row.raw_aqi}</strong>
                                </div>
                                <small style="color: #64748b; display:block; margin-top:5px;">${row.aqi <= 2 ? 'Clean Air' : 'Hazardous'}</small>
                            </div>`)
                .addTo(map);
        });
        renderCharts(data);
    });

// Charts Implementation
function renderCharts(data) {
    const ctx = document.getElementById('aqiBarChart').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.slice(0, 5).map(d => d.location.split(' ')[0]),
            datasets: [{
                label: 'AQI Level',
                data: data.slice(0, 5).map(d => d.raw_aqi),
                backgroundColor: data.slice(0, 5).map(d => d.raw_aqi > 200 ? '#ef4444' : '#0ea5e9'),
                borderRadius: 8,
                barThickness: 25
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

// Routing Logic
let currentRoute = null;
function getRoute() {
    const start = locations[document.getElementById("start").value];
    const end = locations[document.getElementById("end").value];

    if (currentRoute) map.removeLayer(currentRoute);
    const compBox = document.getElementById("route-comparison-box");
    compBox.style.display = "none";
    compBox.innerHTML = ""; 
    
    showToast("Calculating ML-optimized paths...", "fa-route");

    fetch(`${API_BASE}/routes`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ start, end })
    })
    .then(res => res.json())
    .then(data => {
        currentRoute = L.geoJSON(data, {
            style: (f) => {
                if (f.properties.route_type.includes("Cleanest")) {
                    return { color: '#10b981', weight: 6, opacity: 0.9, lineCap: 'round' }; 
                } else if (f.properties.route_type.includes("Fastest")) {
                    return { color: '#ef4444', weight: 4, opacity: 0.6, dashArray: '5, 10', lineCap: 'round' }; 
                }
                return { color: '#0ea5e9', weight: 4, opacity: 0.8 };
            },
            onEachFeature: (f, l) => {
                let isCombo = f.properties.route_type === "Fastest & Cleanest";
                let cardColor = f.properties.route_type.includes("Cleanest") ? "rgba(16, 185, 129, 0.1)" : "rgba(239, 68, 68, 0.1)";
                let iconColor = f.properties.route_type.includes("Cleanest") ? "#10b981" : "#ef4444";
                let iconType = f.properties.route_type.includes("Cleanest") ? "fa-leaf" : "fa-clock";

                if (isCombo) {
                    cardColor = "linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(14, 165, 233, 0.15))";
                    iconType = "fa-star";
                }

                compBox.innerHTML += `
                    <div style="background: ${cardColor}; border: 1px solid ${iconColor}40; padding: 14px; border-radius: 16px; display: flex; align-items: center; justify-content: space-between;">
                        <div style="display: flex; align-items: center; gap: 12px;">
                            <div style="width: 36px; height: 36px; border-radius: 18px; background: rgba(0,0,0,0.2); display:flex; align-items:center; justify-content:center;">
                                <i class="fas ${iconType}" style="color: ${iconColor}; font-size: 1.1rem;"></i>
                            </div>
                            <div>
                                <div style="font-weight: 700; font-size: 0.95rem; color: #fff;">${f.properties.route_type}</div>
                                <div style="font-size: 0.75rem; color: var(--text-secondary);">Avg PM2.5: <strong style="color: ${iconColor};">${f.properties.avg_pollution}</strong></div>
                            </div>
                        </div>
                        <div style="background: ${iconColor}; color: ${isCombo ? 'white' : 'black'}; padding: 4px 10px; border-radius: 8px; font-size: 0.65rem; font-weight: 800; letter-spacing: 0.05em;">
                            ${f.properties.route_type.includes("Cleanest") ? "RECOMMENDED" : "AVOID"}
                        </div>
                    </div>
                `;
            }
        }).addTo(map);

        compBox.style.display = "flex";
        map.fitBounds(currentRoute.getBounds(), { padding: [50, 50] });
    })
    .catch(() => showToast("Error connecting to routing engine", "fa-exclamation-triangle"));
}

// Health Engine
function getHealthAdvice() {
    const age = document.getElementById("age").value;
    const asthma = document.getElementById("asthma").checked;
    fetch(`${API_BASE}/health-advice`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ age: Number(age), asthma, aqi: 200 })
    })
    .then(res => res.json())
    .then(d => {
        showToast(`Advice: ${d.activity} | Mask: ${d.mask}`, "fa-stethoscope");
    });
}

// Simulator
let simChart = null;
function simulateExposure() {
    const location = document.getElementById('sim-location').value;
    const hours = parseFloat(document.getElementById('sim-hours').value);

    fetch(`${API_BASE}/simulator`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ routine: [{ location, duration_hours: hours }], years: 1 })
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById('sim-result').innerHTML = `
        <div class="health-advice-box">
            <div class="health-stat"><div>LUNG AGING</div><div style="color:#ef4444;">+${data.base_lung_aging_years} yrs</div></div>
            <div class="health-stat"><div>RISK REDUCTION</div><div style="color:#10b981;">-${data.what_if_reduction}%</div></div>
        </div>
        <p style="margin-top:10px; font-size:0.8rem; color:var(--text-secondary);">${data.base_risk_reduction_tip}</p>
        `;

        const ctx = document.getElementById('simulator-chart').getContext('2d');
        if (simChart) simChart.destroy();
        simChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Exposure', 'Clean Capacity'],
                datasets: [{
                    data: [data.base_exposure_per_day, 500 - data.base_exposure_per_day],
                    backgroundColor: ['#ef4444', 'rgba(255, 255, 255, 0.05)'],
                    borderWidth: 0,
                    cutout: '80%'
                }]
            },
            options: { maintainAspectRatio: false, plugins: { legend: { display: false } } }
        });
    });
}

// Chatbot
function toggleChatbot() {
    const chat = document.getElementById('chatbot');
    chat.style.display = chat.style.display === 'flex' ? 'none' : 'flex';
}

function sendMessage() {
    const input = document.getElementById('chat-input');
    const msg = input.value;
    if (!msg) return;

    const msgContainer = document.getElementById('chat-messages');
    msgContainer.innerHTML += `<div class="msg user-msg">${msg}</div>`;
    input.value = '';

    fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: msg })
    })
    .then(res => res.json())
    .then(data => {
        msgContainer.innerHTML += `<div class="msg bot-msg">${data.response.replace(/\n/g, '<br>')}</div>`;
        msgContainer.scrollTop = msgContainer.scrollHeight;
    });
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

// --- LIVE ML PREDICTION ON CLICK ---
let livePredictionMarker = null;
map.on('click', function(e) {
    const lat = e.latlng.lat;
    const lon = e.latlng.lng;
    
    showToast("Analyzing micro-climate & running ML inference...", "fa-microchip");
    
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

        let pm25 = data.predicted_pm25;
        let color = pm25 <= 50 ? '#10b981' : pm25 <= 150 ? '#f59e0b' : '#ef4444';
        let status = pm25 <= 50 ? 'Clean Air' : pm25 <= 150 ? 'Moderate Risk' : 'Hazardous';

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
            <div style="color: #0f172a; padding: 5px; min-width: 160px; font-family: 'Outfit', sans-serif;">
                <div style="font-weight: 800; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px; margin-bottom: 8px; display:flex; align-items:center; gap:6px;">
                    <i class="fas fa-robot" style="color: var(--accent);"></i> Live ML Inference
                </div>
                <div style="font-size: 1.6rem; font-weight: 800; color: ${color}; line-height:1;">
                    ${pm25}
                </div>
                <div style="font-size: 0.8rem; font-weight: 700; color: #64748b; margin-bottom: 10px; text-transform:uppercase; letter-spacing:0.05em;">
                    ${status}
                </div>
                <div style="font-size: 0.85rem; color: #475569; display:flex; gap:10px; background:#f1f5f9; padding:6px; border-radius:8px;">
                    <span><i class="fas fa-temperature-high" style="color:#f59e0b;"></i> ${data.live_weather.temp}°C</span>
                    <span><i class="fas fa-wind" style="color:#0ea5e9;"></i> ${data.live_weather.wind}m/s</span>
                </div>
                <div style="font-size: 0.7rem; color: #94a3b8; margin-top: 8px; text-align:center;">
                    Nearest Node: <strong>${data.nearest_station}</strong>
                </div>
            </div>
        `, { closeButton: false, className: 'premium-popup' })
        .addTo(map)
        .openPopup();
        
        showToast("Inference complete", "fa-check-circle");
    })
    .catch(err => {
        showToast("Server unreachable", "fa-times-circle");
        if(livePredictionMarker) map.removeLayer(livePredictionMarker);
    });
});
