"""
API Integration tests for AIRAWARE Flask server endpoints.
"""

import pytest
import json
from app import app


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_index_route(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b"AIRAWARE" in response.data


def test_health_endpoint(client):
    response = client.get('/api/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["status"] == "healthy"
    assert "model_loaded" in data


def test_live_aqi_endpoint(client):
    response = client.get('/api/live-aqi')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert isinstance(data, list)
    assert len(data) > 0
    first_node = data[0]
    assert "location" in first_node
    assert "aqi" in first_node
    assert "raw_aqi" in first_node


def test_predict_point_endpoint(client):
    payload = {"lat": 28.6315, "lon": 77.2167}
    response = client.post(
        '/api/predict-point',
        data=json.dumps(payload),
        content_type='application/json'
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "predicted_pm25" in data
    assert isinstance(data["predicted_pm25"], (int, float))


def test_health_advice_endpoint(client):
    payload = {"age": 30, "asthma": True, "aqi": 180}
    response = client.post(
        '/api/health-advice',
        data=json.dumps(payload),
        content_type='application/json'
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "best_time" in data
    assert "mask" in data
    assert "activity" in data


def test_simulator_endpoint(client):
    payload = {
        "routine": [{"location": "Connaught Place", "duration_hours": 3}],
        "years": 1,
        "changes": {"mask": True, "indoor": False}
    }
    response = client.post(
        '/api/simulator',
        data=json.dumps(payload),
        content_type='application/json'
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "base_exposure_per_day" in data
    assert "base_lung_aging_years" in data
    assert "what_if_exposure" in data


def test_chatbot_endpoint(client):
    payload = {"message": "What is the current AQI in Delhi?"}
    response = client.post(
        '/api/chat',
        data=json.dumps(payload),
        content_type='application/json'
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "response" in data
