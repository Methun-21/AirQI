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
    assert "primary_model" in data
    assert "uncertainty_estimator" in data


def test_live_aqi_endpoint(client):
    response = client.get('/api/live-aqi')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert isinstance(data, list)
    assert len(data) > 0
    first_node = data[0]
    assert "location" in first_node
    assert "aqi" in first_node
    assert "pm2_5" in first_node


def test_predict_point_endpoint(client):
    payload = {"lat": 28.6315, "lon": 77.2167}
    response = client.post(
        '/api/predict-point',
        data=json.dumps(payload),
        content_type='application/json'
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "pm25_median" in data
    assert "ci_lower" in data
    assert "ci_upper" in data
    assert data["ci_lower"] <= data["ci_upper"]


def test_routes_endpoint(client):
    payload = {
        "start": [77.2167, 28.6315],
        "end": [77.0460, 28.5921]
    }
    response = client.post(
        '/api/routes',
        data=json.dumps(payload),
        content_type='application/json'
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) > 0
    first_feat = data["features"][0]
    assert "expected_pm25" in first_feat["properties"]
    assert "ci_80" in first_feat["properties"]
