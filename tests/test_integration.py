"""End-to-end checks against the real exported model and dataset.

Skipped automatically until ``python -m src.train`` has produced ``models/``.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.config import DATA_PATH, FORECAST_HORIZON, MODEL_DIR, WEATHER_PATH

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (MODEL_DIR / "MLmodel").exists() or not DATA_PATH.exists() or not WEATHER_PATH.exists(),
        reason="Trained model or data not available.",
    ),
]


@pytest.fixture(scope="module")
def client():
    from api.main import create_app

    with TestClient(create_app()) as test_client:
        yield test_client


def test_real_model_loads(client):
    assert client.get("/health").json()["status"] == "ok"
    metadata = client.get("/model").json()["metadata"]
    assert metadata["forecast_horizon"] == FORECAST_HORIZON


def test_real_forecast_is_plausible(client):
    response = client.post("/predict", json={})

    assert response.status_code == 200
    values = np.array([p["demand_mw"] for p in response.json()["forecast"]])
    assert len(values) == FORECAST_HORIZON
    # PJME demand historically stays between roughly 14 and 63 GW.
    assert ((values > 10_000) & (values < 70_000)).all()


def test_hotter_forecast_raises_summer_demand(client):
    """The model must respond to temperature in the physically expected way."""
    from api.main import load_default_inputs

    inputs = load_default_inputs()  # ends 3 August 2018: summer
    base = {
        "history": inputs.history,
        "last_timestamp": inputs.last_timestamp.isoformat(),
        "temperature_history": inputs.temperature_history,
    }
    normal = client.post("/predict", json={**base, "temperature_forecast": inputs.temperature_forecast})
    hotter = client.post("/predict", json={
        **base, "temperature_forecast": [t + 4 for t in inputs.temperature_forecast]})

    def total(response):
        return sum(p["demand_mw"] for p in response.json()["forecast"])

    assert total(hotter) > total(normal) * 1.02
