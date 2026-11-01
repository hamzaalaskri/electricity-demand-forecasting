from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from api.main import MAX_HISTORY, ForecastInputs, create_app
from src.config import FORECAST_HORIZON, MIN_HISTORY, MODEL_INPUT_NAMES
from tests.conftest import synthetic_series, synthetic_temperature

LAST_TIMESTAMP = datetime(2020, 2, 1, 12)


def default_inputs() -> ForecastInputs:
    series = synthetic_series(MIN_HISTORY, seed=1)
    temperature = synthetic_temperature(MIN_HISTORY + FORECAST_HORIZON, seed=1)
    end = series.index[-1]
    return ForecastInputs(
        series.tolist(), end.to_pydatetime(),
        temperature.loc[:end].iloc[-24:].tolist(),
        temperature.iloc[MIN_HISTORY:].tolist(),
    )


def custom_payload(**overrides) -> dict:
    payload = {
        "history": [30_000.0] * (MIN_HISTORY - 1) + [31_000.0],
        "last_timestamp": LAST_TIMESTAMP.isoformat(),
        "temperature_history": [5.0] * 24,
        "temperature_forecast": [6.0] * FORECAST_HORIZON,
    }
    payload.update(overrides)
    return {k: v for k, v in payload.items() if v is not None}


@pytest.fixture
def client(fake_model):
    app = create_app(
        model_loader=lambda: (fake_model, {"model_uri": "test", "version": "7"}),
        inputs_loader=default_inputs,
    )
    with TestClient(app) as test_client:
        yield test_client


def failing_loader():
    raise RuntimeError("boom")


def test_health_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "default_inputs_loaded": True,
    }


def test_model_info(client):
    body = client.get("/model").json()

    assert body["model_version"] == "7"
    assert body["forecast_horizon"] == FORECAST_HORIZON
    assert body["min_history"] == MIN_HISTORY
    assert body["expected_features"] == MODEL_INPUT_NAMES


def test_predict_from_default_inputs(client):
    response = client.post("/predict", json={})

    assert response.status_code == 200
    body = response.json()
    last = default_inputs().last_timestamp
    assert body["last_observation"] == last.isoformat()
    assert len(body["forecast"]) == FORECAST_HORIZON
    first = body["forecast"][0]
    assert first["horizon"] == 1
    assert first["timestamp"] == (last + timedelta(hours=1)).isoformat()


def test_predict_from_custom_inputs(client, fake_model):
    response = client.post("/predict", json=custom_payload())

    assert response.status_code == 200
    forecast = response.json()["forecast"]
    assert forecast[-1]["timestamp"] == (LAST_TIMESTAMP + timedelta(hours=24)).isoformat()
    assert forecast[0]["demand_mw"] == 31_001.0  # FakeModel: latest value + h
    features = fake_model.calls[-1]
    assert list(features.columns) == MODEL_INPUT_NAMES
    assert features.loc[0, "hour"] == 12
    assert features.loc[0, "temperature"] == 5.0
    assert features.loc[0, "temp_forecast_h24"] == 6.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"history": [30_000.0] * (MIN_HISTORY - 1)},
        {"history": [30_000.0] * (MAX_HISTORY + 1)},
        {"history": [30_000.0] * (MIN_HISTORY - 1) + [-5.0]},
        {"history": ["abc"] * MIN_HISTORY},
        {"last_timestamp": None},
        {"temperature_history": None},
        {"temperature_history": [5.0] * 23},
        {"temperature_forecast": [6.0] * 23},
        {"temperature_forecast": [6.0] * 23 + [99.0]},
    ],
    ids=["too-short", "too-long", "negative", "non-numeric", "missing-timestamp",
         "missing-temperature-history", "short-temperature-history",
         "short-forecast", "implausible-temperature"],
)
def test_invalid_requests_return_422(client, overrides):
    assert client.post("/predict", json=custom_payload(**overrides)).status_code == 422


def test_partial_request_without_history_returns_422(client):
    assert client.post("/predict", json={"last_timestamp": "2020-02-01T12:00:00"}).status_code == 422


def test_ready_when_model_loaded(client):
    assert client.get("/ready").status_code == 200


def test_degraded_when_model_fails_to_load():
    app = create_app(model_loader=failing_loader, inputs_loader=default_inputs)
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "degraded"
        assert client.get("/ready").status_code == 503
        assert client.get("/model").status_code == 503
        assert client.post("/predict", json={}).status_code == 503


def test_custom_inputs_still_work_without_bundled_data(fake_model):
    app = create_app(model_loader=lambda: (fake_model, {}), inputs_loader=failing_loader)
    with TestClient(app) as client:
        assert client.get("/health").json()["default_inputs_loaded"] is False
        assert client.post("/predict", json={}).status_code == 422
        assert client.post("/predict", json=custom_payload()).status_code == 200


def test_model_errors_return_500():
    class BrokenModel:
        def predict(self, features):
            return [1.0, 2.0]  # wrong length

    app = create_app(model_loader=lambda: (BrokenModel(), {}), inputs_loader=default_inputs)
    with TestClient(app) as client:
        response = client.post("/predict", json={})
        assert response.status_code == 500
        assert response.json()["detail"] == "Prediction failed."
