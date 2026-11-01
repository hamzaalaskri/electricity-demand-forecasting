"""FastAPI inference service for 24-hour electricity demand forecasts.

The model is loaded once at startup from ``MODEL_URI``. By default this is
the self-contained MLflow model directory written by ``python -m src.train``;
it can also be a registry URI such as
``models:/ElectricityDemandForecaster@champion`` when ``MLFLOW_TRACKING_URI``
is set. ``DATA_PATH`` and ``WEATHER_PATH`` select the files used for the
default (no-input) forecast.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, model_validator

from src.config import (
    DATA_PATH,
    FORECAST_HORIZON,
    MIN_HISTORY,
    MODEL_DIR,
    MODEL_INPUT_NAMES,
    MODEL_NAME,
    WEATHER_PATH,
)
from src.data import load_series
from src.features import TEMPERATURE_WINDOW, build_inference_features
from src.weather import load_temperature

logger = logging.getLogger(__name__)

MAX_HISTORY = 24 * 365
API_VERSION = "3.0.0"

Temperature = Annotated[float, Field(ge=-50, le=60)]


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------


class PredictionRequest(BaseModel):
    history: list[float] | None = Field(
        default=None,
        min_length=MIN_HISTORY,
        max_length=MAX_HISTORY,
        description=(
            f"At least {MIN_HISTORY} consecutive hourly demand readings in MW, "
            "oldest first. Omit all fields to forecast from the latest bundled data."
        ),
    )
    last_timestamp: datetime | None = Field(
        default=None, description="Local timestamp of the last reading in `history`."
    )
    temperature_history: list[Temperature] | None = Field(
        default=None,
        min_length=TEMPERATURE_WINDOW,
        max_length=MAX_HISTORY,
        description=f"At least {TEMPERATURE_WINDOW} hourly temperatures (°C) ending at `last_timestamp`.",
    )
    temperature_forecast: list[Temperature] | None = Field(
        default=None,
        min_length=FORECAST_HORIZON,
        max_length=FORECAST_HORIZON,
        description=f"Temperature forecast (°C) for each of the next {FORECAST_HORIZON} hours.",
    )

    @model_validator(mode="after")
    def _check_completeness(self) -> PredictionRequest:
        fields = (self.history, self.last_timestamp, self.temperature_history,
                  self.temperature_forecast)
        provided = [f is not None for f in fields]
        if any(provided) and not all(provided):
            raise ValueError(
                "Custom requests need `history`, `last_timestamp`, "
                "`temperature_history` and `temperature_forecast` together."
            )
        return self


class ForecastPoint(BaseModel):
    horizon: int = Field(description="Hours after the last observation.")
    timestamp: datetime
    demand_mw: float


class PredictionResponse(BaseModel):
    model_name: str
    model_version: str
    last_observation: datetime
    forecast_horizon: int
    unit: str = "MW"
    forecast: list[ForecastPoint]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    default_inputs_loaded: bool


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    model_uri: str
    forecast_horizon: int
    min_history: int
    expected_features: list[str]
    metadata: dict[str, Any]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ForecastInputs:
    history: list[float]
    last_timestamp: datetime
    temperature_history: list[float]
    temperature_forecast: list[float]


def load_model_from_uri() -> tuple[Any, dict[str, Any]]:
    import mlflow

    model_uri = os.environ.get("MODEL_URI", str(MODEL_DIR))
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(tracking_uri)

    model = mlflow.pyfunc.load_model(model_uri)
    if model_uri.startswith("models:/"):
        version = model_uri.rsplit("/", 1)[-1].lstrip("@")
    else:
        version = "local"
    return model, {
        "model_uri": model_uri,
        "version": version,
        **(model.metadata.metadata or {}),
    }


def load_default_inputs() -> ForecastInputs:
    """Latest demand and temperature from the bundled files, cleaned exactly
    as in training. Temperatures after the last reading stand in for a
    weather forecast."""
    series, _ = load_series(Path(os.environ.get("DATA_PATH", DATA_PATH)))
    temperature = load_temperature(Path(os.environ.get("WEATHER_PATH", WEATHER_PATH)))

    tail = series.iloc[-MIN_HISTORY:]
    end = tail.index[-1]
    past = temperature.loc[:end].iloc[-TEMPERATURE_WINDOW:]
    future = temperature.loc[end + pd.Timedelta(hours=1):].iloc[:FORECAST_HORIZON]
    if len(future) < FORECAST_HORIZON:
        raise ValueError("Bundled weather does not cover the next 24 hours.")
    return ForecastInputs(tail.tolist(), end.to_pydatetime(), past.tolist(), future.tolist())


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------


def create_app(
    model_loader: Callable[[], tuple[Any, dict[str, Any]]] = load_model_from_uri,
    inputs_loader: Callable[[], ForecastInputs] = load_default_inputs,
) -> FastAPI:
    """Build the application. Loaders are injectable for testing."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model, app.state.model_metadata = None, {}
        app.state.default_inputs = None

        try:
            app.state.model, app.state.model_metadata = model_loader()
            logger.info("Model loaded: %s", app.state.model_metadata.get("model_uri"))
        except Exception:
            logger.exception("Could not load model; /predict will return 503.")

        try:
            app.state.default_inputs = inputs_loader()
        except Exception:
            logger.exception("Could not load default inputs; custom inputs will be required.")

        yield

    app = FastAPI(
        title="Electricity Demand Forecasting API",
        description="Direct multi-horizon (24h) hourly demand forecasts for PJM East.",
        version=API_VERSION,
        lifespan=lifespan,
    )

    def require_model(request: Request) -> Any:
        model = request.app.state.model
        if model is None:
            raise HTTPException(status_code=503, detail="Model is unavailable.")
        return model

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        model_loaded = request.app.state.model is not None
        return HealthResponse(
            status="ok" if model_loaded else "degraded",
            model_loaded=model_loaded,
            default_inputs_loaded=request.app.state.default_inputs is not None,
        )

    @app.get("/ready", response_model=HealthResponse)
    def ready(request: Request) -> HealthResponse:
        """Readiness probe: 503 until the model is loaded (used by the
        container HEALTHCHECK), whereas /health always answers."""
        status = health(request)
        if not status.model_loaded:
            raise HTTPException(status_code=503, detail="Model is not loaded.")
        return status

    @app.get("/model", response_model=ModelInfoResponse)
    def model_info(request: Request) -> ModelInfoResponse:
        require_model(request)
        metadata = dict(request.app.state.model_metadata)
        return ModelInfoResponse(
            model_name=MODEL_NAME,
            model_version=str(metadata.pop("version", "local")),
            model_uri=str(metadata.pop("model_uri", "")),
            forecast_horizon=FORECAST_HORIZON,
            min_history=MIN_HISTORY,
            expected_features=MODEL_INPUT_NAMES,
            metadata=metadata,
        )

    @app.post("/predict", response_model=PredictionResponse)
    def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
        model = require_model(request)

        if payload.history is not None:
            inputs = ForecastInputs(payload.history, payload.last_timestamp,
                                    payload.temperature_history, payload.temperature_forecast)
        elif request.app.state.default_inputs is not None:
            inputs = request.app.state.default_inputs
        else:
            raise HTTPException(
                status_code=422,
                detail="No bundled data is available; provide custom inputs.",
            )

        try:
            features = build_inference_features(
                inputs.history, inputs.last_timestamp,
                inputs.temperature_history, inputs.temperature_forecast,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        try:
            values = _to_forecast_array(model.predict(features))
        except Exception as exc:
            logger.exception("Prediction failed")
            raise HTTPException(status_code=500, detail="Prediction failed.") from exc

        last = inputs.last_timestamp
        return PredictionResponse(
            model_name=MODEL_NAME,
            model_version=str(request.app.state.model_metadata.get("version", "local")),
            last_observation=last,
            forecast_horizon=FORECAST_HORIZON,
            forecast=[
                ForecastPoint(horizon=h, timestamp=last + timedelta(hours=h),
                              demand_mw=round(float(value), 2))
                for h, value in enumerate(values, start=1)
            ],
        )

    return app


def _to_forecast_array(prediction: Any) -> np.ndarray:
    if isinstance(prediction, pd.DataFrame):
        values = prediction.iloc[0].to_numpy(dtype=np.float64)
    else:
        values = np.asarray(prediction, dtype=np.float64).reshape(-1)

    if len(values) != FORECAST_HORIZON:
        raise RuntimeError(f"Expected {FORECAST_HORIZON} predictions, got {len(values)}.")
    if not np.isfinite(values).all():
        raise RuntimeError("Model returned non-finite predictions.")
    return values


logging.basicConfig(level=logging.INFO)
app = create_app()
