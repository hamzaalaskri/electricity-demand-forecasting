import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import FORECAST_HORIZON, MIN_HISTORY, TARGET_COL  # noqa: E402


def synthetic_temperature(hours: int, start: str = "2020-01-01 00:00", seed: int = 0) -> pd.Series:
    """Hourly temperature (°C) with seasonal and daily cycles."""
    rng = np.random.default_rng(seed + 100)
    index = pd.date_range(start, periods=hours, freq="h")
    t = np.arange(hours)
    values = (12 - 12 * np.cos(2 * np.pi * (t - 24 * 15) / (24 * 365))
              + 4 * np.sin(2 * np.pi * (t - 9) / 24) + rng.normal(0, 1, hours))
    return pd.Series(values, index=index, name="temperature")


def synthetic_series(hours: int, start: str = "2020-01-01 00:00", seed: int = 0) -> pd.Series:
    """Hourly demand with daily and weekly cycles, temperature sensitivity, and noise."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=hours, freq="h")
    t = np.arange(hours)
    temperature = synthetic_temperature(hours, start, seed).to_numpy()
    values = (
        30_000
        + 5_000 * np.sin(2 * np.pi * t / 24)
        + 2_000 * np.sin(2 * np.pi * t / 168)
        + 60 * (temperature - 15) ** 2
        + rng.normal(0, 300, hours)
    )
    return pd.Series(values, index=index, name=TARGET_COL)


def write_inputs(directory: Path, hours: int) -> tuple[Path, Path]:
    """Demand and weather CSVs in the formats the pipeline reads."""
    demand = synthetic_series(hours)
    temperature = synthetic_temperature(hours + 48, start=str(demand.index[0] - pd.Timedelta(hours=24)))
    data_path, weather_path = directory / "demand.csv", directory / "weather.csv"
    demand.rename_axis("Datetime").reset_index().to_csv(data_path, index=False)
    temperature.rename_axis("Datetime").reset_index().to_csv(weather_path, index=False)
    return data_path, weather_path


@pytest.fixture
def series() -> pd.Series:
    return synthetic_series(MIN_HISTORY + 500)


@pytest.fixture
def temperature(series) -> pd.Series:
    return synthetic_temperature(len(series) + FORECAST_HORIZON, start=str(series.index[0]))


class FakeModel:
    """Stands in for the MLflow PyFunc model: predicts the latest value + h."""

    def __init__(self):
        self.calls = []

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        self.calls.append(features)
        latest = float(features[TARGET_COL].iloc[0])
        return pd.DataFrame(
            [[latest + h for h in range(1, FORECAST_HORIZON + 1)]],
            columns=[f"forecast_h{h}" for h in range(1, FORECAST_HORIZON + 1)],
        )


@pytest.fixture
def fake_model() -> FakeModel:
    return FakeModel()
