import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_NAMES, TEMPERATURE_FORECAST_NAMES
from src.modeling import forecast_error_sigma, with_forecast_error
from src.weather import load_temperature
from src.data import DataValidationError


def frame(rows=5000):
    rng = np.random.default_rng(0)
    columns = FEATURE_NAMES + TEMPERATURE_FORECAST_NAMES
    return pd.DataFrame(rng.normal(10, 5, (rows, len(columns))), columns=columns)


def test_forecast_error_only_changes_forecasts():
    X = frame()
    noisy = with_forecast_error(X, seed=1)

    pd.testing.assert_frame_equal(noisy[FEATURE_NAMES], X[FEATURE_NAMES])
    assert not np.allclose(noisy[TEMPERATURE_FORECAST_NAMES], X[TEMPERATURE_FORECAST_NAMES])


def test_forecast_error_grows_with_horizon_and_is_reproducible():
    X = frame()
    errors = with_forecast_error(X, seed=1)[TEMPERATURE_FORECAST_NAMES] - X[TEMPERATURE_FORECAST_NAMES]

    assert errors["temp_forecast_h1"].std() == pytest.approx(forecast_error_sigma(1), rel=0.1)
    assert errors["temp_forecast_h24"].std() == pytest.approx(forecast_error_sigma(24), rel=0.1)
    assert errors["temp_forecast_h24"].std() > 3 * errors["temp_forecast_h1"].std()
    pd.testing.assert_frame_equal(with_forecast_error(X, seed=1), with_forecast_error(X, seed=1))


def test_weather_loader_validates(tmp_path):
    path = tmp_path / "weather.csv"
    pd.DataFrame({"Datetime": ["2020-01-01 00:00", "2020-01-01 01:00"],
                  "temperature": [5.0, 80.0]}).to_csv(path, index=False)
    with pytest.raises(DataValidationError, match="implausible"):
        load_temperature(path)

    with pytest.raises(FileNotFoundError, match="src.weather"):
        load_temperature(tmp_path / "missing.csv")


def test_weather_loader_regularizes_dst(tmp_path):
    path = tmp_path / "weather.csv"
    pd.DataFrame({
        "Datetime": ["2020-11-01 00:00", "2020-11-01 01:00", "2020-11-01 01:00", "2020-11-01 03:00"],
        "temperature": [4.0, 2.0, 4.0, 6.0],
    }).to_csv(path, index=False)

    series = load_temperature(path)

    assert series.tolist() == [4.0, 3.0, 4.5, 6.0]
