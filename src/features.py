"""Feature and target construction shared by training and inference."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from src.config import (
    FEATURE_NAMES,
    FORECAST_HORIZON,
    HORIZON_FEATURE_NAMES,
    INTEGER_FEATURES,
    LAGS,
    MIN_HISTORY,
    MODEL_INPUT_NAMES,
    ROLLING_WINDOWS,
    TARGET_COL,
)

TEMPERATURE_WINDOW = 24


def build_features(series: pd.Series, temperature: pd.Series) -> pd.DataFrame:
    """Build the model input for every timestamp of a regular hourly series.

    Row ``t`` contains demand information observed at or before ``t``, the
    calendar, the temperature observed up to ``t``, and the temperature
    forecast for each target hour ``t+1 .. t+24``. During training, the
    forecasts are the observed temperatures ("perfect forecast"); the
    pipeline separately measures how much accuracy is lost when they carry
    realistic forecast error. Rows without enough history contain NaN and
    are dropped by the caller.
    """
    _require_hourly_index(series)
    _require_hourly_index(temperature)

    index = series.index
    features = pd.DataFrame(index=index)
    features[TARGET_COL] = series

    for lag in LAGS:
        features[f"lag_{lag}"] = series.shift(lag)

    features["hour"] = index.hour
    features["day_of_week"] = index.dayofweek
    features["month"] = index.month
    features["is_weekend"] = (index.dayofweek >= 5).astype(int)
    features["is_holiday"], features["is_holiday_tomorrow"] = _holiday_flags(index)
    features["hour_sin"] = np.sin(2 * np.pi * index.hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * index.hour / 24)

    # Windows end at t-1. The latest reading is already a feature in its own
    # right, so the rolling means describe the preceding demand level.
    previous = series.shift(1)
    for window in ROLLING_WINDOWS:
        features[f"rolling_mean_{window}"] = previous.rolling(window).mean()

    features["temperature"] = temperature.reindex(index)
    features["temperature_mean_24"] = (
        temperature.rolling(TEMPERATURE_WINDOW).mean().reindex(index)
    )
    for h in range(1, FORECAST_HORIZON + 1):
        features[f"temp_forecast_h{h}"] = temperature.shift(-h).reindex(index)

    features = features[MODEL_INPUT_NAMES]
    features[INTEGER_FEATURES] = features[INTEGER_FEATURES].astype("int32")
    return features


def horizon_matrix(features: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Input of the horizon-``h`` estimator: origin features plus the
    temperature forecast for ``t+h`` under a horizon-independent name."""
    matrix = features[FEATURE_NAMES].copy()
    matrix["temperature_target"] = features[f"temp_forecast_h{horizon}"].to_numpy()
    return matrix[HORIZON_FEATURE_NAMES]


def build_targets(series: pd.Series, horizon: int = FORECAST_HORIZON) -> pd.DataFrame:
    """Direct multi-horizon targets: ``target_h`` is the value at ``t+h``."""
    _require_hourly_index(series)
    return pd.DataFrame(
        {f"target_{h}": series.shift(-h) for h in range(1, horizon + 1)},
        index=series.index,
    )


def build_inference_features(
    history: Sequence[float],
    last_timestamp: datetime,
    temperature_history: Sequence[float],
    temperature_forecast: Sequence[float],
) -> pd.DataFrame:
    """One model-input row from the most recent readings.

    ``history`` holds at least 721 consecutive hourly demand values and
    ``temperature_history`` at least 24 hourly temperatures, both ending at
    ``last_timestamp``. ``temperature_forecast`` holds exactly 24 values for
    the following hours. Reuses :func:`build_features`, so inference is
    identical to training by construction.
    """
    values = _validated(history, "History", MIN_HISTORY, positive=True)
    temps = _validated(temperature_history, "Temperature history", TEMPERATURE_WINDOW)
    forecast = _validated(temperature_forecast, "Temperature forecast", FORECAST_HORIZON)
    if len(forecast) != FORECAST_HORIZON:
        raise ValueError(f"Temperature forecast must contain exactly {FORECAST_HORIZON} values.")

    end = pd.Timestamp(last_timestamp)
    if end.tzinfo is not None:
        end = end.tz_convert(None)

    values = values[-MIN_HISTORY:]
    demand = pd.Series(
        values, index=pd.date_range(end=end, periods=len(values), freq="h"), name=TARGET_COL
    )
    temps = temps[-MIN_HISTORY:]
    temperature = pd.Series(
        np.concatenate([temps, forecast]),
        index=pd.date_range(
            end=end + pd.Timedelta(hours=FORECAST_HORIZON),
            periods=len(temps) + FORECAST_HORIZON,
            freq="h",
        ),
    )

    row = build_features(demand, temperature).iloc[[-1]]
    if row.isna().any(axis=None):
        raise ValueError("Could not build a complete feature row from the inputs.")
    return row.reset_index(drop=True)


def _validated(values: Sequence[float], label: str, minimum: int, positive: bool = False) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < minimum:
        raise ValueError(f"{label} needs at least {minimum} hourly values.")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains missing or non-finite values.")
    if positive and (array <= 0).any():
        raise ValueError(f"{label} contains zero or negative demand values.")
    return array


def _holiday_flags(index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    days = index.normalize()
    holidays = USFederalHolidayCalendar().holidays(
        start=days.min(), end=days.max() + pd.Timedelta(days=1)
    )
    is_holiday = days.isin(holidays).astype(int)
    is_holiday_tomorrow = (days + pd.Timedelta(days=1)).isin(holidays).astype(int)
    return is_holiday, is_holiday_tomorrow


def _require_hourly_index(series: pd.Series) -> None:
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("Series must be indexed by a DatetimeIndex.")
    if len(series) > 1:
        steps = series.index[1:] - series.index[:-1]
        if not (steps == pd.Timedelta(hours=1)).all():
            raise ValueError("Series index must be strictly regular hourly.")
