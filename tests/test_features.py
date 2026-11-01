import numpy as np
import pandas as pd
import pytest

from src.config import (
    FEATURE_NAMES,
    FORECAST_HORIZON,
    HORIZON_FEATURE_NAMES,
    MIN_HISTORY,
    MODEL_INPUT_NAMES,
    TARGET_NAMES,
)
from src.features import (
    build_features,
    build_inference_features,
    build_targets,
    horizon_matrix,
)


def test_feature_columns_and_order(series, temperature):
    features = build_features(series, temperature)
    assert list(features.columns) == MODEL_INPUT_NAMES


def test_first_complete_row_needs_min_history(series, temperature):
    features = build_features(series, temperature)
    first_complete = features.dropna().index[0]
    assert series.index.get_loc(first_complete) == MIN_HISTORY - 1


def test_lag_and_rolling_values(series, temperature):
    features = build_features(series, temperature)
    t = series.index[800]
    i = series.index.get_loc(t)

    assert features.loc[t, "lag_1"] == series.iloc[i - 1]
    assert features.loc[t, "lag_24"] == series.iloc[i - 24]
    assert features.loc[t, "lag_720"] == series.iloc[i - 720]
    assert features.loc[t, "rolling_mean_24"] == pytest.approx(series.iloc[i - 24:i].mean())
    assert features.loc[t, "rolling_mean_720"] == pytest.approx(series.iloc[i - 720:i].mean())


def test_temperature_features(series, temperature):
    features = build_features(series, temperature)
    t = series.index[800]

    assert features.loc[t, "temperature"] == temperature.loc[t]
    assert features.loc[t, "temperature_mean_24"] == pytest.approx(
        temperature.loc[t - pd.Timedelta(hours=23):t].mean())
    for h in (1, 13, FORECAST_HORIZON):
        assert features.loc[t, f"temp_forecast_h{h}"] == temperature.loc[t + pd.Timedelta(hours=h)]


def test_calendar_and_holiday_features(series, temperature):
    features = build_features(series, temperature)

    saturday = features.loc[pd.Timestamp("2020-01-04 18:00")]
    assert saturday["hour"] == 18
    assert saturday["day_of_week"] == 5
    assert saturday["is_weekend"] == 1
    assert saturday["hour_sin"] == pytest.approx(np.sin(2 * np.pi * 18 / 24))

    # 20 January 2020 was Martin Luther King Jr. Day.
    assert features.loc[pd.Timestamp("2020-01-20 10:00"), "is_holiday"] == 1
    assert features.loc[pd.Timestamp("2020-01-19 10:00"), "is_holiday_tomorrow"] == 1
    assert features.loc[pd.Timestamp("2020-01-21 10:00"), "is_holiday"] == 0


def test_demand_features_do_not_use_future_values(series, temperature):
    """Changing demand after t must not change the feature row at t."""
    t = series.index[900]
    original = build_features(series, temperature).loc[t]

    perturbed = series.copy()
    perturbed.iloc[901:] *= 10
    assert build_features(perturbed, temperature).loc[t].equals(original)


def test_horizon_matrix_selects_matching_forecast(series, temperature):
    features = build_features(series, temperature).dropna()

    matrix = horizon_matrix(features, 7)

    assert list(matrix.columns) == HORIZON_FEATURE_NAMES
    np.testing.assert_array_equal(matrix["temperature_target"], features["temp_forecast_h7"])
    pd.testing.assert_frame_equal(matrix[FEATURE_NAMES], features[FEATURE_NAMES])


def test_targets_are_future_values(series):
    targets = build_targets(series)
    i = 100

    assert list(targets.columns) == TARGET_NAMES
    for h in (1, 12, FORECAST_HORIZON):
        assert targets.iloc[i][f"target_{h}"] == series.iloc[i + h]
    assert targets.iloc[-1].isna().all()


def test_irregular_index_is_rejected(series, temperature):
    with pytest.raises(ValueError, match="regular hourly"):
        build_features(series.drop(series.index[10]), temperature)


def inference_inputs(series, temperature, end):
    forecast = temperature.loc[end + pd.Timedelta(hours=1):].iloc[:FORECAST_HORIZON]
    return (series.loc[:end].tolist(), end.to_pydatetime(),
            temperature.loc[:end].tolist(), forecast.tolist())


def test_inference_matches_training_features(series, temperature):
    """The API path must produce exactly the row the model was trained on."""
    training = build_features(series, temperature)
    end = series.index[-1]

    row = build_inference_features(*inference_inputs(series, temperature, end))

    pd.testing.assert_series_equal(
        row.iloc[0], training.loc[end], check_names=False, check_dtype=False
    )
    assert row.dtypes.equals(training.dtypes)


def test_inference_uses_only_latest_history(series, temperature):
    history, end, temps, forecast = inference_inputs(series, temperature, series.index[-1])

    pd.testing.assert_frame_equal(
        build_inference_features([1.0] * 50 + history, end, [0.0] * 50 + temps, forecast),
        build_inference_features(history, end, temps, forecast),
    )


def test_inference_accepts_timezone_aware_timestamp(series, temperature):
    history, end, temps, forecast = inference_inputs(series, temperature, series.index[-1])
    aware = pd.Timestamp(end).tz_localize("UTC").to_pydatetime()

    row = build_inference_features(history, aware, temps, forecast)

    assert row.loc[0, "hour"] == end.hour


GOOD_TEMPS = [10.0] * 24


@pytest.mark.parametrize(
    "history, temps, forecast, message",
    [
        ([30_000.0] * (MIN_HISTORY - 1), GOOD_TEMPS, GOOD_TEMPS, "at least"),
        ([30_000.0] * (MIN_HISTORY - 1) + [float("nan")], GOOD_TEMPS, GOOD_TEMPS, "non-finite"),
        ([30_000.0] * (MIN_HISTORY - 1) + [-1.0], GOOD_TEMPS, GOOD_TEMPS, "negative"),
        ([30_000.0] * MIN_HISTORY, [10.0] * 23, GOOD_TEMPS, "Temperature history"),
        ([30_000.0] * MIN_HISTORY, GOOD_TEMPS, [10.0] * 25, "exactly"),
    ],
)
def test_inference_rejects_invalid_inputs(history, temps, forecast, message):
    with pytest.raises(ValueError, match=message):
        build_inference_features(history, pd.Timestamp("2020-02-01").to_pydatetime(), temps, forecast)
