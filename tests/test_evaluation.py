import numpy as np
import pandas as pd
import pytest

from src.config import FORECAST_HORIZON
from src.evaluation import baseline_forecasts, chronological_split, horizon_metrics, summarize


def test_split_is_chronological_and_purged():
    n, h = 10_000, FORECAST_HORIZON
    split = chronological_split(n, horizon=h)

    order = [split.fit, split.early_stopping, split.validation, split.test]
    for earlier, later in zip(order, order[1:]):
        # Every target of the earlier partition (row + h) precedes the later one.
        assert earlier.stop - 1 + h < later.start

    assert split.train.stop - 1 + h < split.validation.start
    assert split.train_validation.stop - 1 + h < split.test.start
    assert split.test.stop == n


def test_split_proportions():
    split = chronological_split(10_000)
    assert split.validation.start == 7_000
    assert split.test.start == 8_500


def test_split_rejects_tiny_dataset():
    with pytest.raises(ValueError):
        chronological_split(30)


def test_baselines_only_use_past_values(series):
    origins = series.index[200:210]
    forecasts = baseline_forecasts(series, origins)
    t = origins[0]
    i = series.index.get_loc(t)

    naive = forecasts["Naive"]
    assert (naive.loc[t] == series.iloc[i]).all()

    seasonal_24 = forecasts["Seasonal naive (24h)"]
    for h in (1, 24):
        # Forecast for t+h is the value 24 hours before t+h, i.e. at t+h-24 <= t.
        assert seasonal_24.loc[t, f"target_{h}"] == series.iloc[i + h - 24]

    seasonal_168 = forecasts["Seasonal naive (168h)"]
    assert seasonal_168.loc[t, "target_5"] == series.iloc[i + 5 - 168]


def test_horizon_metrics_known_values():
    y_true = pd.DataFrame({"target_1": [100.0, 200.0], "target_2": [100.0, 100.0]})
    y_pred = pd.DataFrame({"target_1": [110.0, 190.0], "target_2": [100.0, 80.0]})

    metrics = horizon_metrics(y_true, y_pred)

    assert metrics.loc[0, "mae"] == pytest.approx(10.0)
    assert metrics.loc[0, "rmse"] == pytest.approx(10.0)
    assert metrics.loc[0, "mape"] == pytest.approx((10 + 5) / 2)
    assert metrics.loc[1, "mae"] == pytest.approx(10.0)
    assert metrics.loc[1, "rmse"] == pytest.approx(np.sqrt(200.0))

    summary = summarize(metrics)
    assert summary["mae_h1"] == pytest.approx(10.0)
    assert summary["mean_mae"] == pytest.approx(10.0)
