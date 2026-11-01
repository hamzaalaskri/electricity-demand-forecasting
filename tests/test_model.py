import numpy as np
import pandas as pd
import pytest

from src.config import FORECAST_HORIZON, HORIZON_FEATURE_NAMES, MODEL_INPUT_NAMES
from src.features import build_features, build_targets
from src.model import ElectricityDemandForecaster, load_estimators, save_estimators
from src.modeling import fit_direct, predict_direct


class ConstantRegressor:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        assert list(X.columns) == HORIZON_FEATURE_NAMES
        return np.full(len(X), self.value, dtype=float)


def make_forecaster():
    return ElectricityDemandForecaster(
        {h: ConstantRegressor(1000.0 + h) for h in range(1, FORECAST_HORIZON + 1)}
    )


def zeros(rows=1, columns=MODEL_INPUT_NAMES):
    return pd.DataFrame(np.zeros((rows, len(columns))), columns=columns)


def test_output_has_one_column_per_horizon_in_order():
    output = make_forecaster().predict(None, zeros(3))

    assert output.shape == (3, FORECAST_HORIZON)
    assert list(output.columns) == [f"forecast_h{h}" for h in range(1, FORECAST_HORIZON + 1)]
    assert output.iloc[0, 0] == 1001.0
    assert output.iloc[0, -1] == 1000.0 + FORECAST_HORIZON


def test_extra_columns_are_ignored_and_order_is_enforced():
    X = zeros()[MODEL_INPUT_NAMES[::-1]].assign(extra=1)
    assert make_forecaster().predict(None, X).shape == (1, FORECAST_HORIZON)


def test_missing_feature_raises():
    with pytest.raises(ValueError, match="Missing input features"):
        make_forecaster().predict(None, zeros(columns=MODEL_INPUT_NAMES[:-1]))


def test_horizon_gaps_are_rejected():
    with pytest.raises(ValueError):
        ElectricityDemandForecaster({1: ConstantRegressor(0), 3: ConstantRegressor(0)})


@pytest.fixture(scope="module")
def small_dataset():
    from tests.conftest import synthetic_series, synthetic_temperature

    series = synthetic_series(1500)
    temperature = synthetic_temperature(1500 + FORECAST_HORIZON)
    frame = build_features(series, temperature).join(build_targets(series)).dropna()
    return frame[MODEL_INPUT_NAMES], frame[[f"target_{h}" for h in range(1, 25)]]


@pytest.mark.parametrize("family", ["LightGBM", "XGBoost"])
def test_native_format_round_trip_preserves_predictions(tmp_path, small_dataset, family):
    X, Y = small_dataset
    params = {"n_estimators": 15, "learning_rate": 0.1, "random_state": 0, "n_jobs": 1}
    if family == "LightGBM":
        params["verbose"] = -1
    stop = slice(600, None)
    models, _ = fit_direct(family, X.iloc[:500], Y.iloc[:500], X.iloc[stop], Y.iloc[stop],
                           params=params)

    save_estimators(models, family, tmp_path)
    loaded = load_estimators(tmp_path)

    expected = predict_direct(models, X.iloc[-50:]).to_numpy()
    actual = ElectricityDemandForecaster(loaded).predict(None, X.iloc[-50:]).to_numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-5)


def test_loading_rejects_a_different_feature_set(tmp_path, small_dataset):
    import json

    X, Y = small_dataset
    params = {"n_estimators": 5, "verbose": -1, "n_jobs": 1}
    models, _ = fit_direct("LightGBM", X.iloc[:300], Y.iloc[:300], params=params)
    save_estimators(models, "LightGBM", tmp_path)

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["feature_names"] = manifest["feature_names"][:-1]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="different feature set"):
        load_estimators(tmp_path)
