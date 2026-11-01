"""Shared building blocks: dataset assembly, direct multi-horizon fitting,
prediction, temperature-forecast error simulation, and model export.

Used by training, tuning, backtesting, and the production simulation so
that every stage fits and evaluates models in exactly the same way.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature

from src.config import (
    DATA_PATH,
    EARLY_STOPPING_ROUNDS,
    FORECAST_HORIZON,
    MODEL_INPUT_NAMES,
    PROJECT_DIR,
    TARGET_NAMES,
    TEMPERATURE_FORECAST_NAMES,
    WEATHER_PATH,
    model_params,
)
from src.data import DataQualityReport, load_series
from src.features import build_features, build_targets, horizon_matrix
from src.model import ElectricityDemandForecaster, save_estimators
from src.weather import load_temperature

logger = logging.getLogger(__name__)

FAMILIES = ("LightGBM", "XGBoost")


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


@dataclass
class Dataset:
    series: pd.Series             # regular hourly demand
    temperature: pd.Series        # regular hourly temperature (°C)
    X: pd.DataFrame               # model input, one row per forecast origin
    Y: pd.DataFrame               # target_1 .. target_24
    quality: DataQualityReport


def load_dataset(data_path: Path = DATA_PATH, weather_path: Path = WEATHER_PATH) -> Dataset:
    series, quality = load_series(data_path)
    temperature = load_temperature(weather_path)

    first, last = series.index.min(), series.index.max() + pd.Timedelta(hours=FORECAST_HORIZON)
    if temperature.index.min() > first or temperature.index.max() < last:
        raise ValueError("Weather data must cover the full demand period plus one day.")

    frame = build_features(series, temperature).join(build_targets(series)).dropna()
    return Dataset(series, temperature, frame[MODEL_INPUT_NAMES], frame[TARGET_NAMES], quality)


# --------------------------------------------------------------------------
# Fitting and prediction
# --------------------------------------------------------------------------


def make_estimator(family: str, params: dict, early_stopping: bool):
    if family == "LightGBM":
        import lightgbm as lgb

        return lgb.LGBMRegressor(**params)
    if family == "XGBoost":
        from xgboost import XGBRegressor

        extra = {"eval_metric": "mae"}
        if early_stopping:
            extra["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
        return XGBRegressor(**params, **extra)
    raise ValueError(f"Unknown model family: {family}")


def fit_one(family: str, params: dict, X_fit, y_fit, X_stop=None, y_stop=None):
    """Fit one estimator. Returns the model and the number of trees it uses."""
    early_stopping = X_stop is not None
    model = make_estimator(family, params, early_stopping)

    if family == "LightGBM":
        import lightgbm as lgb

        if early_stopping:
            model.fit(
                X_fit, y_fit,
                eval_X=(X_stop,), eval_y=(y_stop,), eval_metric="l1",
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
            )
            return model, int(model.best_iteration_)
        model.fit(X_fit, y_fit)
        return model, int(params["n_estimators"])

    if early_stopping:
        model.fit(X_fit, y_fit, eval_set=[(X_stop, y_stop)], verbose=False)
        return model, int(model.best_iteration) + 1
    model.fit(X_fit, y_fit, verbose=False)
    return model, int(params["n_estimators"])


def fit_direct(
    family: str,
    X_fit: pd.DataFrame,
    Y_fit: pd.DataFrame,
    X_stop: pd.DataFrame | None = None,
    Y_stop: pd.DataFrame | None = None,
    n_estimators: dict[int, int] | None = None,
    params: dict | None = None,
    horizons: tuple[int, ...] | None = None,
) -> tuple[dict[int, object], dict[int, int]]:
    """Fit one estimator per horizon.

    With ``X_stop``/``Y_stop`` each estimator early-stops on that slice;
    with ``n_estimators`` each uses a fixed tree count (e.g. found earlier).
    """
    base = dict(params or model_params(family))
    models, trees = {}, {}
    for h in horizons or range(1, FORECAST_HORIZON + 1):
        horizon_params = dict(base)
        if n_estimators is not None:
            horizon_params["n_estimators"] = n_estimators[h]
        target = f"target_{h}"
        models[h], trees[h] = fit_one(
            family,
            horizon_params,
            horizon_matrix(X_fit, h),
            Y_fit[target],
            None if X_stop is None else horizon_matrix(X_stop, h),
            None if Y_stop is None else Y_stop[target],
        )
        logger.debug("  %s horizon %2d: %4d trees", family, h, trees[h])
    return models, trees


def predict_direct(models: dict[int, object], X: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {f"target_{h}": models[h].predict(horizon_matrix(X, h)) for h in sorted(models)},
        index=X.index,
    )


# --------------------------------------------------------------------------
# Realistic temperature forecasts
# --------------------------------------------------------------------------


def forecast_error_sigma(h: int) -> float:
    """Standard deviation (°C) of a temperature forecast ``h`` hours ahead.

    Numerical weather prediction 2 m temperature errors grow from roughly
    0.5 °C in the first hours to about 2 °C one day ahead.
    """
    return 0.5 + 1.5 * h / FORECAST_HORIZON


def with_forecast_error(X: pd.DataFrame, seed: int = 0, correlation: float = 0.7) -> pd.DataFrame:
    """Replace the perfect temperature forecasts with simulated forecasts.

    Errors share a common per-origin component (a forecast that runs warm
    tends to run warm all day) plus an independent part per horizon.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    common = rng.standard_normal(n)
    noisy = X.copy()
    for h, column in enumerate(TEMPERATURE_FORECAST_NAMES, start=1):
        z = correlation * common + np.sqrt(1 - correlation**2) * rng.standard_normal(n)
        noisy[column] = X[column].to_numpy() + forecast_error_sigma(h) * z
    return noisy


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------


def pip_requirements(family: str) -> list[str]:
    packages = ["mlflow-skinny", "pandas", "numpy",
                "lightgbm" if family == "LightGBM" else "xgboost"]
    return [f"{p}=={version(p)}" for p in packages]


class ExportBundle:
    """Estimators written to a temporary directory plus the kwargs that
    ``mlflow.pyfunc.save_model`` / ``log_model`` need. Use as a context manager."""

    def __init__(self, models: dict[int, object], family: str,
                 input_example: pd.DataFrame, metadata: dict | None = None):
        self._tmp = Path(tempfile.mkdtemp(prefix="estimators_"))
        estimators = save_estimators(models, family, self._tmp / "estimators")
        output = ElectricityDemandForecaster(models).predict(None, input_example)
        # The wrapper is passed *empty*: MLflow pickles only a reference to
        # the class in src/model.py (bundled via code_paths), and the
        # estimators are read from the native-format artifacts in
        # load_context. MLmodel then holds only relative paths, so a model
        # exported on Windows loads on Linux and vice versa.
        self.kwargs = {
            "python_model": ElectricityDemandForecaster(),
            "artifacts": {"estimators": str(estimators)},
            "code_paths": [str(PROJECT_DIR / "src")],
            "signature": infer_signature(input_example, output),
            "input_example": input_example,
            "pip_requirements": pip_requirements(family),
            "metadata": {"model_family": family, **(metadata or {})},
        }

    def __enter__(self) -> ExportBundle:
        return self

    def __exit__(self, *exc) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)


def export_model(models, family, path: Path, input_example, metadata=None) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with ExportBundle(models, family, input_example, metadata) as bundle:
        mlflow.pyfunc.save_model(path=str(path), **bundle.kwargs)
    return path
