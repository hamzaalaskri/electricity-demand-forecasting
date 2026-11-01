"""Chronological splitting, baselines, and multi-horizon metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import (
    EARLY_STOPPING_FRACTION,
    FORECAST_HORIZON,
    TRAIN_FRACTION,
    VALIDATION_FRACTION,
)


@dataclass(frozen=True)
class Split:
    """Positional row ranges of a supervised dataset.

    Every partition except the last ends ``horizon`` rows early ("purging"),
    so no target in one partition is a value from the next one.
    """

    fit: slice
    early_stopping: slice
    train: slice
    validation: slice
    train_validation: slice
    test: slice


def chronological_split(
    n_rows: int,
    horizon: int = FORECAST_HORIZON,
    train_fraction: float = TRAIN_FRACTION,
    validation_fraction: float = VALIDATION_FRACTION,
    early_stopping_fraction: float = EARLY_STOPPING_FRACTION,
) -> Split:
    train_end = int(n_rows * train_fraction)
    validation_end = train_end + int(n_rows * validation_fraction)
    early_stopping_start = int(train_end * (1 - early_stopping_fraction))

    if early_stopping_start - horizon <= 0 or validation_end >= n_rows:
        raise ValueError("Not enough rows for the requested split.")

    return Split(
        fit=slice(0, early_stopping_start - horizon),
        early_stopping=slice(early_stopping_start, train_end - horizon),
        train=slice(0, train_end - horizon),
        validation=slice(train_end, validation_end - horizon),
        train_validation=slice(0, validation_end - horizon),
        test=slice(validation_end, n_rows),
    )


def baseline_forecasts(
    series: pd.Series,
    index: pd.DatetimeIndex,
    horizon: int = FORECAST_HORIZON,
) -> dict[str, pd.DataFrame]:
    """Forecasts issued at each origin in ``index`` by three reference methods.

    * Naive: every horizon repeats the latest observed value.
    * Seasonal naive 24h: the value at the same hour on the previous day.
    * Seasonal naive 168h: the value at the same hour one week earlier.

    All three only use values observed at or before the forecast origin.
    """
    columns = [f"target_{h}" for h in range(1, horizon + 1)]
    latest = series.reindex(index).to_numpy()

    def seasonal(period: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                f"target_{h}": series.shift(period - h).reindex(index).to_numpy()
                for h in range(1, horizon + 1)
            },
            index=index,
        )

    return {
        "Naive": pd.DataFrame(
            np.repeat(latest[:, None], horizon, axis=1), index=index, columns=columns
        ),
        "Seasonal naive (24h)": seasonal(24),
        "Seasonal naive (168h)": seasonal(168),
    }


def horizon_metrics(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> pd.DataFrame:
    """MAE, RMSE, and MAPE for each forecast horizon."""
    rows = []
    for h, column in enumerate(y_true.columns, start=1):
        actual = y_true[column].to_numpy(dtype=np.float64)
        predicted = np.asarray(y_pred[column], dtype=np.float64)
        error = actual - predicted
        rows.append(
            {
                "horizon": h,
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "mape": float(np.mean(np.abs(error) / actual) * 100),
            }
        )
    return pd.DataFrame(rows)


def summarize(metrics: pd.DataFrame) -> dict[str, float]:
    """Average over horizons, plus the first and last horizon."""
    return {
        "mean_mae": float(metrics["mae"].mean()),
        "mean_rmse": float(metrics["rmse"].mean()),
        "mean_mape": float(metrics["mape"].mean()),
        "mae_h1": float(metrics["mae"].iloc[0]),
        "mae_h24": float(metrics["mae"].iloc[-1]),
    }
