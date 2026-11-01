"""Figures for the EDA and the model reports, written as PNG files."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import FORECAST_HORIZON  # noqa: E402

DPI = 150
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Exploratory analysis
# --------------------------------------------------------------------------


def eda_figures(series: pd.Series, out_dir: Path,
                temperature: pd.Series | None = None) -> list[Path]:
    """Series overview, seasonal profiles, peaks, autocorrelation, and the
    demand–temperature relationship."""
    from statsmodels.tsa.stattools import acf

    paths = []

    if temperature is not None:
        joined = pd.DataFrame({"demand": series, "temperature": temperature}).dropna()
        sample = joined.sample(min(len(joined), 20_000), random_state=0)
        bins = pd.cut(joined["temperature"], np.arange(-20, 42, 2))
        profile = joined.groupby(bins, observed=True)["demand"].mean()
        centres = [interval.mid for interval in profile.index]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(sample["temperature"], sample["demand"], s=2, alpha=0.15, label="Hourly readings")
        ax.plot(centres, profile.values, color="tab:red", linewidth=2.5, label="Mean per 2 °C bin")
        ax.set_xlabel("Temperature (°C, mean of 4 load centres)")
        ax.set_ylabel("Demand (MW)")
        ax.set_title("Demand vs temperature: heating on the left, cooling on the right")
        ax.grid(alpha=0.3)
        ax.legend()
        paths.append(_save(fig, out_dir / "eda_demand_vs_temperature.png"))

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    axes[0].plot(series.index, series.values, linewidth=0.3)
    axes[0].set_title("Hourly demand, 2002–2018")
    daily = series.resample("D").mean()
    weekly = series.resample("W").mean()
    axes[1].plot(daily.index, daily.values, linewidth=0.5, alpha=0.6, label="Daily mean")
    axes[1].plot(weekly.index, weekly.values, linewidth=1.2, label="Weekly mean")
    axes[1].set_title("Daily and weekly average demand")
    axes[1].legend()
    for ax in axes:
        ax.set_ylabel("Demand (MW)")
        ax.grid(alpha=0.3)
    paths.append(_save(fig, out_dir / "eda_time_series.png"))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    by_hour = series.groupby(series.index.hour).mean()
    by_day = series.groupby(series.index.dayofweek).mean()
    by_month = series.groupby(series.index.month).mean()
    axes[0].plot(by_hour.index, by_hour.values, marker="o", markersize=3)
    axes[0].set_xticks(range(0, 24, 3))
    axes[0].set_title("By hour of day")
    axes[1].bar([DAY_NAMES[d] for d in by_day.index], by_day.values)
    axes[1].set_title("By day of week")
    axes[2].bar([MONTH_NAMES[m - 1] for m in by_month.index], by_month.values)
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].set_title("By month")
    for ax in axes:
        ax.set_ylabel("Mean demand (MW)")
        ax.grid(alpha=0.3, axis="y")
    axes[1].set_ylim(by_day.min() * 0.9, by_day.max() * 1.03)
    axes[2].set_ylim(by_month.min() * 0.9, by_month.max() * 1.03)
    paths.append(_save(fig, out_dir / "eda_seasonal_profiles.png"))

    threshold = series.quantile(0.99)
    peaks = series[series >= threshold]
    counts = peaks.groupby(peaks.index.month).size().reindex(range(1, 13), fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(MONTH_NAMES, counts.values)
    ax.set_title(f"Peak hours (top 1 %, ≥ {threshold:,.0f} MW) by month")
    ax.set_ylabel("Number of hours")
    ax.grid(alpha=0.3, axis="y")
    paths.append(_save(fig, out_dir / "eda_peaks_by_month.png"))

    lags = 168
    values = acf(series.to_numpy(), nlags=lags, fft=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.vlines(range(lags + 1), 0, values, linewidth=1)
    for marker in (24, 48, 168):
        ax.axvline(marker, color="tab:red", linestyle="--", alpha=0.5)
        ax.text(marker, 1.02, f"{marker}h", ha="center", color="tab:red", fontsize=9)
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Autocorrelation of hourly demand (up to one week)", pad=18)
    ax.set_ylim(min(0, values.min()) - 0.05, 1.08)
    ax.grid(alpha=0.3)
    paths.append(_save(fig, out_dir / "eda_autocorrelation.png"))

    return paths


# --------------------------------------------------------------------------
# Model results
# --------------------------------------------------------------------------


def _by_horizon(results: dict[str, pd.DataFrame], metric: str, title: str,
                highlight: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, metrics in results.items():
        style = {"linewidth": 2.5} if name == highlight else {"linestyle": "--"}
        ax.plot(metrics["horizon"], metrics[metric], marker="o", markersize=3,
                label=name, **style)
    ax.set_xlabel("Forecast horizon (hours ahead)")
    ax.set_ylabel(f"{metric.upper()} (MW)")
    ax.set_title(title)
    ax.set_xticks(range(1, FORECAST_HORIZON + 1, 2))
    ax.grid(alpha=0.3)
    ax.legend()
    return _save(fig, path)


def model_figures(
    validation_results: dict[str, pd.DataFrame],
    test_results: dict[str, pd.DataFrame],
    candidates: tuple[str, ...],
    selected: str,
    Y_test: pd.DataFrame,
    Y_pred: pd.DataFrame,
    final_models: dict,
    feature_names: list[str],
    out_dir: Path,
) -> list[Path]:
    paths = [
        _by_horizon(test_results, "mae", "Test-set MAE by forecast horizon",
                    selected, out_dir / "mae_by_horizon.png"),
        _by_horizon(test_results, "rmse", "Test-set RMSE by forecast horizon",
                    selected, out_dir / "rmse_by_horizon.png"),
        _by_horizon({n: validation_results[n] for n in candidates},
                    "mae", "Validation MAE by horizon: model comparison",
                    selected, out_dir / "validation_model_comparison.png"),
    ]

    # One-week window of day-ahead (h=24) forecasts from the test partition.
    window = slice(len(Y_test) // 2, len(Y_test) // 2 + 168)
    target_time = Y_test.index[window] + pd.Timedelta(hours=FORECAST_HORIZON)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(target_time, Y_test["target_24"].iloc[window], label="Actual")
    ax.plot(target_time, Y_pred["target_24"].iloc[window],
            label=f"{selected} forecast (24h ahead)")
    ax.set_ylabel("Demand (MW)")
    ax.set_title("Actual vs day-ahead forecast, one test-set week")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    paths.append(_save(fig, out_dir / "actual_vs_forecast.png"))

    # Pooled absolute errors over all horizons.
    origins = np.repeat(Y_test.index.to_numpy(), FORECAST_HORIZON)
    horizons = np.tile(np.arange(1, FORECAST_HORIZON + 1), len(Y_test))
    target_hour = (pd.DatetimeIndex(origins) + pd.to_timedelta(horizons, "h")).hour
    actual = Y_test.to_numpy().ravel()
    abs_error = np.abs(actual - Y_pred[Y_test.columns].to_numpy().ravel())

    by_hour = pd.Series(abs_error).groupby(target_hour).mean()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(by_hour.index, by_hour.values)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour of day (forecast target)")
    ax.set_ylabel("MAE (MW)")
    ax.set_title("Test-set error by hour of day (all horizons)")
    ax.grid(alpha=0.3, axis="y")
    paths.append(_save(fig, out_dir / "error_by_hour_of_day.png"))

    deciles = pd.qcut(actual, 10, labels=[f"D{i}" for i in range(1, 11)])
    by_decile = pd.Series(abs_error).groupby(deciles, observed=True).mean()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(by_decile.index.astype(str), by_decile.values)
    ax.set_xlabel("Actual demand decile (D1 = lowest, D10 = peak)")
    ax.set_ylabel("MAE (MW)")
    ax.set_title("Test-set error grows with demand level")
    ax.grid(alpha=0.3, axis="y")
    paths.append(_save(fig, out_dir / "error_by_demand_level.png"))

    importance = pd.DataFrame(
        {
            f"h={h}": _normalized_gain(final_models[h], feature_names)
            for h in (1, 12, FORECAST_HORIZON)
        }
    ).sort_values(f"h={FORECAST_HORIZON}")
    fig, ax = plt.subplots(figsize=(9, 6.5))
    importance.plot.barh(ax=ax, width=0.8)
    ax.set_xlabel("Share of total gain")
    ax.set_title("Feature importance at short, medium, and day-ahead horizons")
    ax.grid(alpha=0.3, axis="x")
    paths.append(_save(fig, out_dir / "feature_importance.png"))

    return paths


def _normalized_gain(model, feature_names: list[str]) -> pd.Series:
    if hasattr(model, "booster_"):  # LightGBM
        gain = model.booster_.feature_importance(importance_type="gain")
        series = pd.Series(gain, index=feature_names, dtype=float)
    else:  # XGBoost
        scores = model.get_booster().get_score(importance_type="total_gain")
        series = pd.Series(scores, dtype=float).reindex(feature_names, fill_value=0.0)
    return series / series.sum()
