"""Rolling-origin (expanding-window) backtest across calendar years.

    python -m src.backtest

For each test year Y (2014–2018), the selected model family is retrained
from scratch on all data before 1 January of Y (purged), early-stopped on
the last 10 % of that history, and evaluated on every forecast origin in
Y. This measures how stable accuracy is across years with different
weather, rather than relying on a single hold-out window.
"""

from __future__ import annotations

import argparse
import json
import logging
import os

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import (  # noqa: E402
    DEFAULT_TRACKING_URI,
    EARLY_STOPPING_FRACTION,
    FORECAST_HORIZON,
    REPORTS_DIR,
)
from src.evaluation import baseline_forecasts, horizon_metrics, summarize  # noqa: E402
from src.modeling import (  # noqa: E402
    fit_direct,
    load_dataset,
    predict_direct,
    with_forecast_error,
)

logger = logging.getLogger(__name__)

FOLD_YEARS = (2014, 2015, 2016, 2017, 2018)
BASELINE = "Seasonal naive (24h)"


def fold_rows(index: pd.DatetimeIndex, year: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Positional rows for fit, early stopping, and test of one fold."""
    positions = np.arange(len(index))
    test = positions[index.year == year]
    history_end = test[0] - FORECAST_HORIZON          # purge: targets stay before Y
    stop_start = int(history_end * (1 - EARLY_STOPPING_FRACTION))
    fit = positions[: stop_start - FORECAST_HORIZON]
    stop = positions[stop_start:history_end]
    return fit, stop, test


def run(family: str | None = None, years=FOLD_YEARS,
        tracking_uri: str | None = DEFAULT_TRACKING_URI, reports_dir=REPORTS_DIR) -> dict:
    family = family or _selected_family(reports_dir)
    data = load_dataset()
    X, Y = data.X, data.Y

    folds = []
    for year in years:
        fit, stop, test = fold_rows(X.index, year)
        logger.info("Fold %d: fit %d rows, early stopping %d, test %d", year, len(fit), len(stop), len(test))
        models, _ = fit_direct(family, X.iloc[fit], Y.iloc[fit], X.iloc[stop], Y.iloc[stop])

        X_test, Y_test = X.iloc[test], Y.iloc[test]
        realistic = summarize(horizon_metrics(
            Y_test, predict_direct(models, with_forecast_error(X_test, seed=year))))
        observed = summarize(horizon_metrics(Y_test, predict_direct(models, X_test)))
        baseline = summarize(horizon_metrics(
            Y_test, baseline_forecasts(data.series, X_test.index)[BASELINE]))

        folds.append({
            "year": year,
            "test_start": str(X_test.index.min()),
            "test_end": str(X_test.index.max()),
            "train_rows": int(len(fit) + len(stop)),
            "test_rows": int(len(test)),
            "mae": realistic["mean_mae"],
            "rmse": realistic["mean_rmse"],
            "mape": realistic["mean_mape"],
            "mae_observed_temperature": observed["mean_mae"],
            "baseline_mae": baseline["mean_mae"],
            "improvement_over_baseline": 1 - realistic["mean_mae"] / baseline["mean_mae"],
        })
        logger.info("  %d: MAE %.1f MW (%.2f%%), baseline %.1f", year,
                    realistic["mean_mae"], realistic["mean_mape"], baseline["mean_mae"])

    table = pd.DataFrame(folds)
    summary = {
        "family": family,
        "baseline": BASELINE,
        "temperature": "simulated forecast error (see src/modeling.with_forecast_error)",
        "folds": folds,
        "mae_mean": float(table["mae"].mean()),
        "mae_std": float(table["mae"].std(ddof=1)),
        "mape_mean": float(table["mape"].mean()),
        "mape_std": float(table["mape"].std(ddof=1)),
        "improvement_over_baseline_min": float(table["improvement_over_baseline"].min()),
        "improvement_over_baseline_mean": float(table["improvement_over_baseline"].mean()),
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "backtest.json").write_text(json.dumps(summary, indent=2))
    table.to_csv(reports_dir / "backtest_folds.csv", index=False, float_format="%.3f")
    figure = _figure(table, family, reports_dir / "figures" / "backtest_by_year.png")

    if tracking_uri:
        import mlflow

        from src.train import setup_mlflow

        setup_mlflow(tracking_uri)
        with mlflow.start_run(run_name="rolling_origin_backtest"):
            mlflow.set_tags({"model_family": family, "role": "backtest"})
            mlflow.log_param("fold_years", ",".join(map(str, years)))
            for row in folds:
                for key in ("mae", "mape", "baseline_mae", "improvement_over_baseline"):
                    mlflow.log_metric(f"fold.{key}", row[key], step=row["year"])
            for key in ("mae_mean", "mae_std", "mape_mean", "mape_std",
                        "improvement_over_baseline_min", "improvement_over_baseline_mean"):
                mlflow.log_metric(key, summary[key])
            mlflow.log_artifact(str(reports_dir / "backtest.json"))
            mlflow.log_artifact(str(figure))
    return summary


def _selected_family(reports_dir) -> str:
    metrics = reports_dir / "metrics.json"
    if metrics.exists():
        return json.loads(metrics.read_text())["selected_model"]
    return "LightGBM"


def _figure(table: pd.DataFrame, family: str, path):
    from src.plots import _save, plt

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(table))
    ax.bar(x - 0.2, table["mae"], width=0.4, label=f"{family} (forecast temperatures)")
    ax.bar(x + 0.2, table["baseline_mae"], width=0.4, label=BASELINE, color="tab:gray")
    for i, row in table.iterrows():
        ax.text(i - 0.2, row["mae"] + 30, f"{row['mape']:.1f}%", ha="center", fontsize=9)
    labels = [str(y) if y < 2018 else "2018 (Jan–Aug)" for y in table["year"]]
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean MAE over 24 horizons (MW)")
    ax.set_title("Rolling-origin backtest: retrained before each test year")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    return _save(fig, path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Rolling-origin backtest by year.")
    parser.add_argument("--family", choices=["LightGBM", "XGBoost"])
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args(argv)

    from src.train import configure_logging

    configure_logging()
    summary = run(args.family, tracking_uri=None if args.no_mlflow else DEFAULT_TRACKING_URI)
    logger.info("Backtest MAE %.1f ± %.1f MW, MAPE %.2f ± %.2f %%",
                summary["mae_mean"], summary["mae_std"], summary["mape_mean"], summary["mape_std"])


if __name__ == "__main__":
    main()
