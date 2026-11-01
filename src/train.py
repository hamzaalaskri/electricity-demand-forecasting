"""End-to-end, reproducible training pipeline.

Run from the project root (after `python -m src.weather`, and optionally
`python -m src.tune`):

    python -m src.train

Steps:

1. Load demand and temperature, regularized to one strict hourly grid.
2. Build the shared features and the 24 direct-horizon targets.
3. Split chronologically into train / validation / test, purging the
   rows whose targets would cross into the next partition.
4. Fit LightGBM and XGBoost (one estimator per horizon, tuned parameters
   if available) on the training partition, early-stopping on a held-out
   tail of that same partition.
5. Compare both against naive baselines on the validation partition and
   select the model family with the lowest mean MAE.
6. Refit the selected family on train + validation with the tree counts
   found in step 4 and evaluate it once on the untouched test set, both
   with observed temperatures and with simulated temperature-forecast error.
7. Export the deployable model, write reports and figures, and log the run
   to MLflow (registering a new version and pointing the `champion` alias
   at it).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import warnings
from pathlib import Path

# Silence MLflow's optional-integration noise; must be set before importing it.
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")

import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import (  # noqa: E402
    CHAMPION_ALIAS,
    DATA_PATH,
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_TRACKING_URI,
    EXPERIMENT_NAME,
    FEATURE_NAMES,
    FORECAST_HORIZON,
    HORIZON_FEATURE_NAMES,
    MODEL_DIR,
    MODEL_NAME,
    REPORTS_DIR,
    TUNED_PARAMS_PATH,
    WEATHER_PATH,
    model_params,
)
from src.evaluation import (  # noqa: E402
    baseline_forecasts,
    chronological_split,
    horizon_metrics,
    summarize,
)
from src.modeling import (  # noqa: E402
    FAMILIES,
    ExportBundle,
    export_model,
    fit_direct,
    load_dataset,
    predict_direct,
    with_forecast_error,
)
from src.plots import eda_figures, model_figures  # noqa: E402

logger = logging.getLogger(__name__)

REALISTIC = "with forecast error"
PERFECT = "observed temperature"


# --------------------------------------------------------------------------
# Error analysis
# --------------------------------------------------------------------------


def error_analysis(Y_true: pd.DataFrame, Y_pred: pd.DataFrame) -> dict:
    """Where does the model fail? Pooled over all horizons on one partition."""
    origins = np.repeat(Y_true.index.to_numpy(), FORECAST_HORIZON)
    horizons = np.tile(np.arange(1, FORECAST_HORIZON + 1), len(Y_true))
    long = pd.DataFrame(
        {
            "target_time": pd.DatetimeIndex(origins) + pd.to_timedelta(horizons, "h"),
            "actual": Y_true.to_numpy().ravel(),
            "predicted": Y_pred[Y_true.columns].to_numpy().ravel(),
        }
    )
    long["error"] = long["actual"] - long["predicted"]
    long["abs_error"] = long["error"].abs()

    by_hour = long.groupby(long["target_time"].dt.hour)["abs_error"].mean()
    high = long["actual"] >= long["actual"].median()
    peak_threshold = long["actual"].quantile(0.90)
    peak = long[long["actual"] >= peak_threshold]

    return {
        "mae_by_hour_of_day": {int(k): round(float(v), 2) for k, v in by_hour.items()},
        "worst_hour_of_day": int(by_hour.idxmax()),
        "best_hour_of_day": int(by_hour.idxmin()),
        "mae_high_demand": float(long.loc[high, "abs_error"].mean()),
        "mae_low_demand": float(long.loc[~high, "abs_error"].mean()),
        "peak_threshold_mw": float(peak_threshold),
        "peak_mae": float(peak["abs_error"].mean()),
        "peak_mean_bias_mw": float(peak["error"].mean()),
        "peak_underprediction_share": float((peak["error"] > 0).mean()),
        "p95_abs_error": float(long["abs_error"].quantile(0.95)),
    }


def comparison_table(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Wide per-horizon MAE table with one column per method."""
    table = pd.DataFrame({"horizon": range(1, FORECAST_HORIZON + 1)})
    for name, metrics in results.items():
        table[name] = metrics["mae"].to_numpy()
    return table


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------


def run(
    data_path: Path = DATA_PATH,
    weather_path: Path = WEATHER_PATH,
    model_dir: Path = MODEL_DIR,
    reports_dir: Path = REPORTS_DIR,
    tracking_uri: str | None = DEFAULT_TRACKING_URI,
    candidates: tuple[str, ...] = FAMILIES,
    make_figures: bool = True,
) -> dict:
    data = load_dataset(data_path, weather_path)
    X, Y = data.X, data.Y
    logger.info("Data: %s", data.quality.to_dict())

    split = chronological_split(len(X))

    def ranged(part: slice) -> dict:
        idx = X.index[part]
        return {"rows": len(idx), "start": str(idx.min()), "end": str(idx.max())}

    partitions = {
        "fit": ranged(split.fit),
        "early_stopping": ranged(split.early_stopping),
        "validation": ranged(split.validation),
        "test": ranged(split.test),
    }
    logger.info("Partitions: %s", partitions)

    # --- Model selection on the validation partition ----------------------
    params = {name: model_params(name) for name in candidates}
    validation_results: dict[str, pd.DataFrame] = {}
    iterations: dict[str, dict[int, int]] = {}
    candidate_models: dict[str, dict] = {}
    X_val, Y_val = X.iloc[split.validation], Y.iloc[split.validation]
    for name in candidates:
        logger.info("Fitting %s candidate (24 horizons)", name)
        candidate_models[name], iterations[name] = fit_direct(
            name,
            X.iloc[split.fit], Y.iloc[split.fit],
            X.iloc[split.early_stopping], Y.iloc[split.early_stopping],
            params=params[name],
        )
        validation_results[name] = horizon_metrics(
            Y_val, predict_direct(candidate_models[name], X_val)
        )

    for name, forecast in baseline_forecasts(data.series, X_val.index).items():
        validation_results[name] = horizon_metrics(Y_val, forecast)

    selected = min(candidates, key=lambda n: validation_results[n]["mae"].mean())
    logger.info("Selected %s on validation mean MAE", selected)

    # --- Final model: refit on train + validation, test once --------------
    logger.info("Refitting %s on train + validation", selected)
    final_models, _ = fit_direct(
        selected,
        X.iloc[split.train_validation], Y.iloc[split.train_validation],
        n_estimators=iterations[selected],
        params=params[selected],
    )

    X_test, Y_test = X.iloc[split.test], Y.iloc[split.test]
    predictions_perfect = predict_direct(final_models, X_test)
    predictions = predict_direct(final_models, with_forecast_error(X_test, seed=0))

    test_results = {selected: horizon_metrics(Y_test, predictions)}
    for name, forecast in baseline_forecasts(data.series, Y_test.index).items():
        test_results[name] = horizon_metrics(Y_test, forecast)
    test_perfect = horizon_metrics(Y_test, predictions_perfect)

    best_baseline = min(
        (n for n in test_results if n != selected),
        key=lambda n: test_results[n]["mae"].mean(),
    )
    test_summary = summarize(test_results[selected])
    improvement = 1 - test_summary["mean_mae"] / test_results[best_baseline]["mae"].mean()

    report = {
        "data_quality": data.quality.to_dict(),
        "partitions": partitions,
        "features": {"origin": FEATURE_NAMES, "per_horizon": HORIZON_FEATURE_NAMES},
        "candidates": list(candidates),
        "params": params,
        "params_source": "tuned" if TUNED_PARAMS_PATH.exists() else "default",
        "selected_model": selected,
        "trees_per_horizon": iterations[selected],
        "validation": {n: summarize(m) for n, m in validation_results.items()},
        "test": {n: summarize(m) for n, m in test_results.items()},
        "test_temperature_scenarios": {
            REALISTIC: test_summary,
            PERFECT: summarize(test_perfect),
        },
        "test_best_baseline": best_baseline,
        "test_improvement_over_best_baseline": float(improvement),
        "test_error_analysis": error_analysis(Y_test, predictions),
        "trained_through": partitions["validation"]["end"],
    }

    # --- Reports ------------------------------------------------------------
    reports_dir.mkdir(parents=True, exist_ok=True)
    comparison_table(validation_results).to_csv(
        reports_dir / "validation_mae_by_horizon.csv", index=False, float_format="%.2f"
    )
    comparison_table({**test_results, f"{selected} ({PERFECT})": test_perfect}).to_csv(
        reports_dir / "test_mae_by_horizon.csv", index=False, float_format="%.2f"
    )
    test_results[selected].to_csv(
        reports_dir / "test_metrics_selected_model.csv", index=False, float_format="%.2f"
    )
    if make_figures:
        figures_dir = reports_dir / "figures"
        eda_figures(data.series, figures_dir, temperature=data.temperature)
        model_figures(
            validation_results, test_results, candidates, selected,
            Y_test, predictions, final_models, HORIZON_FEATURE_NAMES, figures_dir,
        )

    # --- Deployable model ---------------------------------------------------
    input_example = X_test.iloc[:1]
    metadata = {
        "forecast_horizon": FORECAST_HORIZON,
        "trained_through": report["trained_through"],
        "test_mean_mae_mw": round(test_summary["mean_mae"], 2),
        "test_mean_mape_pct": round(test_summary["mean_mape"], 3),
    }
    export_model(final_models, selected, model_dir, input_example, metadata)
    logger.info("Saved deployable model to %s", model_dir)

    if tracking_uri:
        report["mlflow"] = _log_to_mlflow(
            tracking_uri, report, iterations, validation_results, test_results,
            test_perfect, reports_dir, candidate_models, final_models, input_example,
            metadata,
        )

    (reports_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    return report


def _log_horizon_curves(metrics: pd.DataFrame, prefix: str) -> None:
    for row in metrics.itertuples():
        mlflow.log_metric(f"{prefix}.mae_by_horizon", row.mae, step=row.horizon)
        mlflow.log_metric(f"{prefix}.rmse_by_horizon", row.rmse, step=row.horizon)


def _metric_key(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")


def setup_mlflow(tracking_uri: str) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=_artifact_root(tracking_uri))
    mlflow.set_experiment(EXPERIMENT_NAME)


def _artifact_root(tracking_uri: str) -> str | None:
    """Keep artifacts next to a local sqlite store instead of the CWD."""
    if tracking_uri == DEFAULT_TRACKING_URI:
        return DEFAULT_ARTIFACT_ROOT
    if tracking_uri.startswith("sqlite:///"):
        return (Path(tracking_uri.removeprefix("sqlite:///")).parent / "mlartifacts").as_uri()
    return None


def _log_to_mlflow(tracking_uri, report, iterations, validation_results, test_results,
                   test_perfect, reports_dir, candidate_models, final_models,
                   input_example, metadata) -> dict:
    """One parent run per training, with a child run per candidate and one
    for the final model:

        training_pipeline
        ├── LightGBM_24h_candidate   params, tree counts, validation curves, 24-horizon model
        ├── XGBoost_24h_candidate    (same)
        └── <selected>_24h_final     test metrics, reports, registered model (alias: champion)
    """
    setup_mlflow(tracking_uri)
    selected = report["selected_model"]

    with mlflow.start_run(run_name="training_pipeline") as parent:
        mlflow.set_tags({
            "selected_model": selected,
            "forecasting_strategy": "direct multi-horizon",
            "params_source": report["params_source"],
        })
        mlflow.log_params({
            "forecast_horizon": FORECAST_HORIZON,
            "n_origin_features": len(FEATURE_NAMES),
            "features": ",".join(HORIZON_FEATURE_NAMES),
            "candidates": ",".join(report["candidates"]),
            "selected_model": selected,
            **{f"data.{k}": v for k, v in report["data_quality"].items()},
            **{f"split.{p}.{k}": v for p, info in report["partitions"].items()
               for k, v in info.items()},
        })
        for stage, results in (("val", validation_results), ("test", test_results)):
            for name, metrics in results.items():
                for k, v in summarize(metrics).items():
                    mlflow.log_metric(f"{stage}.{_metric_key(name)}.{k}", v)

        for name, models in candidate_models.items():
            with mlflow.start_run(run_name=f"{name}_{FORECAST_HORIZON}h_candidate", nested=True):
                mlflow.set_tags({"role": "candidate", "model_family": name,
                                 "selected": str(name == selected)})
                mlflow.log_params({f"model.{k}": v for k, v in report["params"][name].items()})
                mlflow.log_params({
                    "fit_rows": report["partitions"]["fit"]["rows"],
                    "early_stopping_rows": report["partitions"]["early_stopping"]["rows"],
                })
                for k, v in summarize(validation_results[name]).items():
                    mlflow.log_metric(f"val.{k}", v)
                _log_horizon_curves(validation_results[name], "val")
                for h, n_trees in iterations[name].items():
                    mlflow.log_metric("trees_by_horizon", n_trees, step=h)
                with ExportBundle(models, name, input_example) as bundle:
                    mlflow.pyfunc.log_model(name="model", **bundle.kwargs)

        with mlflow.start_run(run_name=f"{selected}_{FORECAST_HORIZON}h_final", nested=True) as final:
            mlflow.set_tags({"role": "final", "model_family": selected})
            mlflow.log_params({f"model.{k}": v for k, v in report["params"][selected].items()})
            mlflow.log_params({"trained_on": "train+validation",
                               "trained_through": report["trained_through"]})
            for k, v in summarize(test_results[selected]).items():
                mlflow.log_metric(f"test.{k}", v)
            for k, v in summarize(test_perfect).items():
                mlflow.log_metric(f"test_observed_temperature.{k}", v)
            mlflow.log_metric("test.improvement_over_best_baseline",
                              report["test_improvement_over_best_baseline"])
            _log_horizon_curves(test_results[selected], "test")
            mlflow.log_artifacts(str(reports_dir), artifact_path="reports")
            with ExportBundle(final_models, selected, input_example, metadata) as bundle:
                info = mlflow.pyfunc.log_model(
                    name="model", registered_model_name=MODEL_NAME, **bundle.kwargs
                )

    version = str(info.registered_model_version)
    mlflow.MlflowClient().set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, version)
    logger.info("Logged MLflow run %s, registered %s version %s (@%s)",
                parent.info.run_id, MODEL_NAME, version, CHAMPION_ALIAS)
    return {
        "parent_run_id": parent.info.run_id,
        "final_run_id": final.info.run_id,
        "registered_model": MODEL_NAME,
        "registered_version": version,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--weather", type=Path, default=WEATHER_PATH)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI,
                        help="MLflow tracking URI (default: local sqlite).")
    parser.add_argument("--no-mlflow", action="store_true",
                        help="Skip MLflow logging and registration.")
    parser.add_argument("--skip-xgboost", action="store_true",
                        help="Only train LightGBM (faster).")
    args = parser.parse_args(argv)

    configure_logging()
    candidates = ("LightGBM",) if args.skip_xgboost else FAMILIES
    report = run(
        data_path=args.data,
        weather_path=args.weather,
        model_dir=args.model_dir,
        reports_dir=args.reports_dir,
        tracking_uri=None if args.no_mlflow else args.tracking_uri,
        candidates=candidates,
    )
    selected = report["selected_model"]
    logger.info(
        "Done. %s test mean MAE %.1f MW (%.2f%% MAPE) with forecast temperatures, "
        "%.1f%% better than %s.",
        selected,
        report["test"][selected]["mean_mae"],
        report["test"][selected]["mean_mape"],
        100 * report["test_improvement_over_best_baseline"],
        report["test_best_baseline"],
    )


def configure_logging() -> None:
    warnings.filterwarnings("ignore", message=".*Integer columns in Python.*")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("mlflow").setLevel(logging.ERROR)


if __name__ == "__main__":
    main()
