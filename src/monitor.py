"""Production simulation: monitoring, retraining, promotion gate, rollback.

    python -m src.monitor

The test period (Feb 2016 – Aug 2018) is replayed one day at a time, as if
the system were live. Every evening at 23:00 the current *champion* model
issues a 24-hour forecast for the next calendar day using realistic
(noisy) temperature forecasts. When the day's actual demand arrives:

1. **Data quality**: incoming readings are checked (missing, non-positive,
   flat-lined, implausible jumps). A failing input blocks the forecast
   instead of producing a silently wrong one.
2. **Performance**: the 28-day MAE is tracked relative to the 24 h
   seasonal-naive forecast on the same days. The ratio (model MAE /
   baseline MAE) removes the seasonal swing in absolute error, so a hot
   summer is not mistaken for model decay. An alert fires when the ratio
   stays 20 % above its reference level for 7 consecutive days.
3. **Drift**: the population stability index (PSI) of temperature and
   demand over the last 28 days is compared with the *same calendar
   months* of the champion's training data. Drift is reported, but on its
   own never replaces a model.
4. **Retraining**: triggered by a performance alert or a 180-day schedule
   (at most once every 60 days). The candidate is trained on all data up
   to 28 days before the decision date.
5. **Promotion gate**: champion and candidate are both scored on those
   last 28 days (unseen by the candidate). The candidate is deployed only
   if it is at least 1 % better.
6. **Rollback**: after any deployment, the previous champion keeps
   running in shadow mode for 14 days. If the new champion is more than
   5 % worse over at least 7 days, the deployment is rolled back.

Two clearly labelled faults are injected (disable with ``--no-faults``)
so the safeguards can be seen working: a three-day meter outage, and a
manual "hotfix" deployment that bypasses the gate with a model trained on
mis-calibrated data. Models are versioned in the MLflow registry under
``ElectricityDemandForecaster-simulation`` with ``champion`` /
``previous`` aliases, separate from the production model's lineage.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import (  # noqa: E402
    CHAMPION_ALIAS,
    DEFAULT_TRACKING_URI,
    DRIFT_PSI_THRESHOLD,
    FORECAST_HORIZON,
    MIN_DAYS_BETWEEN_RETRAINS,
    MODEL_DIR,
    MODEL_NAME,
    MONITOR_WINDOW_DAYS,
    PERFORMANCE_ALERT_DAYS,
    PERFORMANCE_ALERT_RATIO,
    PROMOTION_EVAL_DAYS,
    PROMOTION_MIN_IMPROVEMENT,
    REPORTS_DIR,
    ROLLBACK_EVAL_DAYS,
    ROLLBACK_TOLERANCE,
    SCHEDULED_RETRAIN_DAYS,
    TARGET_COL,
)
from src.evaluation import baseline_forecasts, chronological_split  # noqa: E402
from src.modeling import (  # noqa: E402
    ExportBundle,
    fit_direct,
    load_dataset,
    predict_direct,
    with_forecast_error,
)

logger = logging.getLogger(__name__)

SIMULATION_MODEL_NAME = f"{MODEL_NAME}-simulation"
BASELINE = "Seasonal naive (24h)"
DRIFT_FEATURES = ("temperature", TARGET_COL)

INJECTED_OUTAGE = (pd.Timestamp("2017-03-14 00:00"), pd.Timestamp("2017-03-16 23:00"))
INJECTED_HOTFIX_DATE = pd.Timestamp("2017-09-05")
INJECTED_CALIBRATION_ERROR = 1.08     # hotfix trained on readings 8 % too high

DRIFT_PSI_CLEAR = 0.15                # drift alert clears below this (hysteresis)
MAX_HOURLY_CHANGE = 0.35             # |x_t / x_{t-1} - 1| above this is implausible
MAX_FLATLINE_HOURS = 6


# --------------------------------------------------------------------------
# Pure policy functions (unit-tested)
# --------------------------------------------------------------------------


def population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """PSI with quantile bins from the reference distribution."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    expected = np.histogram(reference, edges)[0] / len(reference)
    actual = np.histogram(current, edges)[0] / len(current)
    expected = np.clip(expected, 1e-4, None)
    actual = np.clip(actual, 1e-4, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def quality_issues(readings: np.ndarray) -> list[str]:
    """Problems in a window of incoming hourly demand readings."""
    values = np.asarray(readings, dtype=float)
    issues = []
    if not np.isfinite(values).all():
        issues.append(f"{int((~np.isfinite(values)).sum())} missing readings")
        values = values[np.isfinite(values)]
    if len(values) == 0:
        return issues
    if (values <= 0).any():
        issues.append("non-positive readings")
    run = longest = 1
    for previous, current in zip(values, values[1:]):
        run = run + 1 if current == previous else 1
        longest = max(longest, run)
    if longest >= MAX_FLATLINE_HOURS:
        issues.append(f"flat-lined for {longest} hours")
    positive = values[values > 0]
    if len(positive) > 1 and (np.abs(positive[1:] / positive[:-1] - 1) > MAX_HOURLY_CHANGE).any():
        issues.append("implausible hour-to-hour jump")
    return issues


def performance_alert(ratios: list[float], reference: float,
                      ratio_limit: float = PERFORMANCE_ALERT_RATIO,
                      days: int = PERFORMANCE_ALERT_DAYS) -> bool:
    """True when the last ``days`` skill ratios all exceed the limit."""
    recent = [r for r in ratios[-days:] if np.isfinite(r)]
    return len(recent) == days and all(r > reference * ratio_limit for r in recent)


def promote(candidate_mae: float, champion_mae: float,
            min_improvement: float = PROMOTION_MIN_IMPROVEMENT) -> bool:
    return candidate_mae <= champion_mae * (1 - min_improvement)


def should_roll_back(champion_mae: float, previous_mae: float,
                     tolerance: float = ROLLBACK_TOLERANCE) -> bool:
    return champion_mae > previous_mae * (1 + tolerance)


# --------------------------------------------------------------------------
# Model versions and registry
# --------------------------------------------------------------------------


@dataclass
class ModelVersion:
    number: int
    predict: Callable[[pd.DataFrame], np.ndarray]
    trained_through: pd.Timestamp
    source: str
    deployed: pd.Timestamp | None = None
    registry_version: str | None = None


class SimulationRegistry:
    """Tracks versions and, when MLflow is enabled, mirrors them into the
    model registry with ``champion`` / ``previous`` aliases."""

    def __init__(self, family: str, input_example: pd.DataFrame, use_mlflow: bool):
        self.family = family
        self.input_example = input_example
        self.use_mlflow = use_mlflow
        self.versions: list[ModelVersion] = []

    def add(self, predict, trained_through, source, models=None, run_uri=None) -> ModelVersion:
        version = ModelVersion(len(self.versions) + 1, predict, trained_through, source)
        if self.use_mlflow:
            import mlflow

            if run_uri is not None:
                info = mlflow.register_model(run_uri, SIMULATION_MODEL_NAME)
                version.registry_version = str(info.version)
            else:
                with ExportBundle(models, self.family, self.input_example,
                                  {"source": source, "trained_through": str(trained_through)}) as bundle:
                    info = mlflow.pyfunc.log_model(
                        name=f"model_v{version.number}",
                        registered_model_name=SIMULATION_MODEL_NAME,
                        **bundle.kwargs,
                    )
                version.registry_version = str(info.registered_model_version)
        self.versions.append(version)
        return version

    def set_aliases(self, champion: ModelVersion, previous: ModelVersion | None) -> None:
        if not self.use_mlflow:
            return
        import mlflow

        client = mlflow.MlflowClient()
        client.set_registered_model_alias(SIMULATION_MODEL_NAME, CHAMPION_ALIAS,
                                          champion.registry_version)
        if previous is not None:
            client.set_registered_model_alias(SIMULATION_MODEL_NAME, "previous",
                                              previous.registry_version)


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------


@dataclass
class SimulationState:
    champion: ModelVersion
    shadow: ModelVersion | None = None
    last_retrain: pd.Timestamp | None = None
    alert_active: bool = False
    drift_active: dict = field(default_factory=dict)
    daily: list = field(default_factory=list)
    events: list = field(default_factory=list)


def run(inject_faults: bool = True, tracking_uri: str | None = DEFAULT_TRACKING_URI,
        reports_dir=REPORTS_DIR, model_dir=MODEL_DIR) -> dict:
    report = json.loads((reports_dir / "metrics.json").read_text())
    family = report["selected_model"]
    trees = {int(h): n for h, n in report["trees_per_horizon"].items()}
    params = report["params"][family]

    data = load_dataset()
    X, Y, series = data.X, data.Y, data.series
    split = chronological_split(len(X))
    reference_ratio = _reference_skill_ratio(report, X.iloc[split.validation],
                                             Y.iloc[split.validation], series, tracking_uri)
    logger.info("Reference skill ratio (validation, forecast temperatures): %.3f", reference_ratio)
    test_index = X.index[split.test]
    origins = test_index[test_index.hour == 23]

    received = series.copy()
    if inject_faults:
        received.loc[INJECTED_OUTAGE[0]:INJECTED_OUTAGE[1]] = np.nan

    X_live = with_forecast_error(X.loc[origins], seed=7)
    baseline = baseline_forecasts(series, origins)[BASELINE]

    def retrain(cutoff: pd.Timestamp, target_scale: float = 1.0):
        """Train on origins whose targets are all observed before ``cutoff``."""
        rows = X.index <= cutoff - pd.Timedelta(hours=FORECAST_HORIZON)
        models, _ = fit_direct(family, X[rows], Y[rows] * target_scale,
                               n_estimators=trees, params=params)
        return models

    def in_memory(models):
        return lambda frame: predict_direct(models, frame).to_numpy()

    use_mlflow = tracking_uri is not None
    if use_mlflow:
        import mlflow

        from src.train import setup_mlflow

        setup_mlflow(tracking_uri)
        run_context = mlflow.start_run(run_name="production_simulation")
    else:
        from contextlib import nullcontext

        run_context = nullcontext()

    with run_context:
        registry = SimulationRegistry(family, X.iloc[:1], use_mlflow)

        import mlflow.pyfunc

        deployed_model = mlflow.pyfunc.load_model(str(model_dir))
        initial = registry.add(
            lambda frame: deployed_model.predict(frame).to_numpy(),
            pd.Timestamp(report["trained_through"]),
            "training pipeline",
            run_uri=(f"runs:/{report['mlflow']['final_run_id']}/model"
                     if use_mlflow and "mlflow" in report else None),
        )
        initial.deployed = origins[0]
        registry.set_aliases(initial, None)
        state = SimulationState(champion=initial, last_retrain=initial.trained_through)

        def event(day, kind, detail, **extra):
            state.events.append({"date": str(day.date()), "event": kind, "detail": detail, **extra})
            logger.info("%s  %-22s %s", day.date(), kind, detail)

        def deploy(day, version, kind, detail):
            previous = state.champion
            version.deployed = day
            state.shadow, state.champion = previous, version
            registry.set_aliases(version, previous)
            event(day, kind, detail, version=version.number, replaced=previous.number)

        hotfix_done = not inject_faults
        for origin in origins:
            day = origin + pd.Timedelta(hours=1)          # forecast day starts at 00:00
            row = {"date": day.normalize(), "champion": state.champion.number}

            # 1. Data quality of the inputs available at forecast time.
            issues = quality_issues(received.loc[origin - pd.Timedelta(hours=23):origin].to_numpy())
            if issues:
                event(day, "data_quality_block", "; ".join(issues))
                row.update(mae=np.nan, baseline_mae=np.nan)
                state.daily.append(row)
                continue

            # 2. Forecast, then score once actuals arrive.
            features = X_live.loc[[origin]]
            actual = Y.loc[origin].to_numpy()
            prediction = state.champion.predict(features)[0]
            row["mae"] = float(np.mean(np.abs(actual - prediction)))
            row["baseline_mae"] = float(np.mean(np.abs(actual - baseline.loc[origin].to_numpy())))
            if state.shadow is not None:
                row["shadow_mae"] = float(np.mean(np.abs(actual - state.shadow.predict(features)[0])))
            state.daily.append(row)

            history = pd.DataFrame(state.daily).dropna(subset=["mae"])
            window = history[history["date"] > day - pd.Timedelta(days=MONITOR_WINDOW_DAYS)]
            row["rolling_mae"] = float(window["mae"].mean())
            row["rolling_baseline_mae"] = float(window["baseline_mae"].mean())
            row["skill_ratio"] = row["rolling_mae"] / row["rolling_baseline_mae"]

            # 3. Drift against the same days of the year in the training data.
            recent = slice(origin - pd.Timedelta(days=MONITOR_WINDOW_DAYS), origin)
            days_of_year = pd.date_range(recent.start, recent.stop, freq="D").dayofyear
            past = X.index <= state.champion.trained_through
            same_season = X.index.dayofyear.isin(days_of_year) & past
            for feature in DRIFT_FEATURES:
                source = data.temperature if feature == "temperature" else series
                score = population_stability_index(X.loc[same_season, feature], source.loc[recent])
                row[f"psi_{feature}"] = score
                # Hysteresis: raise above the threshold, clear only well below it.
                if not state.drift_active.get(feature) and score > DRIFT_PSI_THRESHOLD:
                    state.drift_active[feature] = True
                    event(day, "drift_alert", f"{feature} PSI {score:.2f} vs same season in training")
                elif state.drift_active.get(feature) and score < DRIFT_PSI_CLEAR:
                    state.drift_active[feature] = False

            # 4. Rollback check while the previous champion runs in shadow.
            if state.shadow is not None:
                since = history[history["date"] >= state.champion.deployed.normalize()]
                since = since.dropna(subset=["shadow_mae"])
                if len(since) >= 7 and should_roll_back(since["mae"].mean(), since["shadow_mae"].mean()):
                    bad = state.champion
                    state.champion, state.shadow = state.shadow, None
                    registry.set_aliases(state.champion, bad)
                    event(day, "rollback",
                          f"v{bad.number} MAE {since['mae'].mean():.0f} vs v{state.champion.number} "
                          f"{since['shadow_mae'].mean():.0f} over {len(since)} days",
                          version=state.champion.number, replaced=bad.number)
                elif len(since) >= ROLLBACK_EVAL_DAYS:
                    event(day, "deployment_confirmed",
                          f"v{state.champion.number} within tolerance after {len(since)} days")
                    state.shadow = None

            # 5. Injected fault: manual hotfix bypassing the promotion gate.
            if not hotfix_done and day >= INJECTED_HOTFIX_DATE and state.shadow is None:
                hotfix_done = True
                models = retrain(origin, target_scale=INJECTED_CALIBRATION_ERROR)
                version = registry.add(in_memory(models), origin, "manual hotfix (INJECTED FAULT)",
                                       models=models)
                state.last_retrain = day
                deploy(day, version, "manual_deploy_injected",
                       "hotfix trained on mis-calibrated readings, gate bypassed")
                continue

            # 6. Performance alert and retraining decision.
            alert = performance_alert(list(history.get("skill_ratio", pd.Series(dtype=float))
                                           .dropna()) + [row["skill_ratio"]], reference_ratio)
            if alert and not state.alert_active:
                event(day, "performance_alert",
                      f"skill ratio {row['skill_ratio']:.3f} > {PERFORMANCE_ALERT_RATIO:.2f} x "
                      f"reference {reference_ratio:.3f} for {PERFORMANCE_ALERT_DAYS} days")
            state.alert_active = alert

            days_since = (day - state.last_retrain).days
            scheduled = days_since >= SCHEDULED_RETRAIN_DAYS
            if (alert or scheduled) and days_since >= MIN_DAYS_BETWEEN_RETRAINS and state.shadow is None:
                reason = "performance alert" if alert else "schedule"
                gate_start = origin - pd.Timedelta(days=PROMOTION_EVAL_DAYS)
                models = retrain(gate_start)
                state.last_retrain = day

                gate_rows = (X.index > gate_start) & (
                    X.index <= origin - pd.Timedelta(hours=FORECAST_HORIZON))
                X_gate = with_forecast_error(X[gate_rows], seed=day.dayofyear)
                Y_gate = Y[gate_rows].to_numpy()
                candidate_mae = float(np.mean(np.abs(Y_gate - in_memory(models)(X_gate))))
                champion_mae = float(np.mean(np.abs(Y_gate - state.champion.predict(X_gate))))
                detail = (f"{reason}: candidate {candidate_mae:.0f} vs champion "
                          f"{champion_mae:.0f} MW on last {PROMOTION_EVAL_DAYS} days")
                if promote(candidate_mae, champion_mae):
                    version = registry.add(in_memory(models), gate_start, f"retrain ({reason})",
                                           models=models)
                    deploy(day, version, "promoted", detail)
                else:
                    event(day, "candidate_rejected", detail)

        daily = pd.DataFrame(state.daily)
        events = pd.DataFrame(state.events)
        summary = _summarize(daily, events, registry, reference_ratio, inject_faults)

        out = reports_dir / "monitoring"
        out.mkdir(parents=True, exist_ok=True)
        daily.to_csv(out / "daily.csv", index=False, float_format="%.4f")
        events.to_csv(out / "events.csv", index=False)
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        figure = _timeline(daily, events, reports_dir / "figures" / "monitoring_timeline.png")

        if use_mlflow:
            mlflow.set_tags({"role": "production_simulation", "model_family": family,
                             "faults_injected": str(inject_faults)})
            for i, record in enumerate(daily.dropna(subset=["rolling_mae"]).itertuples()):
                mlflow.log_metric("live.rolling_mae", record.rolling_mae, step=i)
                mlflow.log_metric("live.skill_ratio", record.skill_ratio, step=i)
            for key, value in summary["counts"].items():
                mlflow.log_metric(f"events.{key}", value)
            mlflow.log_metric("live.mae", summary["live_mae"])
            mlflow.log_artifacts(str(out), artifact_path="monitoring")
            mlflow.log_artifact(str(figure))
    return summary


def _reference_skill_ratio(report, X_val, Y_val, series, tracking_uri) -> float:
    """Expected model/baseline MAE ratio under live conditions.

    The selected candidate never trained on the validation partition, so
    scoring it there with simulated temperature-forecast error gives an
    unbiased reference for the live ratio. Falls back to the
    observed-temperature validation ratio when MLflow is not available.
    """
    family = report["selected_model"]
    baseline_mae = report["validation"][BASELINE]["mean_mae"]
    if tracking_uri is None or "mlflow" not in report:
        return report["validation"][family]["mean_mae"] / baseline_mae

    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    runs = mlflow.search_runs(
        experiment_names=["electricity-demand-forecasting"],
        filter_string=(f"tags.mlflow.parentRunId = '{report['mlflow']['parent_run_id']}' "
                       f"and tags.role = 'candidate' and tags.model_family = '{family}'"),
    )
    candidate = mlflow.pyfunc.load_model(f"runs:/{runs['run_id'].iloc[0]}/model")
    predictions = candidate.predict(with_forecast_error(X_val, seed=3)).to_numpy()
    model_mae = float(np.mean(np.abs(Y_val.to_numpy() - predictions)))
    return model_mae / baseline_mae


def _summarize(daily, events, registry, reference_ratio, inject_faults) -> dict:
    scored = daily.dropna(subset=["mae"])
    counts = events["event"].value_counts().to_dict() if len(events) else {}
    return {
        "period": {"start": str(daily["date"].min().date()), "end": str(daily["date"].max().date())},
        "days": int(len(daily)),
        "days_forecast": int(len(scored)),
        "days_blocked_by_quality": int(daily["mae"].isna().sum()),
        "live_mae": float(scored["mae"].mean()),
        "live_baseline_mae": float(scored["baseline_mae"].mean()),
        "live_improvement_over_baseline": float(1 - scored["mae"].mean() / scored["baseline_mae"].mean()),
        "reference_skill_ratio": reference_ratio,
        "faults_injected": inject_faults,
        "counts": {k: int(v) for k, v in counts.items()},
        "versions": [
            {"version": v.number, "source": v.source, "trained_through": str(v.trained_through),
             "deployed": None if v.deployed is None else str(v.deployed.date()),
             "registry_version": v.registry_version}
            for v in registry.versions
        ],
        "final_champion": int(daily["champion"].iloc[-1]),
    }


def _timeline(daily: pd.DataFrame, events: pd.DataFrame, path):
    from src.plots import _save, plt

    fig, (ax, ax_psi) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                     gridspec_kw={"height_ratios": [2.2, 1]})
    ax.plot(daily["date"], daily["rolling_mae"], label="Champion, 28-day MAE", linewidth=2)
    ax.plot(daily["date"], daily["rolling_baseline_mae"], label="Seasonal naive 24h, 28-day MAE",
            color="tab:gray", linestyle="--")
    ax.set_ylabel("MAE (MW)")
    ax.set_title("Production simulation: live accuracy and lifecycle events")
    ax.grid(alpha=0.3)

    styles = {
        "promoted": ("tab:green", "Promoted"),
        "candidate_rejected": ("tab:olive", "Candidate rejected"),
        "rollback": ("tab:red", "Rollback"),
        "manual_deploy_injected": ("tab:orange", "Injected faulty deploy"),
        "data_quality_block": ("tab:purple", "Data-quality block"),
        "performance_alert": ("black", "Performance alert"),
    }
    if len(events):
        dates = pd.to_datetime(events["date"])
        for kind, (color, label) in styles.items():
            hits = dates[events["event"] == kind]
            if kind == "data_quality_block" and len(hits):
                hits = hits.iloc[:1]
            for i, date in enumerate(hits):
                ax.axvline(date, color=color, linestyle=":", linewidth=1.8,
                           label=label if i == 0 else None)
    ax.legend(loc="upper left", fontsize=8, ncol=2)

    for feature, color in (("temperature", "tab:red"), (TARGET_COL, "tab:blue")):
        column = f"psi_{feature}"
        if column in daily:
            ax_psi.plot(daily["date"], daily[column], color=color, label=f"PSI {feature}")
    ax_psi.axhline(DRIFT_PSI_THRESHOLD, color="black", linestyle="--", linewidth=1,
                   label="Drift threshold")
    ax_psi.set_ylabel("PSI vs same season")
    ax_psi.grid(alpha=0.3)
    ax_psi.legend(loc="upper left", fontsize=8)
    fig.autofmt_xdate()
    return _save(fig, path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Replay the test period as live production.")
    parser.add_argument("--no-faults", action="store_true", help="Do not inject faults.")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args(argv)

    from src.train import configure_logging

    configure_logging()
    summary = run(inject_faults=not args.no_faults,
                  tracking_uri=None if args.no_mlflow else DEFAULT_TRACKING_URI)
    logger.info("Live MAE %.1f MW (%.1f%% better than baseline); events: %s",
                summary["live_mae"], 100 * summary["live_improvement_over_baseline"],
                summary["counts"])


if __name__ == "__main__":
    main()
