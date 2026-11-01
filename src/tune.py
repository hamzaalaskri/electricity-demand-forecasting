"""Hyperparameter search for both model families with Optuna.

    python -m src.tune            # writes configs/tuned_params.json

Protocol (the validation and test partitions are never touched):

* fit on the "fit" part of the training partition;
* the "early stopping" tail of the training partition is split in two
  (with a purge gap): early stopping uses the first half, the trial is
  scored on the second half;
* the score is the mean MAE over representative horizons (1, 12, 24 h);
* the current default parameters are always evaluated as the first trial,
  so the selected parameters are never worse than the defaults on this
  objective.

Both families get the same trial budget so the later comparison is fair.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import numpy as np  # noqa: E402
import optuna  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import (  # noqa: E402
    DEFAULT_LGBM_PARAMS,
    DEFAULT_TRACKING_URI,
    DEFAULT_XGB_PARAMS,
    FORECAST_HORIZON,
    REPORTS_DIR,
    TUNED_PARAMS_PATH,
)
from src.evaluation import chronological_split  # noqa: E402
from src.features import horizon_matrix  # noqa: E402
from src.modeling import FAMILIES, fit_one, load_dataset  # noqa: E402

logger = logging.getLogger(__name__)

TUNING_HORIZONS = (1, 12, 24)
TUNED_KEYS = {
    "LightGBM": ["learning_rate", "num_leaves", "min_child_samples", "subsample",
                 "colsample_bytree", "reg_alpha", "reg_lambda"],
    "XGBoost": ["learning_rate", "max_depth", "min_child_weight", "subsample",
                "colsample_bytree", "reg_alpha", "reg_lambda"],
}


def suggest(family: str, trial: optuna.Trial) -> dict:
    if family == "LightGBM":
        params = dict(DEFAULT_LGBM_PARAMS)
        params.update(
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
            num_leaves=trial.suggest_int("num_leaves", 31, 255, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 200, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        )
    else:
        params = dict(DEFAULT_XGB_PARAMS)
        params.update(
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
            max_depth=trial.suggest_int("max_depth", 4, 10),
            min_child_weight=trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        )
    return params


def default_trial(family: str) -> dict:
    defaults = DEFAULT_LGBM_PARAMS if family == "LightGBM" else DEFAULT_XGB_PARAMS
    values = {k: defaults[k] for k in TUNED_KEYS[family] if k in defaults}
    if family == "XGBoost":
        values.setdefault("min_child_weight", 1.0)
        values.setdefault("reg_alpha", 1e-3)
        values.setdefault("reg_lambda", 1.0)
    return values


def tuning_data(X: pd.DataFrame, Y: pd.DataFrame) -> dict:
    split = chronological_split(len(X))
    stop = np.arange(len(X))[split.early_stopping]
    middle = len(stop) // 2
    stop_rows, score_rows = stop[: middle - FORECAST_HORIZON], stop[middle:]

    data = {}
    for h in TUNING_HORIZONS:
        target = f"target_{h}"
        data[h] = {
            "fit": (horizon_matrix(X.iloc[split.fit], h), Y.iloc[split.fit][target]),
            "stop": (horizon_matrix(X.iloc[stop_rows], h), Y.iloc[stop_rows][target]),
            "score": (horizon_matrix(X.iloc[score_rows], h), Y.iloc[score_rows][target]),
        }
    return data


def tune_family(family: str, data: dict, n_trials: int, timeout: float) -> optuna.Study:
    def objective(trial: optuna.Trial) -> float:
        params = suggest(family, trial)
        maes, trees = [], {}
        for step, h in enumerate(TUNING_HORIZONS):
            X_fit, y_fit = data[h]["fit"]
            X_stop, y_stop = data[h]["stop"]
            X_score, y_score = data[h]["score"]
            model, trees[h] = fit_one(family, params, X_fit, y_fit, X_stop, y_stop)
            maes.append(float(np.mean(np.abs(y_score.to_numpy() - model.predict(X_score)))))
            trial.report(float(np.mean(maes)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("trees", trees)
        return float(np.mean(maes))

    study = optuna.create_study(
        study_name=f"{family}-tuning",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    study.enqueue_trial(default_trial(family))
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return study


def run(n_trials: int, timeout: float, tracking_uri: str | None, families=FAMILIES) -> dict:
    data = load_dataset()
    prepared = tuning_data(data.X, data.Y)

    results, histories = {}, []
    for family in families:
        logger.info("Tuning %s (%d trials, %.0f s budget)", family, n_trials, timeout)
        started = time.time()
        study = tune_family(family, prepared, n_trials, timeout)
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        default_score = study.trials[0].value
        best = study.best_trial
        results[family] = {
            "params": {k: best.params[k] for k in TUNED_KEYS[family]},
            "objective_mae": best.value,
            "default_objective_mae": default_score,
            "improvement_over_default": 1 - best.value / default_score,
            "trials_completed": len(completed),
            "trials_pruned": len(study.trials) - len(completed),
            "seconds": round(time.time() - started),
        }
        logger.info("%s best MAE %.1f (default %.1f)", family, best.value, default_score)
        frame = study.trials_dataframe(attrs=("number", "value", "params", "state"))
        histories.append(frame.assign(family=family))

    payload = {
        "protocol": "fit on train 'fit' part; early-stop and score on the two halves "
                    "of the training tail; objective = mean MAE at horizons 1, 12, 24",
        "horizons": list(TUNING_HORIZONS),
        **results,
    }
    TUNED_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    TUNED_PARAMS_PATH.write_text(json.dumps(payload, indent=2))
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    history_path = REPORTS_DIR / "tuning_history.csv"
    pd.concat(histories).to_csv(history_path, index=False)

    if tracking_uri:
        import mlflow

        from src.train import setup_mlflow

        setup_mlflow(tracking_uri)
        with mlflow.start_run(run_name="hyperparameter_tuning"):
            for family, result in results.items():
                mlflow.log_params({f"{family}.{k}": v for k, v in result["params"].items()})
                mlflow.log_metric(f"{family}.objective_mae", result["objective_mae"])
                mlflow.log_metric(f"{family}.default_objective_mae", result["default_objective_mae"])
            mlflow.log_artifact(str(TUNED_PARAMS_PATH))
            mlflow.log_artifact(str(history_path))
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Tune LightGBM and XGBoost hyperparameters.")
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--timeout", type=float, default=900, help="Seconds per family.")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args(argv)

    from src.train import configure_logging

    configure_logging()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    run(args.trials, args.timeout, None if args.no_mlflow else DEFAULT_TRACKING_URI)


if __name__ == "__main__":
    main()
