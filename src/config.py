"""Single source of truth for paths and modeling constants.

Both the training pipeline and the inference API import from here, so the
feature definition cannot silently drift between the two.
"""

import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_DIR / "data" / "PJME_hourly.csv"
WEATHER_PATH = PROJECT_DIR / "data" / "weather_hourly.csv"
MODEL_DIR = PROJECT_DIR / "models" / "electricity_demand_forecaster"
REPORTS_DIR = PROJECT_DIR / "reports"
TUNED_PARAMS_PATH = PROJECT_DIR / "configs" / "tuned_params.json"

TIMESTAMP_COL = "Datetime"
TARGET_COL = "PJME_MW"

MODEL_NAME = "ElectricityDemandForecaster"
CHAMPION_ALIAS = "champion"
EXPERIMENT_NAME = "electricity-demand-forecasting"
DEFAULT_TRACKING_URI = f"sqlite:///{(PROJECT_DIR / 'mlflow.db').as_posix()}"
DEFAULT_ARTIFACT_ROOT = (PROJECT_DIR / "mlartifacts").as_uri()

# Forecasting setup
FORECAST_HORIZON = 24
LAGS = (1, 24, 48, 168, 720)
ROLLING_WINDOWS = (24, 168, 720)

# Gaps longer than this are treated as a data problem rather than filled.
MAX_INTERPOLATION_GAP_HOURS = 3

# Chronological split fractions (the remainder is the test partition).
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
# Tail of the training partition held out for early stopping only.
EARLY_STOPPING_FRACTION = 0.10

# Features known at the forecast origin t (same for every horizon).
FEATURE_NAMES = [
    TARGET_COL,
    *(f"lag_{lag}" for lag in LAGS),
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_holiday",
    "is_holiday_tomorrow",
    "hour_sin",
    "hour_cos",
    *(f"rolling_mean_{window}" for window in ROLLING_WINDOWS),
    "temperature",
    "temperature_mean_24",
]

INTEGER_FEATURES = ["hour", "day_of_week", "month", "is_weekend",
                    "is_holiday", "is_holiday_tomorrow"]

# Temperature forecast for each target hour t+1 .. t+24 (°C).
TEMPERATURE_FORECAST_NAMES = [f"temp_forecast_h{h}" for h in range(1, FORECAST_HORIZON + 1)]

# Columns the deployed model receives: origin features + the 24 forecasts.
MODEL_INPUT_NAMES = FEATURE_NAMES + TEMPERATURE_FORECAST_NAMES

# Columns each horizon-h estimator sees: origin features + the forecast for t+h.
HORIZON_FEATURE_NAMES = FEATURE_NAMES + ["temperature_target"]

TARGET_NAMES = [f"target_{h}" for h in range(1, FORECAST_HORIZON + 1)]

# The longest look-back is lag_720 (the value 720 hours before the latest
# observation), so 721 consecutive readings are needed for one feature row.
MIN_HISTORY = max(max(LAGS), max(ROLLING_WINDOWS)) + 1

RANDOM_STATE = 42
EARLY_STOPPING_ROUNDS = 100

DEFAULT_LGBM_PARAMS = {
    "objective": "regression",
    "n_estimators": 5000,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 30,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.5,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}

DEFAULT_XGB_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 5000,
    "learning_rate": 0.05,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


def model_params(family: str) -> dict:
    """Default parameters, overridden by `python -m src.tune` results if present."""
    params = dict(DEFAULT_LGBM_PARAMS if family == "LightGBM" else DEFAULT_XGB_PARAMS)
    if TUNED_PARAMS_PATH.exists():
        tuned = json.loads(TUNED_PARAMS_PATH.read_text())
        params.update(tuned.get(family, {}).get("params", {}))
    return params


# --- Production simulation (src/monitor.py) -------------------------------
MONITOR_WINDOW_DAYS = 28            # rolling window for live MAE and drift
PERFORMANCE_ALERT_RATIO = 1.20      # alert when live MAE > ratio x reference MAE
PERFORMANCE_ALERT_DAYS = 7          # ...for this many consecutive days
DRIFT_PSI_THRESHOLD = 0.25          # population stability index alert level
MIN_DAYS_BETWEEN_RETRAINS = 60
SCHEDULED_RETRAIN_DAYS = 180        # retrain at least this often
PROMOTION_MIN_IMPROVEMENT = 0.01    # candidate must be >= 1 % better to deploy
PROMOTION_EVAL_DAYS = 28            # recent window used by the promotion gate
ROLLBACK_EVAL_DAYS = 14             # post-deployment shadow comparison window
ROLLBACK_TOLERANCE = 0.05           # roll back if champion > 5 % worse than previous
