"""The deployable MLflow PyFunc model: one gradient-boosted estimator per horizon.

Estimators are stored in their libraries' native, version-stable formats
(LightGBM text, XGBoost UBJ, both gzipped) and loaded in ``load_context``.
The wrapper itself is saved empty, so MLflow's pickle holds only a
reference to this class (bundled with the model as code), never model
state; the model loads on any OS and Python version that has the
libraries installed.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from src.config import FORECAST_HORIZON, HORIZON_FEATURE_NAMES, MODEL_INPUT_NAMES
from src.features import horizon_matrix

MANIFEST = "manifest.json"


class ElectricityDemandForecaster(mlflow.pyfunc.PythonModel):
    """Direct multi-horizon forecaster.

    ``models[h]`` predicts demand ``h`` hours after the latest observation
    from :func:`src.features.horizon_matrix`. The output has one row per
    input row and columns ``forecast_h1`` to ``forecast_hN``.
    """

    def __init__(self, models: dict[int, object] | None = None):
        self.models = {}
        if models is not None:
            self._set_models(models)

    def load_context(self, context) -> None:
        self._set_models(load_estimators(Path(context.artifacts["estimators"])))

    def _set_models(self, models: dict[int, object]) -> None:
        horizons = sorted(models)
        if horizons != list(range(1, len(horizons) + 1)):
            raise ValueError("Models must cover horizons 1..N without gaps.")
        self.models = dict(models)

    @property
    def horizon(self) -> int:
        return len(self.models)

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        missing = set(MODEL_INPUT_NAMES) - set(model_input.columns)
        if missing:
            raise ValueError(f"Missing input features: {sorted(missing)}")

        predictions = np.column_stack([
            self.models[h].predict(horizon_matrix(model_input, h))
            for h in range(1, self.horizon + 1)
        ])
        return pd.DataFrame(
            predictions,
            columns=[f"forecast_h{h}" for h in range(1, self.horizon + 1)],
            index=model_input.index,
        )


# --------------------------------------------------------------------------
# Native serialization
# --------------------------------------------------------------------------


def save_estimators(models: dict[int, object], family: str, directory: Path) -> Path:
    """Write each fitted sklearn-API estimator in its native format, gzipped
    (LightGBM text model, XGBoost Universal Binary JSON)."""
    directory.mkdir(parents=True, exist_ok=True)
    entries = {}
    for h, model in sorted(models.items()):
        if family == "LightGBM":
            filename = f"h{h:02d}.txt.gz"
            best = getattr(model, "best_iteration_", 0) or None
            payload = model.booster_.model_to_string(num_iteration=best).encode()
            n_trees = best or model.booster_.current_iteration()
        elif family == "XGBoost":
            filename = f"h{h:02d}.ubj.gz"
            booster = model.get_booster()
            best = getattr(model, "best_iteration", None)
            n_trees = (best + 1) if best is not None else booster.num_boosted_rounds()
            payload = bytes(booster[:n_trees].save_raw("ubj"))
        else:
            raise ValueError(f"Unknown model family: {family}")
        (directory / filename).write_bytes(gzip.compress(payload, compresslevel=6))
        entries[str(h)] = {"file": filename, "n_trees": int(n_trees)}

    manifest = {
        "family": family,
        "forecast_horizon": FORECAST_HORIZON,
        "feature_names": HORIZON_FEATURE_NAMES,
        "estimators": entries,
    }
    (directory / MANIFEST).write_text(json.dumps(manifest, indent=2))
    return directory


def load_estimators(directory: Path) -> dict[int, object]:
    manifest = json.loads((directory / MANIFEST).read_text())
    if manifest["feature_names"] != HORIZON_FEATURE_NAMES:
        raise ValueError("Saved estimators were trained on a different feature set.")

    loader = {"LightGBM": _LightGBMEstimator, "XGBoost": _XGBoostEstimator}[manifest["family"]]
    return {
        int(h): loader(directory / entry["file"], entry["n_trees"])
        for h, entry in manifest["estimators"].items()
    }


class _LightGBMEstimator:
    def __init__(self, path: Path, n_trees: int):
        import lightgbm as lgb

        self.booster = lgb.Booster(model_str=gzip.decompress(path.read_bytes()).decode())
        self.n_trees = n_trees

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(X, num_iteration=self.n_trees)


class _XGBoostEstimator:
    def __init__(self, path: Path, n_trees: int):
        import xgboost as xgb

        self.booster = xgb.Booster()
        self.booster.load_model(bytearray(gzip.decompress(path.read_bytes())))
        self.n_trees = n_trees
        self._DMatrix = xgb.DMatrix

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(self._DMatrix(X), iteration_range=(0, self.n_trees))
