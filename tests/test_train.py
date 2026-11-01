"""End-to-end smoke test of the training pipeline on small synthetic data."""

import json

import mlflow
import numpy as np
import pytest

import src.train as train
from src.config import CHAMPION_ALIAS, MODEL_NAME, model_params
from src.modeling import load_dataset, predict_direct
from tests.conftest import write_inputs


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    directory = tmp_path_factory.mktemp("pipeline")
    data_path, weather_path = write_inputs(directory, hours=24 * 120)

    def tiny(family):
        params = model_params(family)
        params.update(n_estimators=20, n_jobs=1)
        return params

    original = train.model_params
    train.model_params = tiny
    try:
        report = train.run(
            data_path=data_path,
            weather_path=weather_path,
            model_dir=directory / "model",
            reports_dir=directory / "reports",
            tracking_uri="sqlite:///" + (directory / "mlflow.db").as_posix(),
        )
    finally:
        train.model_params = original
    return directory, data_path, weather_path, report


def test_report_contents(trained):
    directory, *_, report = trained

    assert report["selected_model"] in ("LightGBM", "XGBoost")
    assert set(report["validation"]) >= {"LightGBM", "XGBoost", "Seasonal naive (24h)"}
    assert set(report["test_temperature_scenarios"]) == {"with forecast error", "observed temperature"}
    saved = json.loads((directory / "reports" / "metrics.json").read_text())
    assert saved["selected_model"] == report["selected_model"]
    assert (directory / "reports" / "figures" / "mae_by_horizon.png").exists()
    assert (directory / "reports" / "figures" / "eda_demand_vs_temperature.png").exists()


def test_partitions_do_not_overlap(trained):
    *_, report = trained
    parts = report["partitions"]
    order = ["fit", "early_stopping", "validation", "test"]
    for earlier, later in zip(order, order[1:]):
        assert parts[earlier]["end"] < parts[later]["start"]


def test_exported_model_loads_and_matches_metadata(trained):
    directory, data_path, weather_path, report = trained

    model = mlflow.pyfunc.load_model(str(directory / "model"))
    data = load_dataset(data_path, weather_path)
    output = model.predict(data.X.iloc[-5:])

    assert output.shape == (5, 24)
    assert np.isfinite(output.to_numpy()).all()
    assert model.metadata.metadata["model_family"] == report["selected_model"]
    assert (directory / "model" / "artifacts" / "estimators" / "manifest.json").exists()


def test_exported_model_is_portable_across_operating_systems(trained):
    """Built on Windows, served on Linux: MLmodel must only use relative
    paths, and the pickle must hold no model state (estimators are native files)."""
    import yaml

    directory, *_ = trained
    mlmodel = yaml.safe_load((directory / "model" / "MLmodel").read_text())
    pyfunc = mlmodel["flavors"]["python_function"]

    assert pyfunc["python_model"] == "python_model.pkl"
    assert pyfunc["artifacts"]["estimators"]["path"] == "artifacts/estimators"
    assert "model_code_path" not in pyfunc
    assert (directory / "model" / "python_model.pkl").stat().st_size < 10_000
    assert list((directory / "model" / "artifacts" / "estimators").glob("h*.gz"))


def test_mlflow_runs_and_registry(trained):
    directory, *_, report = trained
    mlflow.set_tracking_uri("sqlite:///" + (directory / "mlflow.db").as_posix())

    runs = mlflow.search_runs(experiment_names=["electricity-demand-forecasting"])
    names = set(runs["tags.mlflow.runName"])
    assert {"training_pipeline", "LightGBM_24h_candidate", "XGBoost_24h_candidate"} <= names

    client = mlflow.MlflowClient(registry_uri="sqlite:///" + (directory / "mlflow.db").as_posix())
    champion = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    assert str(champion.version) == report["mlflow"]["registered_version"]

    registered = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}")
    local = mlflow.pyfunc.load_model(str(directory / "model"))
    data = load_dataset(*trained[1:3])
    np.testing.assert_allclose(
        registered.predict(data.X.iloc[-3:]).to_numpy(),
        local.predict(data.X.iloc[-3:]).to_numpy(),
    )
