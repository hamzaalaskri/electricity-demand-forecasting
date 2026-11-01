from src.plots import eda_figures
from tests.conftest import synthetic_series, synthetic_temperature


def test_eda_figures_are_written(tmp_path):
    series = synthetic_series(24 * 400)

    paths = eda_figures(series, tmp_path, temperature=synthetic_temperature(24 * 400))

    assert {p.name for p in paths} == {
        "eda_demand_vs_temperature.png",
        "eda_time_series.png",
        "eda_seasonal_profiles.png",
        "eda_peaks_by_month.png",
        "eda_autocorrelation.png",
    }
    assert all(p.stat().st_size > 10_000 for p in paths)
