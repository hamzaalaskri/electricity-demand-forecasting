import pandas as pd
import pytest

from src.data import DataValidationError, load_raw, to_hourly_series


def write_csv(tmp_path, rows):
    path = tmp_path / "data.csv"
    pd.DataFrame(rows, columns=["Datetime", "PJME_MW"]).to_csv(path, index=False)
    return path


def test_duplicate_timestamps_are_averaged(tmp_path):
    path = write_csv(tmp_path, [
        ("2020-11-01 00:00", 100.0),
        ("2020-11-01 01:00", 200.0),
        ("2020-11-01 01:00", 300.0),
        ("2020-11-01 02:00", 400.0),
    ])

    series, report = to_hourly_series(load_raw(path))

    assert series.loc["2020-11-01 01:00"] == 250.0
    assert report.duplicate_timestamps == 1
    assert report.final_rows == 3


def test_single_missing_hour_is_interpolated(tmp_path):
    path = write_csv(tmp_path, [
        ("2020-03-08 00:00", 100.0),
        ("2020-03-08 01:00", 200.0),
        ("2020-03-08 03:00", 400.0),
    ])

    series, report = to_hourly_series(load_raw(path))

    assert series.loc["2020-03-08 02:00"] == pytest.approx(300.0)
    assert report.filled_hours == 1
    assert (series.index[1:] - series.index[:-1] == pd.Timedelta(hours=1)).all()


def test_unsorted_input_is_sorted(tmp_path):
    path = write_csv(tmp_path, [
        ("2020-01-01 02:00", 3.0),
        ("2020-01-01 00:00", 1.0),
        ("2020-01-01 01:00", 2.0),
    ])

    series, _ = to_hourly_series(load_raw(path))

    assert series.tolist() == [1.0, 2.0, 3.0]


def test_long_gap_is_rejected(tmp_path):
    path = write_csv(tmp_path, [
        ("2020-01-01 00:00", 1.0),
        ("2020-01-01 10:00", 2.0),
    ])

    with pytest.raises(DataValidationError, match="gap"):
        to_hourly_series(load_raw(path))


def test_missing_column_is_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"Datetime": ["2020-01-01"], "Load": [1.0]}).to_csv(path, index=False)

    with pytest.raises(DataValidationError, match="PJME_MW"):
        load_raw(path)


@pytest.mark.parametrize("bad_value", [0.0, -5.0, "not-a-number"])
def test_invalid_target_values_are_rejected(tmp_path, bad_value):
    path = write_csv(tmp_path, [
        ("2020-01-01 00:00", 100.0),
        ("2020-01-01 01:00", bad_value),
    ])

    with pytest.raises(DataValidationError):
        load_raw(path)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw(tmp_path / "nope.csv")
