"""Loading, validating, and regularizing the raw PJME demand series."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    DATA_PATH,
    MAX_INTERPOLATION_GAP_HOURS,
    TARGET_COL,
    TIMESTAMP_COL,
)


class DataValidationError(ValueError):
    """Raised when the raw data violates an assumption of the pipeline."""


@dataclass(frozen=True)
class DataQualityReport:
    raw_rows: int
    duplicate_timestamps: int
    filled_hours: int
    final_rows: int
    start: str
    end: str

    def to_dict(self) -> dict:
        return asdict(self)


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the CSV and validate its schema and target values."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    missing = {TIMESTAMP_COL, TARGET_COL} - set(df.columns)
    if missing:
        raise DataValidationError(f"Missing required columns: {sorted(missing)}")

    df = df[[TIMESTAMP_COL, TARGET_COL]].copy()
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="raise")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

    values = df[TARGET_COL].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise DataValidationError("Target contains missing or non-numeric values.")
    if (values <= 0).any():
        raise DataValidationError("Target contains zero or negative demand values.")

    return df


def to_hourly_series(
    df: pd.DataFrame,
    value_col: str = TARGET_COL,
) -> tuple[pd.Series, DataQualityReport]:
    """Turn raw rows into a strictly regular hourly series.

    The raw file is local time, so DST produces two kinds of irregularity:

    * repeated hours in autumn: both readings are genuine, so they are averaged;
    * missing hours in spring: filled by time interpolation.

    Lag and target features are built with positional shifts, so a regular
    grid guarantees that ``lag_24`` really means 24 hours earlier.
    """
    raw_rows = len(df)
    duplicate_timestamps = int(df[TIMESTAMP_COL].duplicated().sum())

    series = (
        df.groupby(TIMESTAMP_COL)[value_col]
        .mean()
        .sort_index()
        .astype(np.float64)
    )

    full_index = pd.date_range(series.index.min(), series.index.max(), freq="h")
    regular = series.reindex(full_index)

    missing = regular.isna()
    longest_gap = _longest_run(missing.to_numpy())
    if longest_gap > MAX_INTERPOLATION_GAP_HOURS:
        raise DataValidationError(
            f"Found a gap of {longest_gap} consecutive missing hours; "
            f"only gaps up to {MAX_INTERPOLATION_GAP_HOURS} hours are filled."
        )

    regular = regular.interpolate(method="time")
    regular.index.name = TIMESTAMP_COL
    regular.name = value_col

    report = DataQualityReport(
        raw_rows=raw_rows,
        duplicate_timestamps=duplicate_timestamps,
        filled_hours=int(missing.sum()),
        final_rows=len(regular),
        start=str(regular.index.min()),
        end=str(regular.index.max()),
    )
    return regular, report


def load_series(path: Path = DATA_PATH) -> tuple[pd.Series, DataQualityReport]:
    return to_hourly_series(load_raw(path))


def _longest_run(mask: np.ndarray) -> int:
    longest = current = 0
    for flag in mask:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return longest
