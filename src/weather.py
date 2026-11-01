"""Hourly temperature for the PJM East service area.

Demand in PJM East is driven by air conditioning in summer and heating in
winter, so temperature is the single most informative exogenous variable.

The series is the equal-weight mean of ERA5 reanalysis 2 m temperature at
four load centres, fetched from the free Open-Meteo archive API:

    python -m src.weather          # writes data/weather_hourly.csv

Timestamps are converted from UTC to naive US/Eastern local time and then
regularized exactly like the demand series (DST duplicates averaged,
missing hours interpolated), so both series share one hourly grid.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import TIMESTAMP_COL, WEATHER_PATH
from src.data import DataValidationError, to_hourly_series

logger = logging.getLogger(__name__)

TEMPERATURE_COL = "temperature"

LOAD_CENTRES = {
    "Philadelphia": (39.95, -75.17),
    "Newark": (40.74, -74.17),
    "Baltimore": (39.29, -76.61),
    "Washington": (38.91, -77.04),
}

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
START_DATE = "2001-12-01"
END_DATE = "2018-08-05"
LOCAL_TIMEZONE = "America/New_York"

PLAUSIBLE_RANGE_C = (-40.0, 50.0)


def fetch_city(latitude: float, longitude: float, start: str, end: str) -> pd.Series:
    query = urllib.parse.urlencode({
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start,
        "end_date": end,
        "hourly": "temperature_2m",
        "timezone": "GMT",
    })
    for attempt in range(3):
        try:
            with urllib.request.urlopen(f"{ARCHIVE_URL}?{query}", timeout=120) as response:
                payload = json.load(response)
            break
        except OSError:
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))

    hourly = payload["hourly"]
    index = pd.to_datetime(hourly["time"], utc=True)
    return pd.Series(hourly["temperature_2m"], index=index, dtype=float)


def download(path: Path = WEATHER_PATH, start: str = START_DATE, end: str = END_DATE) -> Path:
    cities = {}
    for name, (lat, lon) in LOAD_CENTRES.items():
        logger.info("Fetching %s", name)
        cities[name] = fetch_city(lat, lon, start, end)

    utc_mean = pd.DataFrame(cities).mean(axis=1)
    local = utc_mean.tz_convert(LOCAL_TIMEZONE).tz_localize(None)

    frame = pd.DataFrame({TIMESTAMP_COL: local.index, TEMPERATURE_COL: local.round(2).values})
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    logger.info("Wrote %d rows to %s", len(frame), path)
    return path


def load_temperature(path: Path = WEATHER_PATH) -> pd.Series:
    """Validated temperature (°C) on a regular hourly local-time grid."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Weather data not found: {path}. Run `python -m src.weather` to download it."
        )

    raw = pd.read_csv(path)
    if {TIMESTAMP_COL, TEMPERATURE_COL} - set(raw.columns):
        raise DataValidationError(f"Weather file must contain {TIMESTAMP_COL} and {TEMPERATURE_COL}.")

    raw[TIMESTAMP_COL] = pd.to_datetime(raw[TIMESTAMP_COL], errors="raise")
    values = pd.to_numeric(raw[TEMPERATURE_COL], errors="coerce").to_numpy()
    low, high = PLAUSIBLE_RANGE_C
    if not np.isfinite(values).all() or (values < low).any() or (values > high).any():
        raise DataValidationError("Temperature contains missing or implausible values.")

    series, _ = to_hourly_series(raw, value_col=TEMPERATURE_COL)
    return series


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download PJM East hourly temperature.")
    parser.add_argument("--out", type=Path, default=WEATHER_PATH)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    download(args.out)


if __name__ == "__main__":
    main()
