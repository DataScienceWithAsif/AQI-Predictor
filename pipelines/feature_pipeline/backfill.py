"""
feature_pipeline/backfill.py

Pulls REAL historical weather + air-quality data and pushes it into the
Hopsworks Feature Store, so you have enough history to train on by Day 5 —
a few days of live hourly collection alone won't be enough.

Data sources (chosen deliberately — see the note in the roadmap conversation
for why this replaces the OpenWeather-history + AQICN + Kaggle patchwork):

  - Open-Meteo Historical Weather API  (temp, humidity, pressure, wind)
    https://open-meteo.com/en/docs/historical-weather-api
  - Open-Meteo Air Quality API         (pm2.5, pm10, co, no2, so2, o3, US AQI)
    https://open-meteo.com/en/docs/air-quality-api

Both are free, require NO API key, and — importantly — the Air Quality API
gives you a real, computed 0-500 US AQI value historically. That solves the
"AQICN has no free historical bulk endpoint" problem head-on, without
needing a Kaggle dataset or a multi-day manual cron.

Run:
    python feature_pipeline/backfill.py
"""

import os
import logging
import time
from datetime import date, timedelta

import requests
import pandas as pd
from dotenv import load_dotenv

import features  # for RAW_COLUMNS (the raw hourly schema)
import final_features  # for the hybrid hourly-features + daily-targets computation
import hopsworks_io
from get_coords import geocode_city

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("feature_pipeline.backfill")

# Open-Meteo's Air Quality API officially documents up to 92 days of
# history. Going further back has been observed to sometimes work but is
# undocumented/unsupported — don't build on it.
AIR_QUALITY_BACKFILL_DAYS = 90

# The Historical Weather API is built on ERA5 reanalysis, which has a few
# days of processing latency — asking for "yesterday" can return nulls for
# the most recent days. Staying a week behind "today" avoids that.
WEATHER_ARCHIVE_LATENCY_DAYS = 6

REQUEST_TIMEOUT_SECONDS = 500
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2


def _get_with_retries(url: str, params: dict, source_name: str) -> requests.Response:
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt == MAX_RETRIES:
                break
            backoff_seconds = RETRY_BACKOFF_SECONDS ** attempt
            logger.warning(
                f"{source_name} request failed (attempt {attempt}/{MAX_RETRIES}): {e}. "
                f"Retrying in {backoff_seconds}s..."
            )
            time.sleep(backoff_seconds)

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"{source_name} request failed without an explicit exception.")


def fetch_historical_air_quality(lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    days_back = (date.today() - start).days
    if days_back > AIR_QUALITY_BACKFILL_DAYS + WEATHER_ARCHIVE_LATENCY_DAYS + 2:
        # The Air Quality API's real, reliable coverage is ~92 days — unlike
        # the Historical Weather API (which goes back to 1940), it does NOT
        # extend arbitrarily far back. A window starting further back than
        # that will typically return empty/null hourly arrays rather than
        # an error, which would otherwise silently poison your dataset with
        # unusable rows. Fail loudly here instead.
        raise ValueError(
            f"Requested air-quality start date {start} is {days_back} days ago, "
            f"beyond Open-Meteo's ~92-day supported history for this endpoint. "
            f"pm2.5/pm10/us_aqi for this range will likely come back empty. "
            f"Reduce AIR_QUALITY_BACKFILL_DAYS/WEATHER_ARCHIVE_LATENCY_DAYS, or "
            f"don't try to stitch together a second, older window this way."
        )

    resp = _get_with_retries(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "hourly": "pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi",
            "timezone": "UTC",
        },
        source_name="Air quality API",
    )
    hourly = resp.json()["hourly"]

    return pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"], utc=True),
        "pm25": hourly["pm2_5"],
        "pm10": hourly["pm10"],
        "co": hourly["carbon_monoxide"],
        "no2": hourly["nitrogen_dioxide"],
        "so2": hourly["sulphur_dioxide"],
        "o3": hourly["ozone"],
        "aqi": hourly["us_aqi"],
    })


def fetch_historical_weather(lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    resp = _get_with_retries(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "hourly": "temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m",
            "wind_speed_unit": "ms",  # match the m/s units fetch.py gets live from OpenWeather
            "timezone": "UTC",
        },
        source_name="Weather archive API",
    )
    hourly = resp.json()["hourly"]

    return pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"], utc=True),
        "temp": hourly["temperature_2m"],
        "humidity": hourly["relative_humidity_2m"],
        "pressure": hourly["pressure_msl"],
        "wind_speed": hourly["wind_speed_10m"],
    })


def build_historical_raw_df(
    city: str, lat: float, lon: float, days: int = AIR_QUALITY_BACKFILL_DAYS
) -> pd.DataFrame:
    """
    Pulls both historical sources for an overlapping window they both
    support, and merges them into the exact same schema fetch.py produces
    live (see features.RAW_COLUMNS) — that's what lets compute_features()
    run identically over backfilled AND live data later.
    """
    end = date.today() - timedelta(days=WEATHER_ARCHIVE_LATENCY_DAYS)
    start = end - timedelta(days=days)

    logger.info(f"Backfilling {city} ({lat}, {lon}) from {start} to {end} ({days} days)...")

    air_quality_df = fetch_historical_air_quality(lat, lon, start, end)
    weather_df = fetch_historical_weather(lat, lon, start, end)

    merged = pd.merge(weather_df, air_quality_df, on="timestamp", how="inner")
    merged.insert(1, "city", city)

    logger.info(f"Got {len(merged)} merged hourly rows.")
    return merged[features.RAW_COLUMNS]


def main():
    # CITIES is a comma-separated list, e.g. "Islamabad,Lahore,Karachi,Peshawar,Multan".
    # Falls back to the single CITY_NAME/LAT/LON you already have in .env if
    # CITIES isn't set, so this is backward compatible with your existing setup.
    cities_env = os.environ.get("CITIES")
    if cities_env:
        city_names = [c.strip() for c in cities_env.split(",") if c.strip()]
        logger.info(f"Backfilling {len(city_names)} cities: {city_names}")
        raw_dfs = []
        for city_name in city_names:
            lat, lon = geocode_city(city_name)
            raw_dfs.append(build_historical_raw_df(city_name, lat, lon))
        raw_df = pd.concat(raw_dfs, ignore_index=True)
    else:
        city = os.environ["CITY_NAME"]
        lat = float(os.environ["LAT"])
        lon = float(os.environ["LON"])
        raw_df = build_historical_raw_df(city, lat, lon)

    logger.info(f"Total raw rows across all cities: {len(raw_df)}")

    null_fraction = raw_df.drop(columns=["timestamp", "city"]).isna().mean().mean()
    logger.info(f"Average null fraction across raw columns (all cities): {null_fraction:.2%}")
    if null_fraction > 0.05:
        logger.warning(
            "More than 5% nulls in the backfilled raw data — double check "
            "your city names/lat-lon and date range before trusting this for training."
        )
    per_city_counts = raw_df.groupby("city").size()
    logger.info(f"Rows per city:\n{per_city_counts.to_string()}")

    features_df, targets_df = final_features.compute_features_with_daily_targets(raw_df)
    logger.info(f"Computed features_df shape={features_df.shape}, targets_df shape={targets_df.shape}")
    logger.info(f"Target columns: {list(targets_df.columns)}")

    # Local checkpoints first, so you have something to inspect even if the
    # Hopsworks push below fails for a credentials/network reason.
    raw_df.to_csv("day4_backfilled_raw.csv", index=False)
    features_df.to_csv("day4_backfilled_features.csv", index=False)
    targets_df.to_csv("day4_backfilled_targets.csv", index=False)
    logger.info("Saved local CSV checkpoints (day4_backfilled_*.csv).")

    hopsworks_io.insert_features_and_targets(features_df, targets_df, version=3)

    logger.info(
        "Done. Check the Hopsworks UI: Feature Store -> Feature Groups -> "
        "aqi_features / aqi_targets, to confirm the rows landed."
    )


if __name__ == "__main__":
    main()
