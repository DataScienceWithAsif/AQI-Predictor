"""
feature_pipeline/fetch.py

Fetches current weather + air pollution data from OpenWeather, and the
current station AQI from AQICN, and merges them into a single flat row:

    {timestamp, city, temp, humidity, wind_speed, pressure,
     pm25, pm10, co, no2, so2, o3, aqi}

This is the function GitHub Actions will call every hour starting Day 7.

Run standalone for local testing:
    python feature_pipeline/fetch.py
"""

import os
import csv
import logging
from datetime import datetime, timezone
from pathlib import Path

from typing import Optional

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from get_coords import geocode_city

# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------

load_dotenv()  # reads .env in the current working directory, if present
open_weather_api_key = os.getenv("OPENWEATHER_KEY")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("feature_pipeline.fetch")

REQUIRED_ENV_VARS = ["OPENWEATHER_KEY", "AQICN_TOKEN"]


def get_config() -> dict:
    """Load and validate required config from environment variables."""
    missing = [v for v in REQUIRED_ENV_VARS if not os.environ.get(v)]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {missing}. "
            f"Copy .env.example to .env and fill in your keys."
        )
    return {
        "openweather_key": os.environ["OPENWEATHER_KEY"],
        "aqicn_token": os.environ["AQICN_TOKEN"]
    }


def _session_with_retries() -> requests.Session:
    """
    A requests Session that retries transient failures (timeouts, 5xx errors)
    with exponential backoff, but fails fast on 4xx (bad key, bad request) —
    retrying those wouldn't help and would just waste your rate limit.
    """
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1,  # waits 1s, 2s, 4s between retries
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


SESSION = _session_with_retries()
TIMEOUT_SECONDS = 100  # never let an hourly cron job hang forever on a stuck request


# --------------------------------------------------------------------------
# Fetch functions
# --------------------------------------------------------------------------

def fetch_weather(lat: float, lon: float, api_key: str) -> Optional[dict]:
    """
    Calls OpenWeather's Current Weather + Air Pollution endpoints and
    returns one merged dict of raw fields. Returns None if either call
    fails after retries, or the response is missing expected fields.
    """
    try:
        weather_resp = SESSION.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"lat": lat, "lon": lon, "appid": api_key, "units": "metric"},
            timeout=TIMEOUT_SECONDS,
        )
        weather_resp.raise_for_status()
        weather = weather_resp.json()

        pollution_resp = SESSION.get(
            "http://api.openweathermap.org/data/2.5/air_pollution",
            params={"lat": lat, "lon": lon, "appid": api_key},
            timeout=TIMEOUT_SECONDS,
        )
        pollution_resp.raise_for_status()
        pollution = pollution_resp.json()

    except requests.exceptions.RequestException as e:
        logger.error(f"fetch_weather failed: {e}")
        return None

    try:
        components = pollution["list"][0]["components"]
        return {
            "temp": weather["main"]["temp"],
            "humidity": weather["main"]["humidity"],
            "pressure": weather["main"]["pressure"],
            "wind_speed": weather["wind"]["speed"],
            "pm25": components.get("pm2_5"),
            "pm10": components.get("pm10"),
            "co": components.get("co"),
            "no2": components.get("no2"),
            "so2": components.get("so2"),
            "o3": components.get("o3"),
        }
    except (KeyError, IndexError) as e:
        logger.error(f"fetch_weather: unexpected response shape, missing {e}")
        return None


def fetch_aqi(city_or_station: str, token: str) -> Optional[dict]:
    """
    Calls the AQICN feed API for a city/station and returns the standard
    0-500 US-EPA-style AQI plus dominant pollutant. Returns None on failure.
    """
    try:
        resp = SESSION.get(
            f"https://api.waqi.info/feed/{city_or_station}/",
            params={"token": token},
            timeout=TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        payload = resp.json()

        if payload.get("status") != "ok":
            logger.error(f"fetch_aqi: AQICN returned status={payload.get('status')} — {payload}")
            return None

        data = payload["data"]
        return {
            "aqi": data.get("aqi"),
            "dominant_pollutant": data.get("dominentpol"),
            "station_name": data.get("city", {}).get("name"),
        }

    except requests.exceptions.RequestException as e:
        logger.error(f"fetch_aqi failed: {e}")
        return None
    except KeyError as e:
        logger.error(f"fetch_aqi: unexpected response shape, missing {e}")
        return None


# --------------------------------------------------------------------------
# Merge into one feature row
# --------------------------------------------------------------------------

def build_feature_row(
    city: str, lat: float, lon: float, openweather_key: str, aqicn_token: str
) -> dict:
    """
    Fetches weather+pollution and AQI, and merges them into one flat row
    matching the schema: timestamp, city, temp, humidity, wind_speed,
    pressure, pm25, pm10, co, no2, so2, o3, aqi.

    Raises RuntimeError if either source fails, so a failed hourly run shows
    up as a failed (red) GitHub Actions run instead of silently writing a
    row full of nulls into the feature store.
    """
    weather = fetch_weather(lat, lon, openweather_key)
    aqi_data = fetch_aqi(city, aqicn_token)

    if weather is None or aqi_data is None:
        raise RuntimeError(
            f"build_feature_row failed for city={city}: "
            f"weather_ok={weather is not None}, aqi_ok={aqi_data is not None}"
        )

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "city": city,
        **weather,
        "aqi": aqi_data["aqi"],
    }


# --------------------------------------------------------------------------
# Local test runner
# --------------------------------------------------------------------------

def append_row_to_csv(row: dict, path: str = "day2_feature_pipeline_log.csv") -> None:
    """Appends one row to a local CSV, writing the header only if the file is new."""
    file_exists = Path(path).exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    CITY_NAME = "Islamabad,PK"
    config = get_config()
    logger.info(f"Fetching feature row for {CITY_NAME}...")
 
    lat, lon, resolved_name, country = geocode_city(CITY_NAME, open_weather_api_key)

    row = build_feature_row(
        city=resolved_name,
        lat=lat,
        lon=lon,
        openweather_key=config["openweather_key"],
        aqicn_token=config["aqicn_token"],
    )

    logger.info(f"Row: {row}")
    append_row_to_csv(row)
    logger.info("Appended to day2_feature_pipeline_log.csv")


if __name__ == "__main__":
    main()