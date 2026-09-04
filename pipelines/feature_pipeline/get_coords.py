"""
feature_pipeline/geocoding.py

Shared free geocoding helper (Open-Meteo, no API key) — used by both
backfill.py and fetch.py so multi-city support doesn't require looking up
lat/lon by hand for every city, and isn't duplicated across scripts.
"""

import logging
import time

import requests

logger = logging.getLogger("feature_pipeline.geocoding")

REQUEST_TIMEOUT_SECONDS = 20
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2

def geocode_city(city_name: str) -> tuple:
    """
    Resolves a city name to (lat, lon) via Open-Meteo's free geocoding API.
    """
    results = None
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": city_name, "count": 1},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            results = resp.json().get("results")
            break
        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt == MAX_RETRIES:
                raise
            backoff_seconds = RETRY_BACKOFF_SECONDS ** attempt
            logger.warning(
                f"Geocoding request failed for '{city_name}' "
                f"(attempt {attempt}/{MAX_RETRIES}): {e}. Retrying in {backoff_seconds}s..."
            )
            time.sleep(backoff_seconds)

    if results is None and last_error is not None:
        raise last_error

    if not results:
        raise ValueError(f"Geocoding found no match for city name: '{city_name}'")
    match = results[0]
    logger.info(
        f"Geocoded '{city_name}' -> {match['name']}, {match.get('country', '?')} "
        f"(lat={match['latitude']}, lon={match['longitude']})"
    )
    return match["latitude"], match["longitude"]
