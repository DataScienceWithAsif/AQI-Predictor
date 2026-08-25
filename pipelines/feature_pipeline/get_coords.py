"""
feature_pipeline/geocoding.py

Shared free geocoding helper (Open-Meteo, no API key) — used by both
backfill.py and fetch.py so multi-city support doesn't require looking up
lat/lon by hand for every city, and isn't duplicated across scripts.
"""

import logging

import requests

logger = logging.getLogger("feature_pipeline.geocoding")

REQUEST_TIMEOUT_SECONDS = 30


def geocode_city(city_name: str) -> tuple:
    """
    Resolves a city name to (lat, lon) via Open-Meteo's free geocoding API.
    """
    resp = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city_name, "count": 1},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    results = resp.json().get("results")
    if not results:
        raise ValueError(f"Geocoding found no match for city name: '{city_name}'")
    match = results[0]
    logger.info(
        f"Geocoded '{city_name}' -> {match['name']}, {match.get('country', '?')} "
        f"(lat={match['latitude']}, lon={match['longitude']})"
    )
    return match["latitude"], match["longitude"]