"""
Offline tests for feature_pipeline/backfill.py — mocks the Open-Meteo HTTP
responses so the parsing/merge/schema logic can be verified without
spending a real API call or needing network access.

Run with:
    python feature_pipeline/test_backfill.py
"""

from datetime import date
from unittest.mock import patch, MagicMock

import pandas as pd

import backfill
import features
import final_features


FAKE_AIR_QUALITY_JSON = {
    "hourly": {
        "time": ["2026-08-01T00:00", "2026-08-01T01:00", "2026-08-01T02:00"],
        "pm2_5": [10.0, 12.0, 11.0],
        "pm10": [20.0, 22.0, 21.0],
        "carbon_monoxide": [200.0, 210.0, 205.0],
        "nitrogen_dioxide": [15.0, 16.0, 14.0],
        "sulphur_dioxide": [3.0, 3.5, 3.2],
        "ozone": [40.0, 42.0, 41.0],
        "us_aqi": [45, 48, 46],
    }
}

FAKE_WEATHER_JSON = {
    "hourly": {
        "time": ["2026-08-01T00:00", "2026-08-01T01:00", "2026-08-01T02:00"],
        "temperature_2m": [25.0, 24.5, 24.0],
        "relative_humidity_2m": [55, 57, 58],
        "pressure_msl": [1009.0, 1009.2, 1009.5],
        "wind_speed_10m": [2.5, 2.7, 2.3],
    }
}


def _mock_response(json_data):
    mock = MagicMock()
    mock.json.return_value = json_data
    mock.raise_for_status = MagicMock()
    return mock


@patch("backfill.requests.get")
def test_fetch_historical_air_quality_parses_correctly(mock_get):
    mock_get.return_value = _mock_response(FAKE_AIR_QUALITY_JSON)
    df = backfill.fetch_historical_air_quality(31.5, 74.3, date(2026, 8, 1), date(2026, 8, 1))

    assert list(df.columns) == ["timestamp", "pm25", "pm10", "co", "no2", "so2", "o3", "aqi"]
    assert len(df) == 3
    assert df.loc[0, "pm25"] == 10.0
    assert df.loc[0, "aqi"] == 45
    print("PASS: fetch_historical_air_quality parses Open-Meteo JSON correctly")


@patch("backfill.requests.get")
def test_fetch_historical_weather_parses_correctly(mock_get):
    mock_get.return_value = _mock_response(FAKE_WEATHER_JSON)
    df = backfill.fetch_historical_weather(31.5, 74.3, date(2026, 8, 1), date(2026, 8, 1))

    assert list(df.columns) == ["timestamp", "temp", "humidity", "pressure", "wind_speed"]
    assert len(df) == 3
    assert df.loc[0, "temp"] == 25.0
    assert df.loc[0, "wind_speed"] == 2.5
    print("PASS: fetch_historical_weather parses Open-Meteo JSON correctly")


@patch("backfill.requests.get")
def test_build_historical_raw_df_merges_to_exact_schema(mock_get):
    # Called in this order inside build_historical_raw_df: air quality, then weather
    mock_get.side_effect = [
        _mock_response(FAKE_AIR_QUALITY_JSON),
        _mock_response(FAKE_WEATHER_JSON),
    ]

    raw_df = backfill.build_historical_raw_df("Lahore", 31.5, 74.3, days=1)

    assert list(raw_df.columns) == features.RAW_COLUMNS
    assert len(raw_df) == 3
    assert (raw_df["city"] == "Lahore").all()
    assert raw_df.isna().sum().sum() == 0
    print("PASS: build_historical_raw_df produces the exact schema features.py expects")


@patch("backfill.requests.get")
def test_build_historical_raw_df_feeds_final_features_without_error(mock_get):
    """Integration check: final_features.compute_features_with_daily_targets()
    (the module Day 5 actually trains against) should run cleanly on
    backfilled data with no schema mismatches."""
    mock_get.side_effect = [
        _mock_response(FAKE_AIR_QUALITY_JSON),
        _mock_response(FAKE_WEATHER_JSON),
    ]

    raw_df = backfill.build_historical_raw_df("Lahore", 31.5, 74.3, days=1)
    features_df, targets_df = final_features.compute_features_with_daily_targets(raw_df)

    assert len(features_df) == 3  # stays hourly granularity
    assert len(targets_df) == 3
    assert "hour" in features_df.columns
    assert "aqi_avg_next_1d" in targets_df.columns
    assert "aqi_avg_next_2d" in targets_df.columns
    assert "aqi_avg_next_3d" in targets_df.columns
    print("PASS: backfilled data flows cleanly into final_features.compute_features_with_daily_targets()")


@patch("backfill.requests.get")
def test_geocode_city_parses_correctly(mock_get):
    mock_get.return_value = _mock_response({
        "results": [{"name": "Lahore", "country": "Pakistan", "latitude": 31.55, "longitude": 74.34}]
    })
    lat, lon = backfill.geocode_city("Lahore")
    assert lat == 31.55
    assert lon == 74.34
    print("PASS: geocode_city parses Open-Meteo geocoding JSON correctly")


@patch("backfill.requests.get")
def test_geocode_city_raises_on_no_match(mock_get):
    mock_get.return_value = _mock_response({"results": []})
    try:
        backfill.geocode_city("Nonexistentville")
        raise AssertionError("Expected ValueError, got no exception")
    except ValueError as e:
        print(f"PASS: geocode_city raises clearly on no match -> {e}")


@patch("backfill.requests.get")
def test_multi_city_backfill_concatenates_correctly(mock_get):
    """Simulates main()'s CITIES loop: geocode + backfill called twice,
    each producing its own city's rows, concatenated into one raw_df."""
    geocode_lahore = _mock_response({
        "results": [{"name": "Lahore", "country": "Pakistan", "latitude": 31.55, "longitude": 74.34}]
    })
    geocode_karachi = _mock_response({
        "results": [{"name": "Karachi", "country": "Pakistan", "latitude": 24.86, "longitude": 67.01}]
    })
    mock_get.side_effect = [
        geocode_lahore,
        _mock_response(FAKE_AIR_QUALITY_JSON), _mock_response(FAKE_WEATHER_JSON),  # Lahore backfill
        geocode_karachi,
        _mock_response(FAKE_AIR_QUALITY_JSON), _mock_response(FAKE_WEATHER_JSON),  # Karachi backfill
    ]

    raw_dfs = []
    for city_name in ["Lahore", "Karachi"]:
        lat, lon = backfill.geocode_city(city_name)
        raw_dfs.append(backfill.build_historical_raw_df(city_name, lat, lon))
    combined = pd.concat(raw_dfs, ignore_index=True)

    assert len(combined) == 6  # 3 hourly rows each
    assert set(combined["city"].unique()) == {"Lahore", "Karachi"}
    assert list(combined.columns) == features.RAW_COLUMNS
    print("PASS: multi-city backfill loop correctly concatenates distinct cities into one raw_df")


def test_fetch_historical_air_quality_rejects_too_old_start_date():
    """The real check this conversation was about: a start date >92 days
    back should fail loudly, not silently return empty/null data."""
    too_old_start = date.today() - pd.Timedelta(days=150)
    too_old_end = date.today() - pd.Timedelta(days=90)
    try:
        backfill.fetch_historical_air_quality(31.5, 74.3, too_old_start, too_old_end)
        raise AssertionError("Expected ValueError for a start date beyond ~92 days back")
    except ValueError as e:
        print(f"PASS: fetch_historical_air_quality rejects an out-of-range start date -> {e}")


if __name__ == "__main__":
    test_fetch_historical_air_quality_parses_correctly()
    test_fetch_historical_weather_parses_correctly()
    test_build_historical_raw_df_merges_to_exact_schema()
    test_build_historical_raw_df_feeds_final_features_without_error()
    test_geocode_city_parses_correctly()
    test_geocode_city_raises_on_no_match()
    test_multi_city_backfill_concatenates_correctly()
    test_fetch_historical_air_quality_rejects_too_old_start_date()
    print("\nAll Day 4 offline tests passed.")