"""
Offline tests for feature_pipeline/backfill.py — mocks the Open-Meteo HTTP
responses so the parsing/merge/schema logic can be verified without
spending a real API call or needing network access.

Run with:
    python feature_pipeline/test_backfill.py
"""

import os
from datetime import date
from unittest.mock import patch, MagicMock

import backfill
import features
import final_features
import hopsworks_io


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


def test_ensure_hopsworks_cache_dir_handles_windows_temp_paths():
    with patch.dict(os.environ, {}, clear=True):
        home = hopsworks_io._ensure_hopsworks_cache_dir(platform_name="nt")
        assert home.endswith("hopsworks")
        assert os.environ["HOPSWORKS_HOME"] == home
        assert os.environ["TEMP"] == home
        assert os.environ["TMP"] == home
        assert os.environ["TMPDIR"] == home

    with patch.dict(os.environ, {"HOPSWORKS_HOME": "/tmp\eu-west.cloud.hopsworks.ai"}, clear=True):
        home = hopsworks_io._ensure_hopsworks_cache_dir(platform_name="nt")
        assert home.startswith(os.path.expanduser("~"))
        assert "hopsworks" in home
        assert os.environ["HOPSWORKS_HOME"] == home


if __name__ == "__main__":
    test_fetch_historical_air_quality_parses_correctly()
    test_fetch_historical_weather_parses_correctly()
    test_build_historical_raw_df_merges_to_exact_schema()
    test_build_historical_raw_df_feeds_final_features_without_error()
    test_ensure_hopsworks_cache_dir_handles_windows_temp_paths()
    print("\nAll Day 4 offline tests passed.")