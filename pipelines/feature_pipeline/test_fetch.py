"""
Offline sanity test for feature_pipeline/fetch.py — mocks the network calls
so it runs without real API keys or internet access.

Run with:
    python feature_pipeline/test_fetch.py
"""

from unittest.mock import patch, MagicMock

import requests
import fetch  # feature_pipeline/fetch.py


FAKE_WEATHER_RESPONSE = {
    "main": {"temp": 32.5, "humidity": 41, "pressure": 1008},
    "wind": {"speed": 3.6},
}

FAKE_POLLUTION_RESPONSE = {
    "list": [{
        "components": {
            "co": 320.5, "no": 1.2, "no2": 18.4, "o3": 55.1,
            "so2": 4.3, "pm2_5": 88.2, "pm10": 120.6, "nh3": 2.1,
        }
    }]
}

FAKE_AQICN_RESPONSE = {
    "status": "ok",
    "data": {
        "aqi": 152,
        "dominentpol": "pm25",
        "city": {"name": "Lahore"},
    },
}


def _mock_response(json_data):
    mock = MagicMock()
    mock.json.return_value = json_data
    mock.raise_for_status = MagicMock()  # no-op = simulates HTTP 200
    return mock


@patch("fetch.SESSION.get")
def test_build_feature_row_happy_path(mock_get):
    # Calls happen in this order inside build_feature_row:
    # 1) weather, 2) pollution, 3) aqicn
    mock_get.side_effect = [
        _mock_response(FAKE_WEATHER_RESPONSE),
        _mock_response(FAKE_POLLUTION_RESPONSE),
        _mock_response(FAKE_AQICN_RESPONSE),
    ]

    row = fetch.build_feature_row(
        city="Lahore", lat=31.55, lon=74.34,
        openweather_key="fake", aqicn_token="fake",
    )

    assert row["city"] == "Lahore"
    assert row["temp"] == 32.5
    assert row["humidity"] == 41
    assert row["wind_speed"] == 3.6
    assert row["pm25"] == 88.2
    assert row["pm10"] == 120.6
    assert row["aqi"] == 152
    assert "timestamp" in row
    assert set(row.keys()) == {
        "timestamp", "city", "temp", "humidity", "pressure", "wind_speed",
        "pm25", "pm10", "co", "no2", "so2", "o3", "aqi",
    }
    print("PASS: build_feature_row happy path ->", row)


@patch("fetch.SESSION.get")
def test_build_feature_row_raises_on_network_failure(mock_get):
    mock_get.side_effect = requests.exceptions.ConnectionError("simulated network failure")

    try:
        fetch.build_feature_row(
            city="Lahore", lat=31.55, lon=74.34,
            openweather_key="fake", aqicn_token="fake",
        )
        raise AssertionError("Expected RuntimeError, got no exception")
    except RuntimeError as e:
        print(f"PASS: build_feature_row correctly raises RuntimeError -> {e}")


@patch("fetch.SESSION.get")
def test_build_feature_row_raises_on_bad_aqicn_status(mock_get):
    mock_get.side_effect = [
        _mock_response(FAKE_WEATHER_RESPONSE),
        _mock_response(FAKE_POLLUTION_RESPONSE),
        _mock_response({"status": "error", "data": "Invalid key"}),
    ]

    try:
        fetch.build_feature_row(
            city="Lahore", lat=31.55, lon=74.34,
            openweather_key="fake", aqicn_token="bad-token",
        )
        raise AssertionError("Expected RuntimeError, got no exception")
    except RuntimeError as e:
        print(f"PASS: build_feature_row correctly raises RuntimeError on bad AQICN status -> {e}")


if __name__ == "__main__":
    test_build_feature_row_happy_path()
    test_build_feature_row_raises_on_network_failure()
    test_build_feature_row_raises_on_bad_aqicn_status()
    print("\nAll Day 2 sanity tests passed.")