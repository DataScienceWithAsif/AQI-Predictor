"""
Offline sanity test for feature_pipeline/fetch.py — mocks the network calls
so it runs without real API keys or internet access.

Run with:
    python feature_pipeline/test_fetch.py
"""

from unittest.mock import patch, MagicMock

import requests
import pandas as pd
import fetch  # feature_pipeline/fetch.py
import hopsworks_io  # feature_pipeline/hopsworks_io.py


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


@patch("fetch.geocode_city")
def test_get_cities_config_uses_cities_env_when_set(mock_geocode):
    mock_geocode.side_effect = [(31.5, 74.3), (24.9, 67.0)]
    with patch.dict("os.environ", {"CITIES": "Lahore, Karachi"}):
        configs = fetch.get_cities_config()

    assert len(configs) == 2
    assert configs[0] == {"city_name": "Lahore", "lat": 31.5, "lon": 74.3}
    assert configs[1] == {"city_name": "Karachi", "lat": 24.9, "lon": 67.0}
    print("PASS: get_cities_config() parses CITIES and geocodes each name")


def test_get_cities_config_falls_back_to_single_city():
    with patch.dict("os.environ", {"CITY_NAME": "Islamabad", "LAT": "33.6", "LON": "73.0"}, clear=False):
        # Ensure CITIES isn't accidentally set from a prior test/env
        os_environ_backup = dict(__import__("os").environ)
        import os as _os
        _os.environ.pop("CITIES", None)
        configs = fetch.get_cities_config()

    assert configs == [{"city_name": "Islamabad", "lat": 33.6, "lon": 73.0}]
    print("PASS: get_cities_config() falls back to single CITY_NAME/LAT/LON when CITIES isn't set")


@patch("fetch.build_feature_row")
def test_main_saves_successful_cities_even_if_one_fails(mock_build_row, tmp_path, monkeypatch):
    """The important behavior from this conversation: one city's station
    being down should not cost you the other cities' data for that hour,
    but the run should still end in failure so it stays visible."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENWEATHER_KEY", "fake")
    monkeypatch.setenv("AQICN_TOKEN", "fake")
    monkeypatch.setenv("CITIES", "Lahore,Karachi,Islamabad")
    monkeypatch.delenv("CITY_NAME", raising=False)

    def fake_build_row(city, lat, lon, openweather_key, aqicn_token):
        if city == "Karachi":
            raise RuntimeError("simulated AQICN station outage")
        return {"timestamp": "2026-08-24T00:00:00", "city": city, "aqi": 100}

    mock_build_row.side_effect = fake_build_row

    with patch("fetch.geocode_city", return_value=(0.0, 0.0)):
        try:
            fetch.main()
            raise AssertionError("Expected RuntimeError since one city failed")
        except RuntimeError as e:
            assert "Karachi" in str(e)

    csv_path = tmp_path / "day2_feature_pipeline_log.csv"
    assert csv_path.exists(), "Successful cities' rows should still be saved despite the one failure"
    saved = pd.read_csv(csv_path)
    assert set(saved["city"]) == {"Lahore", "Islamabad"}
    assert "Karachi" not in set(saved["city"])
    print("PASS: Lahore/Islamabad rows saved despite Karachi failing; run still raised (visible failure)")


@patch("hopsworks_io.insert_raw_hourly")
@patch("fetch.build_feature_row")
def test_main_pushes_to_hopsworks_when_api_key_set(mock_build_row, mock_insert_raw, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENWEATHER_KEY", "fake")
    monkeypatch.setenv("AQICN_TOKEN", "fake")
    monkeypatch.setenv("HOPSWORKS_API_KEY", "fake-hopsworks-key")
    monkeypatch.setenv("CITIES", "Lahore,Karachi")
    monkeypatch.delenv("CITY_NAME", raising=False)

    mock_build_row.side_effect = lambda city, lat, lon, openweather_key, aqicn_token: {
        "timestamp": "2026-08-24T00:00:00", "city": city, "aqi": 100,
    }

    with patch("fetch.geocode_city", return_value=(0.0, 0.0)):
        fetch.main()

    assert mock_insert_raw.called, "insert_raw_hourly should be called when HOPSWORKS_API_KEY is set"
    pushed_df = mock_insert_raw.call_args[0][0]
    assert set(pushed_df["city"]) == {"Lahore", "Karachi"}
    print("PASS: main() pushes fetched rows to Hopsworks (aqi_raw_hourly) when HOPSWORKS_API_KEY is set")


@patch("hopsworks_io.insert_raw_hourly")
@patch("fetch.build_feature_row")
def test_main_skips_hopsworks_push_without_api_key(mock_build_row, mock_insert_raw, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENWEATHER_KEY", "fake")
    monkeypatch.setenv("AQICN_TOKEN", "fake")
    monkeypatch.delenv("HOPSWORKS_API_KEY", raising=False)
    monkeypatch.setenv("CITIES", "Lahore")
    monkeypatch.delenv("CITY_NAME", raising=False)

    mock_build_row.return_value = {"timestamp": "2026-08-24T00:00:00", "city": "Lahore", "aqi": 100}

    with patch("fetch.geocode_city", return_value=(0.0, 0.0)):
        fetch.main()  # should complete without raising

    assert not mock_insert_raw.called, "insert_raw_hourly should NOT be called without HOPSWORKS_API_KEY"
    assert (tmp_path / "day2_feature_pipeline_log.csv").exists(), "local CSV should still be written"
    print("PASS: main() skips the Hopsworks push (local-only run) when HOPSWORKS_API_KEY is unset")


@patch("hopsworks_io.insert_raw_hourly")
@patch("fetch.build_feature_row")
def test_main_raises_but_keeps_local_csv_if_hopsworks_push_fails(mock_build_row, mock_insert_raw, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENWEATHER_KEY", "fake")
    monkeypatch.setenv("AQICN_TOKEN", "fake")
    monkeypatch.setenv("HOPSWORKS_API_KEY", "fake-hopsworks-key")
    monkeypatch.setenv("CITIES", "Lahore")
    monkeypatch.delenv("CITY_NAME", raising=False)

    mock_build_row.return_value = {"timestamp": "2026-08-24T00:00:00", "city": "Lahore", "aqi": 100}
    mock_insert_raw.side_effect = RuntimeError("simulated Hopsworks outage")

    with patch("fetch.geocode_city", return_value=(0.0, 0.0)):
        try:
            fetch.main()
            raise AssertionError("Expected RuntimeError since the Hopsworks push failed")
        except RuntimeError as e:
            assert "Hopsworks" in str(e)

    assert (tmp_path / "day2_feature_pipeline_log.csv").exists(), (
        "local CSV should be saved even if the downstream Hopsworks push fails"
    )
    print("PASS: a Hopsworks push failure still raises (visible in Actions) without losing the local CSV")


if __name__ == "__main__":
    test_build_feature_row_happy_path()
    test_build_feature_row_raises_on_network_failure()
    test_build_feature_row_raises_on_bad_aqicn_status()
    test_get_cities_config_uses_cities_env_when_set()
    test_get_cities_config_falls_back_to_single_city()
    print("\n(Run the remaining tests via pytest — they need tmp_path/monkeypatch fixtures)")
    print("\nAll Day 2 sanity tests passed.")