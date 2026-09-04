from unittest.mock import patch, MagicMock

import requests

import get_coords


def _mock_response(json_data):
    mock = MagicMock()
    mock.json.return_value = json_data
    mock.raise_for_status = MagicMock()
    return mock


@patch("get_coords.time.sleep")
@patch("get_coords.requests.get")
def test_geocode_city_retries_timeout_then_succeeds(mock_get, mock_sleep):
    mock_get.side_effect = [
        requests.exceptions.ReadTimeout("simulated timeout"),
        _mock_response({
            "results": [{"name": "Lahore", "country": "Pakistan", "latitude": 31.55, "longitude": 74.34}]
        }),
    ]

    lat, lon = get_coords.geocode_city("Lahore")

    assert (lat, lon) == (31.55, 74.34)
    assert mock_get.call_count == 2
    mock_sleep.assert_called_once()


@patch("get_coords.time.sleep")
@patch("get_coords.requests.get")
def test_geocode_city_raises_after_max_retries(mock_get, mock_sleep):
    mock_get.side_effect = requests.exceptions.ConnectionError("simulated connection failure")

    try:
        get_coords.geocode_city("Lahore")
        raise AssertionError("Expected ConnectionError, got no exception")
    except requests.exceptions.ConnectionError:
        pass

    assert mock_get.call_count == get_coords.MAX_RETRIES
    assert mock_sleep.call_count == get_coords.MAX_RETRIES - 1


if __name__ == "__main__":
    test_geocode_city_retries_timeout_then_succeeds()
    test_geocode_city_raises_after_max_retries()
    print("All get_coords retry tests passed.")
