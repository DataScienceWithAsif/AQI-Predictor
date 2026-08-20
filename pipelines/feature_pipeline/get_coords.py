import requests
import os
from dotenv import load_dotenv
load_dotenv()

open_weather_api_key = os.getenv("OPENWEATHER_KEY")


def geocode_city(city_name, api_key):
    url = "http://api.openweathermap.org/geo/1.0/direct"
    params = {"q": city_name, "limit": 1, "appid": api_key}
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        raise ValueError(f"No location found for '{city_name}'. Try a different format, e.g. 'City,CountryCode'.")
    return data[0]["lat"], data[0]["lon"], data[0]["name"], data[0]["country"]

if __name__ == "__main__":
    CITY_NAME = "Lahore,PK" 
    lat, lon, resolved_name, country = geocode_city(CITY_NAME, open_weather_api_key)
    print(f"Resolved: {resolved_name}, {country}  ->  lat={lat}, lon={lon}")
