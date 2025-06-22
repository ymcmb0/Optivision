from typing import Optional
import requests

from ..config import get_settings

_weather_map = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog", 51: "Light drizzle",
    53: "Moderate drizzle", 55: "Dense drizzle", 61: "Light rain",
    63: "Moderate rain", 65: "Heavy rain", 71: "Light snow",
    73: "Moderate snow", 75: "Heavy snow", 95: "Thunderstorm",
    96: "Thunderstorm with hail",
}

def current_weather(lat: float, lon: float) -> Optional[str]:
    """Return simple weather description or None on failure."""
    url = get_settings().open_meteo_base_url
    params = {"latitude": lat, "longitude": lon,
              "current": "weather_code", "timezone": "auto"}
    try:
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        code = resp.json()["current"]["weather_code"]
        return _weather_map.get(code, "Unknown")
    except Exception:
        return None
