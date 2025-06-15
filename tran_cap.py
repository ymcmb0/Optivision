import requests

# Example: Lahore, Pakistan
latitude = 31.5204
longitude = 74.3587

url = f"https://api.open-meteo.com/v1/forecast"
params = {
    'latitude': latitude,
    'longitude': longitude,
    'current': 'weather_code',
    'timezone': 'auto'
}

# Weather code to description map (simplified)
weather_code_map = {
    0: 'Clear sky',
    1: 'Mainly clear',
    2: 'Partly cloudy',
    3: 'Overcast',
    45: 'Fog',
    48: 'Depositing rime fog',
    51: 'Light drizzle',
    53: 'Moderate drizzle',
    55: 'Dense drizzle',
    61: 'Light rain',
    63: 'Moderate rain',
    65: 'Heavy rain',
    71: 'Light snow',
    73: 'Moderate snow',
    75: 'Heavy snow',
    95: 'Thunderstorm',
    96: 'Thunderstorm with hail'
}

response = requests.get(url, params=params)

if response.status_code == 200:
    data = response.json()
    weather_code = data['current']['weather_code']
    weather_type = weather_code_map.get(weather_code, "Unknown")
    print(f"🌦️ Current weather: {weather_type}")
else:
    print(f"❌ Failed to fetch weather data: {response.status_code}")
