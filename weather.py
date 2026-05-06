import requests

CITY_COORDS = {
    "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.6139, 77.2090),
    "Bangalore": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Hyderabad": (17.3850, 78.4867),
    "Pune": (18.5204, 73.8567),
    "Ahmedabad": (23.0225, 72.5714),
    "Kolkata": (22.5726, 88.3639),
    "Surat": (21.1702, 72.8311),
    "Jaipur": (26.9124, 75.7873),
}

def fetch_weather(city: str) -> dict:
    lat, lon = CITY_COORDS.get(city, (19.0760, 72.8777))
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,precipitation,weathercode,windspeed_10m"
            f"&timezone=Asia%2FKolkata"
        )
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        data = r.json()["current"]

        temp      = data["temperature_2m"]
        humidity  = data["relative_humidity_2m"]
        precip    = data["precipitation"]
        wcode     = data["weathercode"]
        windspeed = data["windspeed_10m"]

        # Map WMO weather code → our 3-category system
        if wcode == 0:
            condition = "Sunny"
        elif wcode in range(1, 50):
            condition = "Humid" if humidity > 70 else "Sunny"
        else:
            condition = "Rainy"

        return {
            "success":   True,
            "city":      city,
            "condition": condition,
            "temp_c":    temp,
            "humidity":  humidity,
            "precip_mm": precip,
            "wind_kmh":  windspeed,
            "wmo_code":  wcode,
        }
    except Exception as e:
        return {"success": False, "error": str(e), "condition": "Sunny"}