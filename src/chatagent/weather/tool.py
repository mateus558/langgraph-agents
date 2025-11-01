# simple_weather_tool.py
from __future__ import annotations

import json
from typing import Optional, Tuple

import requests
from pydantic import BaseModel, Field
from langchain.tools import tool


# -------- helpers mínimos --------
def _parse_latlon(q: str) -> Optional[Tuple[float, float]]:
    if "," not in q:
        return None
    try:
        lat_s, lon_s = q.split(",", 1)
        lat = float(lat_s.strip())
        lon = float(lon_s.strip())
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return lat, lon
    except Exception:
        return None
    return None


def _geocode_city(city: str) -> Tuple[float, float, str, Optional[str]]:
    r = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1, "language": "pt", "format": "json"},
        timeout=10,
    )
    r.raise_for_status()
    data = r.json()
    results = data.get("results") or []
    if not results:
        raise ValueError(f"Local não encontrado: {city!r}")
    r0 = results[0]
    return float(r0["latitude"]), float(r0["longitude"]), r0.get("name", city), r0.get("country_code")


def _current_weather(lat: float, lon: float) -> dict:
    r = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={"latitude": lat, "longitude": lon, "current_weather": "true"},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


# -------- schema + tool --------
class WeatherInput(BaseModel):
    """Cidade ou 'lat,lon' (ex.: 'São Paulo' OU '-23.55,-46.63')."""
    query: str = Field(..., description="City name in Portuguese or coordinates 'lat,lon'")

@tool("weather_current_simple", description="Get current weather. Usage: pass the city name as query (ex: 'Juiz de Fora')", args_schema=WeatherInput)
def weather_tool(query: str) -> str:
    """
    Retorna clima atual (Open-Meteo) como JSON string.
    Uso: passe 'São Paulo' ou '-23.55,-46.63'.
    """
    try:
        latlon = _parse_latlon(query)
        if latlon:
            lat, lon = latlon
            city, country = query, None
        else:
            lat, lon, city, country = _geocode_city(query)

        data = _current_weather(lat, lon)
        cw = data.get("current_weather") or {}
        result = {
            "provider": "open-meteo",
            "city": city,
            "country": country,
            "latitude": data.get("latitude"),
            "longitude": data.get("longitude"),
            "temperature_c": cw.get("temperature"),
            "windspeed_kmh": cw.get("windspeed"),
            "observed_at_iso": cw.get("time"),
        }
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)
