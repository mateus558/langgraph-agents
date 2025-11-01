# weather_provider.py
from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import httpx


# ===============================
# Models and errors
# ===============================

@dataclass(frozen=True)
class WeatherCurrent:
    provider: str
    city: str
    country: Optional[str]
    latitude: float
    longitude: float
    temperature_c: float
    wind_kmh: Optional[float]
    description: Optional[str]  # short text description (Portuguese if available)
    observed_at_iso: Optional[str]  # ISO-8601 timestamp if provided by the provider


class WeatherError(Exception):
    """Generic weather provider error."""


# ===============================
# Utilities
# ===============================

def _is_latlon(text: str) -> Optional[Tuple[float, float]]:
    """Detect a 'lat,lon' pattern in the city input."""
    try:
        if "," not in text:
            return None
        lat_s, lon_s = text.split(",", 1)
        lat = float(lat_s.strip())
        lon = float(lon_s.strip())
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return (lat, lon)
    except Exception:
        return None
    return None


def _to_optional_float(x: Any) -> Optional[float]:
    """Safely coerce to float, returning None when value is missing or invalid.

    Accepts int/float/str values; returns None for None or non-convertible types.
    """
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _to_float_or(x: Any, default: float = 0.0) -> float:
    """Safely coerce to float, falling back to a default on failure."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


async def _with_retries(coro_factory, retries: int = 3, base_sleep: float = 0.3):
    """
    Simple exponential retry helper for HTTP calls.

    coro_factory: zero-argument callable that returns a coroutine (so the request
    can be recreated on each attempt).
    """
    # track last exception (optional)
    for attempt in range(retries):
        try:
            return await coro_factory()
        except (httpx.TimeoutException, httpx.TransportError, httpx.RemoteProtocolError):
            await asyncio.sleep(base_sleep * (2 ** attempt))
    # last attempt
    return await coro_factory()


# ===============================
# HTTP clients
# ===============================

_DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0)
_DEFAULT_HEADERS = {"User-Agent": "weather-provider/1.0 (+https://example.local)"}


class OpenMeteoClient:
    GEO_BASE = "https://geocoding-api.open-meteo.com/v1/search"
    WX_BASE = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def geocode(self, query: str) -> Tuple[float, float, str, Optional[str]]:
        """
        Return (lat, lon, city_display, country_code).

        Uses Open-Meteo geocoding. No API key required.
        """
        latlon = _is_latlon(query)
        if latlon:
            lat, lon = latlon
            return lat, lon, query.strip(), None

        params = {"name": query, "count": 1, "language": "pt", "format": "json"}
        async def _do():
            return await self.client.get(self.GEO_BASE, params=params, timeout=_DEFAULT_TIMEOUT, headers=_DEFAULT_HEADERS)

        resp = await _with_retries(_do)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results") or []
        if not results:
            raise WeatherError(f"Location not found: {query!r}")
        r0 = results[0]
        lat = float(r0["latitude"])
        lon = float(r0["longitude"])
        city = r0.get("name") or query
        country = r0.get("country_code")
        return lat, lon, city, country

    async def current(self, lat: float, lon: float) -> WeatherCurrent:
        params = {
            "latitude": lat,
            "longitude": lon,
            "current_weather": "true",
        }

        async def _do():
            return await self.client.get(self.WX_BASE, params=params, timeout=_DEFAULT_TIMEOUT, headers=_DEFAULT_HEADERS)

        resp = await _with_retries(_do)
        resp.raise_for_status()
        data = resp.json()
        cw = data.get("current_weather")
        if not cw:
            raise WeatherError("Open-Meteo response missing 'current_weather'.")

        # Open-Meteo does not provide city/country here — geocode provides that
        return WeatherCurrent(
            provider="open-meteo",
            city="",
            country=None,
            latitude=float(data["latitude"]),
            longitude=float(data["longitude"]),
            temperature_c=float(cw["temperature"]),
            wind_kmh=_to_optional_float(cw.get("windspeed")),
            description=None,  # Open-Meteo does not provide a text description
            observed_at_iso=cw.get("time"),
        )


class WeatherAPIClient:
    BASE = "https://api.weatherapi.com/v1/current.json"

    def __init__(self, client: httpx.AsyncClient, api_key: str):
        self.client = client
        self.api_key = api_key

    async def current_by_query(self, query: str, lang: str = "pt") -> WeatherCurrent:
        params = {
            "key": self.api_key,
            "q": query,
            "lang": lang,
        }

        async def _do():
            return await self.client.get(self.BASE, params=params, timeout=_DEFAULT_TIMEOUT, headers=_DEFAULT_HEADERS)
        resp = await _with_retries(_do)
        resp.raise_for_status()
        data = resp.json()

        loc = data.get("location") or {}
        cur = data.get("current") or {}
        cond = (cur.get("condition") or {})
        return WeatherCurrent(
            provider="weatherapi",
            city=str(loc.get("name") or query),
            country=str(loc.get("country") or None) or None,
            latitude=float(loc.get("lat") or 0.0),
            longitude=float(loc.get("lon") or 0.0),
            temperature_c=_to_float_or(cur.get("temp_c"), 0.0),
            wind_kmh=_to_optional_float(cur.get("wind_kph")),
            description=str(cond.get("text") or None) or None,
            observed_at_iso=str(cur.get("last_updated") or None) or None,
        )


# ===============================
# Simple TTL cache (async-safe)
# ===============================

class _TTLCache:
    def __init__(self, ttl_seconds: int = 120):
        self.ttl = ttl_seconds
        self._store: Dict[str, Tuple[float, Any]] = {}

    def get(self, key: str) -> Optional[Any]:
        item = self._store.get(key)
        if not item:
            return None
        ts, val = item
        if time.time() - ts > self.ttl:
            self._store.pop(key, None)
            return None
        return val

    def set(self, key: str, value: Any) -> None:
        self._store[key] = (time.time(), value)


# ===============================
# Unified provider
# ===============================


class WeatherProvider:
    """Unified weather provider.

    Uses Open-Meteo by default (no API key). If WEATHERAPI_KEY is set, the
    WeatherAPI provider will be used as a fallback when needed.
    """

    def __init__(self, weatherapi_key: Optional[str] = None, cache_ttl_seconds: int = 120):
        self.weatherapi_key = weatherapi_key or os.getenv("WEATHERAPI_KEY")
        self._cache = _TTLCache(ttl_seconds=cache_ttl_seconds)
        self._client = httpx.AsyncClient(http2=True, timeout=_DEFAULT_TIMEOUT, headers=_DEFAULT_HEADERS)
        self._openmeteo = OpenMeteoClient(self._client)
        self._weatherapi = WeatherAPIClient(self._client, self.weatherapi_key) if self.weatherapi_key else None

    async def aclose(self):
        await self._client.aclose()

    async def get_current(self, city_or_latlon: str) -> WeatherCurrent:
        key = f"current:{city_or_latlon.strip().lower()}"
        cached = self._cache.get(key)
        if cached:
            return cached

        # 1) Try Open-Meteo
        try:
            lat, lon, city_disp, country = await self._openmeteo.geocode(city_or_latlon)
            wx = await self._openmeteo.current(lat, lon)
            wx = WeatherCurrent(
                provider=wx.provider,
                city=city_disp,
                country=country,
                latitude=wx.latitude,
                longitude=wx.longitude,
                temperature_c=wx.temperature_c,
                wind_kmh=wx.wind_kmh,
                description=wx.description,
                observed_at_iso=wx.observed_at_iso,
            )
            self._cache.set(key, wx)
            return wx
        except Exception as e_open:
            # 2) Fallback to WeatherAPI (if available)
            if self._weatherapi:
                try:
                    wx = await self._weatherapi.current_by_query(city_or_latlon, lang="pt")
                    self._cache.set(key, wx)
                    return wx
                except Exception as e_wx:
                    raise WeatherError(f"Open-Meteo failure ({e_open}) and WeatherAPI ({e_wx}).") from e_wx
            raise WeatherError(f"Failed to fetch weather from Open-Meteo: {e_open}") from e_open


# ===============================
# Example usage (CLI)
# ===============================

async def _demo():
    import argparse

    parser = argparse.ArgumentParser(description="WeatherProvider demo")
    parser.add_argument("query", help="City name or 'lat,lon' (e.g. 'São Paulo' or '-23.55,-46.63')")
    parser.add_argument("--once", action="store_true", help="Single request and exit")
    args = parser.parse_args()

    provider = WeatherProvider()  # reads WEATHERAPI_KEY from environment automatically
    try:
        wx = await provider.get_current(args.query)
        # Friendly output
        wind = f", wind: {wx.wind_kmh:.1f} km/h" if wx.wind_kmh is not None else ""
        desc = f" — {wx.description}" if wx.description else ""
        print(
            f"[{wx.provider}] {wx.city} ({wx.country or '??'}) "
            f"{wx.latitude:.4f},{wx.longitude:.4f} | {wx.temperature_c:.1f} °C{wind}{desc} "
            f"| obs: {wx.observed_at_iso or '-'}"
        )
        if args.once:
            return

        # Demonstrate TTL cache effect
        wx2 = await provider.get_current(args.query)
        print("(cache) repeated:", wx2 == wx)

    finally:
        await provider.aclose()


if __name__ == "__main__":
    asyncio.run(_demo())
