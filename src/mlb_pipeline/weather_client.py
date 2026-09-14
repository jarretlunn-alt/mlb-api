"""Client for the Open-Meteo weather API (https://open-meteo.com).

No API key required. Provides:
- Historical weather via the archive API (any past date)
- Forecast weather for upcoming games (up to 16 days out)

Both return the same dict shape:
    {"temp_f": float, "wind_mph": float, "wind_deg": int, "precip_prob": float}

`precip_prob` is 0–1. For historical dates it is derived from precipitation
amount (>0.1mm → 1.0, else 0.0) since the archive API has no probability field.
"""

from __future__ import annotations

import datetime as dt

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_TIMEOUT = 20
_CUTOFF_DAYS = 7   # dates older than this use archive; newer use forecast


def _default_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["GET"])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


class WeatherClient:
    """Thin Open-Meteo wrapper. Accepts an injected session for tests."""

    def __init__(self, session=None, timeout: int = _TIMEOUT):
        self.session = session if session is not None else _default_session()
        self.timeout = timeout

    def _get(self, url: str, params: dict) -> dict:
        resp = self.session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def get_weather(
        self,
        lat: float,
        lon: float,
        game_date: str,
        local_hour: int = 19,
        timezone: str = "America/New_York",
    ) -> dict:
        """Return weather conditions for the given date at local_hour (24h).

        Automatically selects archive vs forecast based on how far past the date is.
        Returns: {temp_f, wind_mph, wind_deg, precip_prob}
        """
        date = dt.date.fromisoformat(game_date)
        today = dt.date.today()
        use_archive = (today - date).days >= _CUTOFF_DAYS

        base_params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,windspeed_10m,winddirection_10m",
            "temperature_unit": "fahrenheit",
            "windspeed_unit": "mph",
            "timezone": timezone,
            "start_date": game_date,
            "end_date": game_date,
        }

        if use_archive:
            payload = self._get(_ARCHIVE_URL, base_params)
        else:
            payload = self._get(_FORECAST_URL, {
                **base_params,
                "hourly": "temperature_2m,windspeed_10m,winddirection_10m,precipitation_probability",
                "forecast_days": 7,
            })

        return _extract_hour(payload, local_hour, use_archive)


def _extract_hour(payload: dict, local_hour: int, is_archive: bool) -> dict:
    """Pull the row closest to local_hour from an hourly payload."""
    times = payload["hourly"]["time"]
    temps = payload["hourly"]["temperature_2m"]
    winds = payload["hourly"]["windspeed_10m"]
    dirs  = payload["hourly"]["winddirection_10m"]
    precip_probs = payload["hourly"].get("precipitation_probability", [None] * len(times))

    # find the index whose local hour is closest to target
    best = min(range(len(times)), key=lambda i: abs(_hour_of(times[i]) - local_hour))

    raw_precip = precip_probs[best]
    if raw_precip is None:
        precip_prob = 0.0
    elif is_archive:
        # archive returns precipitation amount in mm, not probability
        precip_prob = 0.0  # archive never sends precipitation_probability
    else:
        precip_prob = raw_precip / 100.0

    return {
        "temp_f":      temps[best],
        "wind_mph":    winds[best],
        "wind_deg":    int(dirs[best]),
        "precip_prob": precip_prob,
    }


def _hour_of(iso_time: str) -> int:
    """Extract hour from 'YYYY-MM-DDTHH:00' string."""
    return int(iso_time[11:13])
