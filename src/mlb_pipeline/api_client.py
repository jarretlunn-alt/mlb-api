"""HTTP client for the public MLB Stats API.

Invariant: every network call in this pipeline goes through this module.
No API key is required for these endpoints.
"""

from __future__ import annotations

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://statsapi.mlb.com"
DEFAULT_TIMEOUT = 30


def _default_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


class MLBApiClient:
    """Thin client over the MLB Stats API endpoints the pipeline uses.

    Accepts an injected session so tests never hit the network.
    """

    def __init__(self, base_url: str = BASE_URL, session=None, timeout: int = DEFAULT_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session if session is not None else _default_session()

    def _get(self, path: str, params: dict | None = None) -> dict:
        response = self.session.get(
            f"{self.base_url}{path}", params=params, timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def get_schedule(self, start_date: str, end_date: str, sport_id: int = 1) -> dict:
        """Schedule for a date range (YYYY-MM-DD). sportId=1 is MLB."""
        return self._get(
            "/api/v1/schedule",
            params={"sportId": sport_id, "startDate": start_date, "endDate": end_date},
        )

    def get_feed_live(self, game_pk: int) -> dict:
        """Full live feed for a game (gameData + liveData)."""
        return self._get(f"/api/v1.1/game/{game_pk}/feed/live")

    def get_boxscore(self, game_pk: int) -> dict:
        """Boxscore with per-player and per-team stats."""
        return self._get(f"/api/v1/game/{game_pk}/boxscore")
