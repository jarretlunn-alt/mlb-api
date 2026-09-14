"""Client for The Odds API (https://the-odds-api.com).

Every network call goes through this module. An API key is required; set the
ODDS_API_KEY environment variable or pass api_key= explicitly.

Plan notes:
- Free tier: 500 requests/month, live odds only.
- Historical odds (odds-history endpoint) require a paid plan.
- Rate limit varies by plan; this client does not auto-throttle.

Usage:
    client = OddsApiClient()          # reads ODDS_API_KEY from env
    events = client.get_live_odds()   # returns list of event dicts
"""

from __future__ import annotations

import os

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "baseball_mlb"
DEFAULT_TIMEOUT = 30

# Preference order for bookmakers. Pinnacle sets the sharpest opening line;
# the others are fallbacks for games Pinnacle doesn't price early.
PREFERRED_BOOKS = ["pinnacle", "draftkings", "fanduel", "betmgm"]


def _default_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session


class OddsApiClient:
    """Thin client over The Odds API v4.

    Accepts an injected session so tests never hit the network.
    The `remaining_requests` property reports quota left after the last call.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = BASE_URL,
        session=None,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.api_key = api_key or os.environ.get("ODDS_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "ODDS_API_KEY not set. Export it or pass api_key= explicitly."
            )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session if session is not None else _default_session()
        self._remaining: int | None = None

    def _get(self, path: str, extra_params: dict | None = None) -> dict | list:
        params = {"apiKey": self.api_key, **(extra_params or {})}
        response = self.session.get(
            f"{self.base_url}{path}", params=params, timeout=self.timeout
        )
        response.raise_for_status()
        remaining = response.headers.get("x-requests-remaining")
        if remaining is not None:
            self._remaining = int(remaining)
        return response.json()

    def get_live_odds(
        self, bookmakers: list[str] | None = None
    ) -> list[dict]:
        """Current moneyline odds for all upcoming MLB games.

        Returns a list of event dicts, each with 'home_team', 'away_team',
        'commence_time', and a 'bookmakers' list.
        """
        return self._get(
            f"/sports/{SPORT}/odds",
            extra_params={
                "regions": "us",
                "markets": "h2h",
                "oddsFormat": "american",
                "bookmakers": ",".join(bookmakers or PREFERRED_BOOKS),
            },
        )

    def get_historical_odds(
        self,
        snapshot_time: str,
        bookmakers: list[str] | None = None,
    ) -> list[dict]:
        """Odds snapshot at a specific UTC datetime (ISO-8601, e.g. '2024-04-01T18:00:00Z').

        Pass a time ~2 hours before first pitch to capture opening lines before
        sharp action moves them significantly. Requires a paid Odds API plan.

        Returns a list of event dicts in the same shape as get_live_odds().
        """
        payload = self._get(
            f"/sports/{SPORT}/odds-history",
            extra_params={
                "regions": "us",
                "markets": "h2h",
                "oddsFormat": "american",
                "bookmakers": ",".join(bookmakers or PREFERRED_BOOKS),
                "date": snapshot_time,
            },
        )
        return payload.get("data", []) if isinstance(payload, dict) else payload

    @property
    def remaining_requests(self) -> int | None:
        """API quota remaining after the last call, or None before any call."""
        return self._remaining
