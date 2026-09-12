import pytest
import requests

from mlb_pipeline.api_client import MLBApiClient


class StubResponse:
    def __init__(self, payload=None, status_code=200):
        self._payload = payload if payload is not None else {}
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


class StubSession:
    def __init__(self, response=None):
        self.response = response or StubResponse({"ok": True})
        self.requests = []

    def get(self, url, params=None, timeout=None):
        self.requests.append({"url": url, "params": params, "timeout": timeout})
        return self.response


def test_get_schedule_builds_url_and_params():
    session = StubSession()
    client = MLBApiClient(session=session)

    result = client.get_schedule("2026-07-01", "2026-07-02")

    assert result == {"ok": True}
    request = session.requests[0]
    assert request["url"] == "https://statsapi.mlb.com/api/v1/schedule"
    assert request["params"] == {
        "sportId": 1,
        "startDate": "2026-07-01",
        "endDate": "2026-07-02",
    }


def test_get_feed_live_uses_v1_1_endpoint():
    session = StubSession()
    client = MLBApiClient(session=session)

    client.get_feed_live(700001)

    assert session.requests[0]["url"] == "https://statsapi.mlb.com/api/v1.1/game/700001/feed/live"


def test_get_boxscore_url():
    session = StubSession()
    client = MLBApiClient(session=session)

    client.get_boxscore(700001)

    assert session.requests[0]["url"] == "https://statsapi.mlb.com/api/v1/game/700001/boxscore"


def test_http_error_raises():
    session = StubSession(StubResponse(status_code=500))
    client = MLBApiClient(session=session)

    with pytest.raises(requests.HTTPError):
        client.get_schedule("2026-07-01", "2026-07-01")
