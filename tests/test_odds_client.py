"""Tests for OddsApiClient — all network calls stubbed via session injection."""
import pytest
import requests

from mlb_pipeline.odds_client import OddsApiClient


class StubResponse:
    def __init__(self, payload=None, status_code=200, remaining="450"):
        self._payload = payload if payload is not None else []
        self.status_code = status_code
        self.headers = {}
        if remaining is not None:
            self.headers["x-requests-remaining"] = remaining

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


class StubSession:
    def __init__(self, response=None):
        self.response = response or StubResponse()
        self.requests = []

    def get(self, url, params=None, timeout=None):
        self.requests.append({"url": url, "params": params, "timeout": timeout})
        return self.response


def make_client(payload=None, remaining="450"):
    session = StubSession(StubResponse(payload=payload, remaining=remaining))
    client = OddsApiClient(api_key="test-key", session=session)
    return client, session


def test_get_live_odds_returns_list():
    events = [{"id": "abc", "home_team": "NYY", "away_team": "BOS"}]
    client, session = make_client(payload=events)
    result = client.get_live_odds()
    assert result == events
    assert len(session.requests) == 1
    req = session.requests[0]
    assert "/sports/baseball_mlb/odds" in req["url"]
    assert req["params"]["oddsFormat"] == "american"
    assert req["params"]["markets"] == "h2h"


def test_get_live_odds_includes_api_key():
    client, session = make_client()
    client.get_live_odds()
    assert session.requests[0]["params"]["apiKey"] == "test-key"


def test_remaining_requests_updated_after_call():
    client, _ = make_client(remaining="123")
    assert client.remaining_requests is None
    client.get_live_odds()
    assert client.remaining_requests == 123


def test_remaining_requests_none_when_header_absent():
    session = StubSession(StubResponse(remaining=None))
    client = OddsApiClient(api_key="test-key", session=session)
    client.get_live_odds()
    assert client.remaining_requests is None


def test_get_historical_odds_uses_correct_endpoint():
    payload = {"data": [{"id": "xyz"}], "timestamp": "2024-04-01T18:00:00Z"}
    client, session = make_client(payload=payload)
    result = client.get_historical_odds("2024-04-01T18:00:00Z")
    assert result == [{"id": "xyz"}]
    req = session.requests[0]
    assert "odds-history" in req["url"]
    assert req["params"]["date"] == "2024-04-01T18:00:00Z"


def test_get_historical_odds_handles_bare_list():
    events = [{"id": "a"}]
    client, _ = make_client(payload=events)
    result = client.get_historical_odds("2024-04-01T18:00:00Z")
    assert result == events


def test_missing_api_key_raises():
    with pytest.raises(ValueError, match="ODDS_API_KEY"):
        OddsApiClient(api_key="")


def test_http_error_propagates():
    session = StubSession(StubResponse(status_code=429))
    client = OddsApiClient(api_key="test-key", session=session)
    with pytest.raises(requests.HTTPError):
        client.get_live_odds()
