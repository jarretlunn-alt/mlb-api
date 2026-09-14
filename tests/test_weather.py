"""Tests for weather_client, parks, and ingest_weather — no network calls."""
import math

import duckdb
import pytest
import requests

from mlb_pipeline import db, ingest_weather
from mlb_pipeline.parks import PARKS, wind_out_mph
from mlb_pipeline.weather_client import WeatherClient, _extract_hour, _hour_of


# ---------------------------------------------------------------------------
# parks.wind_out_mph
# ---------------------------------------------------------------------------

def test_full_tailwind():
    # park faces N (out_bearing=0); wind from S (180°) → full tailwind
    result = wind_out_mph(wind_from_deg=180, wind_speed_mph=10, out_bearing_deg=0)
    assert math.isclose(result, 10.0, abs_tol=0.01)


def test_full_headwind():
    # park faces N (out_bearing=0); wind from N (0°) → full headwind
    result = wind_out_mph(wind_from_deg=0, wind_speed_mph=10, out_bearing_deg=0)
    assert math.isclose(result, -10.0, abs_tol=0.01)


def test_crosswind_is_zero():
    # park faces N; wind from E (90°) → pure crosswind, zero component
    result = wind_out_mph(wind_from_deg=90, wind_speed_mph=10, out_bearing_deg=0)
    assert math.isclose(result, 0.0, abs_tol=0.01)


def test_wrigley_south_wind_blows_out():
    # Wrigley out_bearing ~75° (ENE); S wind (from 180°) has strong out component
    park = PARKS[112]
    result = wind_out_mph(180, 15, park["out_bearing_deg"])
    assert result > 0  # blowing out (tailwind component)


def test_all_parks_have_required_keys():
    for team_id, park in PARKS.items():
        assert "lat" in park, f"team {team_id} missing lat"
        assert "lon" in park, f"team {team_id} missing lon"
        assert "timezone" in park, f"team {team_id} missing timezone"
        assert "out_bearing_deg" in park, f"team {team_id} missing out_bearing_deg"


# ---------------------------------------------------------------------------
# weather_client internals
# ---------------------------------------------------------------------------

def _make_payload(hours=None, include_precip=False):
    hours = hours or ["2024-04-15T19:00", "2024-04-15T20:00"]
    h = {
        "time":               hours,
        "temperature_2m":     [65.0, 66.0][:len(hours)],
        "windspeed_10m":      [12.0, 13.0][:len(hours)],
        "winddirection_10m":  [180,  185 ][:len(hours)],
    }
    if include_precip:
        h["precipitation_probability"] = [20, 30][:len(hours)]
    return {"hourly": h}


def test_extract_hour_picks_closest():
    payload = _make_payload(["2024-04-15T18:00", "2024-04-15T19:00", "2024-04-15T20:00"],
                             include_precip=True)
    payload["hourly"]["temperature_2m"] = [60.0, 65.0, 70.0]
    payload["hourly"]["windspeed_10m"]  = [10.0, 12.0, 14.0]
    payload["hourly"]["winddirection_10m"] = [170, 180, 190]
    payload["hourly"]["precipitation_probability"] = [10, 20, 30]
    result = _extract_hour(payload, local_hour=19, is_archive=False)
    assert result["temp_f"] == 65.0
    assert result["wind_mph"] == 12.0
    assert result["precip_prob"] == pytest.approx(0.20)


def test_extract_hour_archive_no_precip_prob():
    payload = _make_payload()
    result = _extract_hour(payload, local_hour=19, is_archive=True)
    assert result["precip_prob"] == 0.0


def test_hour_of_parses_correctly():
    assert _hour_of("2024-04-15T19:00") == 19
    assert _hour_of("2024-04-15T07:00") == 7


# ---------------------------------------------------------------------------
# WeatherClient with stub session
# ---------------------------------------------------------------------------

class StubResponse:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


class StubSession:
    def __init__(self, payload):
        self._payload = payload
        self.requests = []

    def get(self, url, params=None, timeout=None):
        self.requests.append({"url": url, "params": params})
        return StubResponse(self._payload)

    def mount(self, *a, **kw):
        pass


def _make_open_meteo_payload(temp=68.0, wind_mph=10.0, wind_deg=180, precip=25):
    return {
        "hourly": {
            "time":               ["2024-04-15T19:00"],
            "temperature_2m":     [temp],
            "windspeed_10m":      [wind_mph],
            "winddirection_10m":  [wind_deg],
            "precipitation_probability": [precip],
        }
    }


def test_weather_client_returns_dict():
    session = StubSession(_make_open_meteo_payload())
    client = WeatherClient(session=session)
    result = client.get_weather(41.948, -87.655, "2024-04-15", timezone="America/Chicago")
    assert "temp_f" in result
    assert "wind_mph" in result
    assert "wind_deg" in result
    assert "precip_prob" in result


def test_weather_client_uses_archive_for_old_dates():
    session = StubSession(_make_open_meteo_payload())
    client = WeatherClient(session=session)
    client.get_weather(41.948, -87.655, "2023-06-01", timezone="America/Chicago")
    assert "archive-api" in session.requests[0]["url"]


def test_weather_client_uses_forecast_for_future():
    import datetime as dt
    future = (dt.date.today() + dt.timedelta(days=2)).isoformat()
    session = StubSession(_make_open_meteo_payload())
    client = WeatherClient(session=session)
    client.get_weather(41.948, -87.655, future, timezone="America/Chicago")
    assert "forecast" in session.requests[0]["url"]


# ---------------------------------------------------------------------------
# ingest_weather
# ---------------------------------------------------------------------------

GAME_PK = 700001
HOME_ID  = 112   # Cubs → Wrigley Field


@pytest.fixture
def mem():
    con = duckdb.connect(":memory:")
    db.init_schema(con)
    con.execute(
        "INSERT INTO fact_game VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [GAME_PK, "2024-04-15", 2024, "R", "Final", "Wrigley Field",
         HOME_ID, 111, 3, 2, HOME_ID],
    )
    yield con
    con.close()


class StubWeatherClient:
    def get_weather(self, lat, lon, game_date, local_hour=19, timezone="UTC"):
        return {"temp_f": 68.0, "wind_mph": 12.0, "wind_deg": 180, "precip_prob": 0.2}


def test_ingest_weather_for_date_inserts_row(mem):
    result = ingest_weather.ingest_weather_for_date(StubWeatherClient(), mem, "2024-04-15")
    assert result["games_loaded"] == 1
    row = mem.execute("SELECT temp_f, wind_out_mph FROM fact_game_weather WHERE game_pk = ?",
                      [GAME_PK]).fetchone()
    assert row is not None
    assert row[0] == pytest.approx(68.0)
    # wind from S (180°), Wrigley out_bearing=75° → blowing out
    assert row[1] > 0


def test_ingest_weather_for_date_no_games(mem):
    result = ingest_weather.ingest_weather_for_date(StubWeatherClient(), mem, "2024-04-16")
    assert result["games_loaded"] == 0


def test_ingest_weather_range(mem):
    result = ingest_weather.ingest_weather_range(StubWeatherClient(), mem, "2024-04-15", "2024-04-15")
    assert result["days_processed"] == 1
    assert result["games_loaded"] == 1


def test_ingest_weather_unknown_park_skipped(mem):
    # Insert a game with a team_id not in PARKS
    mem.execute(
        "INSERT INTO fact_game VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [700002, "2024-04-15", 2024, "R", "Final", "Unknown Park", 9999, 111, 3, 2, 9999],
    )
    result = ingest_weather.ingest_weather_for_date(StubWeatherClient(), mem, "2024-04-15")
    # team 9999 skipped, team 112 loaded
    assert result["games_loaded"] == 1
