"""Tests for schedule.fetch_schedule and schedule.predict_games.

All network calls are intercepted via FakeClient from conftest.
"""

import json

import pytest

from mlb_pipeline import db, ingest, schedule


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SCHEDULED_SCHEDULE = {
    "totalGames": 2,
    "dates": [
        {
            "date": "2026-09-14",
            "totalGames": 2,
            "games": [
                {
                    "gamePk": 800001,
                    "gameType": "R",
                    "season": "2026",
                    "officialDate": "2026-09-14",
                    "status": {
                        "abstractGameState": "Preview",
                        "codedGameState": "S",
                        "detailedState": "Scheduled",
                    },
                    "teams": {
                        "away": {"team": {"id": 133, "name": "Athletics"}},
                        "home": {"team": {"id": 136, "name": "Seattle Mariners"}},
                    },
                    "venue": {"id": 680, "name": "T-Mobile Park"},
                },
                {
                    "gamePk": 800002,
                    "gameType": "R",
                    "season": "2026",
                    "officialDate": "2026-09-14",
                    "status": {
                        "abstractGameState": "Preview",
                        "codedGameState": "S",
                        "detailedState": "Scheduled",
                    },
                    "teams": {
                        "away": {"team": {"id": 119, "name": "Los Angeles Dodgers"}},
                        "home": {"team": {"id": 137, "name": "San Francisco Giants"}},
                    },
                    "venue": {"id": 2395, "name": "Oracle Park"},
                },
            ],
        }
    ],
}


class _ScheduleOnlyClient:
    """Fake client that returns a fixed schedule and raises on game-detail calls."""

    def __init__(self, payload):
        self._payload = payload
        self.calls = []

    def get_schedule(self, start_date, end_date, sport_id=1):
        self.calls.append(("schedule", start_date, end_date))
        return self._payload

    def get_feed_live(self, game_pk):
        raise AssertionError("fetch_schedule must not call get_feed_live")

    def get_boxscore(self, game_pk):
        raise AssertionError("fetch_schedule must not call get_boxscore")


@pytest.fixture
def scheduled_client():
    return _ScheduleOnlyClient(SCHEDULED_SCHEDULE)


# ---------------------------------------------------------------------------
# fetch_schedule tests
# ---------------------------------------------------------------------------

def test_fetch_schedule_inserts_scheduled_games(scheduled_client, con, settings):
    result = schedule.fetch_schedule(scheduled_client, con, "2026-09-14", "2026-09-14")

    assert result["games_scheduled"] == 2
    rows = con.execute(
        "SELECT game_pk, status, home_score, away_score FROM fact_game ORDER BY game_pk"
    ).fetchall()
    assert len(rows) == 2
    assert rows[0] == (800001, "Scheduled", None, None)
    assert rows[1] == (800002, "Scheduled", None, None)


def test_fetch_schedule_does_not_call_feed_or_boxscore(scheduled_client, con, settings):
    schedule.fetch_schedule(scheduled_client, con, "2026-09-14", "2026-09-14")
    assert all(call[0] == "schedule" for call in scheduled_client.calls)


def test_fetch_schedule_is_idempotent(scheduled_client, con, settings):
    schedule.fetch_schedule(scheduled_client, con, "2026-09-14", "2026-09-14")
    schedule.fetch_schedule(scheduled_client, con, "2026-09-14", "2026-09-14")

    count = con.execute("SELECT count(*) FROM fact_game").fetchone()[0]
    assert count == 2


def test_fetch_schedule_does_not_overwrite_completed_games(
    fake_client, scheduled_client, con, settings, feed_live, boxscore
):
    """A completed game already in fact_game must not be replaced by a Scheduled row."""
    # Insert a completed game via the normal ingest path
    ingest.ingest_date(fake_client, con, settings, "2026-07-01")
    completed_score = con.execute(
        "SELECT home_score FROM fact_game WHERE game_pk = 700001"
    ).fetchone()[0]
    assert completed_score == 5

    # Now try to fetch-schedule the same game_pk as if it were "Scheduled" (simulated)
    fake_scheduled = {
        "dates": [
            {
                "date": "2026-07-01",
                "games": [
                    {
                        "gamePk": 700001,  # same pk as completed game
                        "gameType": "R",
                        "season": "2026",
                        "officialDate": "2026-07-01",
                        "status": {"abstractGameState": "Preview", "codedGameState": "S",
                                   "detailedState": "Scheduled"},
                        "teams": {
                            "away": {"team": {"id": 133}},
                            "home": {"team": {"id": 136}},
                        },
                    }
                ],
            }
        ]
    }
    client = _ScheduleOnlyClient(fake_scheduled)
    schedule.fetch_schedule(client, con, "2026-07-01", "2026-07-01")

    # Score must still be 5 — INSERT OR IGNORE left the completed row alone
    score_after = con.execute(
        "SELECT home_score FROM fact_game WHERE game_pk = 700001"
    ).fetchone()[0]
    assert score_after == 5


# ---------------------------------------------------------------------------
# predict_games tests
# ---------------------------------------------------------------------------

def test_predict_games_returns_predictions(fake_client, con, settings):
    # Seed history so the predictor has team data
    ingest.ingest_date(fake_client, con, settings, "2026-07-01")
    # Seed a scheduled game for the teams that have history
    schedule.fetch_schedule(
        _ScheduleOnlyClient(SCHEDULED_SCHEDULE), con, "2026-09-14", "2026-09-14"
    )

    preds = schedule.predict_games(con, "2026-09-14")

    assert len(preds) == 2
    for p in preds:
        assert "game_pk" in p
        assert 0 < p["home_win_prob"] < 1
        assert 0 < p["away_win_prob"] < 1
        assert abs(p["home_win_prob"] + p["away_win_prob"] - 1.0) < 1e-9


def test_predict_games_saves_to_fact_prediction(fake_client, con, settings):
    ingest.ingest_date(fake_client, con, settings, "2026-07-01")
    schedule.fetch_schedule(
        _ScheduleOnlyClient(SCHEDULED_SCHEDULE), con, "2026-09-14", "2026-09-14"
    )
    schedule.predict_games(con, "2026-09-14")

    count = con.execute("SELECT count(*) FROM fact_prediction").fetchone()[0]
    assert count == 2
    row = con.execute(
        "SELECT model_name, home_win_prob, away_win_prob "
        "FROM fact_prediction WHERE game_pk = 800001"
    ).fetchone()
    assert row[0] == "pythagorean"
    assert 0 < row[1] < 1
    assert abs(row[1] + row[2] - 1.0) < 1e-9


def test_predict_games_is_idempotent(fake_client, con, settings):
    ingest.ingest_date(fake_client, con, settings, "2026-07-01")
    schedule.fetch_schedule(
        _ScheduleOnlyClient(SCHEDULED_SCHEDULE), con, "2026-09-14", "2026-09-14"
    )
    schedule.predict_games(con, "2026-09-14")
    schedule.predict_games(con, "2026-09-14")

    count = con.execute("SELECT count(*) FROM fact_prediction").fetchone()[0]
    assert count == 2


def test_predict_games_no_games_returns_empty(con, settings):
    preds = schedule.predict_games(con, "2099-01-01")
    assert preds == []


def test_predict_games_no_history_falls_back_to_league_average(con, settings):
    """Teams with no fact_team_game rows should still get a 50/50-ish prediction."""
    schedule.fetch_schedule(
        _ScheduleOnlyClient(SCHEDULED_SCHEDULE), con, "2026-09-14", "2026-09-14"
    )
    preds = schedule.predict_games(con, "2026-09-14")

    assert len(preds) == 2
    for p in preds:
        # Without history both teams are at league average; home advantage tips it slightly
        assert 0.5 < p["home_win_prob"] < 0.6


def test_predict_games_features_json_stored(fake_client, con, settings):
    ingest.ingest_date(fake_client, con, settings, "2026-07-01")
    schedule.fetch_schedule(
        _ScheduleOnlyClient(SCHEDULED_SCHEDULE), con, "2026-09-14", "2026-09-14"
    )
    schedule.predict_games(con, "2026-09-14")

    raw = con.execute(
        "SELECT features_json FROM fact_prediction WHERE game_pk = 800001"
    ).fetchone()[0]
    features = json.loads(raw)
    assert "home_pyth" in features
    assert "away_pyth" in features
    assert "home_advantage" in features
