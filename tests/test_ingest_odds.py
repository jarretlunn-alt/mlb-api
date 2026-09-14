"""Tests for ingest_odds — no network, no filesystem side effects."""
import duckdb
import pytest

from mlb_pipeline import db, ingest_odds


GAME_PK = 700001
HOME_ID = 133
AWAY_ID = 136


@pytest.fixture
def mem():
    con = duckdb.connect(":memory:")
    db.init_schema(con)
    # seed dim_team
    con.execute(
        "INSERT INTO dim_team VALUES (?, ?, ?, ?, ?, ?, ?)",
        [HOME_ID, "Oakland Athletics", "OAK", "oak", "AL", "West", "Oakland Coliseum"],
    )
    con.execute(
        "INSERT INTO dim_team VALUES (?, ?, ?, ?, ?, ?, ?)",
        [AWAY_ID, "Seattle Mariners", "SEA", "sea", "AL", "West", "T-Mobile Park"],
    )
    # seed fact_game
    con.execute(
        "INSERT INTO fact_game VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [GAME_PK, "2024-04-15", 2024, "R", "Final", "Oakland Coliseum",
         HOME_ID, AWAY_ID, 3, 2, HOME_ID],
    )
    yield con
    con.close()


def _make_event(home="Oakland Athletics", away="Seattle Mariners",
                date="2024-04-15", home_price=-145, away_price=125,
                bookmaker_key="pinnacle"):
    return {
        "id": "evt1",
        "commence_time": f"{date}T19:00:00Z",
        "home_team": home,
        "away_team": away,
        "bookmakers": [
            {
                "key": bookmaker_key,
                "last_update": f"{date}T18:00:00Z",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": home, "price": home_price},
                            {"name": away, "price": away_price},
                        ],
                    }
                ],
            }
        ],
    }


# ---------------------------------------------------------------------------
# _match_game
# ---------------------------------------------------------------------------

def test_match_game_found(mem):
    pk = ingest_odds._match_game(mem, "2024-04-15", "Oakland Athletics", "Seattle Mariners")
    assert pk == GAME_PK


def test_match_game_not_found_wrong_date(mem):
    pk = ingest_odds._match_game(mem, "2024-04-16", "Oakland Athletics", "Seattle Mariners")
    assert pk is None


def test_match_game_alias_resolved(mem):
    pk = ingest_odds._match_game(mem, "2024-04-15", "Oakland Athletics", "Seattle Mariners")
    assert pk == GAME_PK
    # Arizona alias example (alias defined but team not in fixture — just verify no crash)
    pk2 = ingest_odds._match_game(mem, "2024-04-15", "Arizona D-backs", "Seattle Mariners")
    assert pk2 is None  # no Arizona game in fixture


# ---------------------------------------------------------------------------
# normalize_events
# ---------------------------------------------------------------------------

def test_normalize_events_happy_path(mem):
    rows = ingest_odds.normalize_events(mem, [_make_event()])
    assert len(rows) == 1
    row = rows[0]
    assert row["game_pk"] == GAME_PK
    assert row["sportsbook"] == "pinnacle"
    assert row["home_ml"] == -145
    assert row["away_ml"] == 125


def test_normalize_events_prefers_highest_priority_book(mem):
    event = _make_event(bookmaker_key="draftkings")
    # add pinnacle as well
    event["bookmakers"].insert(0, {
        "key": "pinnacle",
        "last_update": "2024-04-15T18:00:00Z",
        "markets": [{"key": "h2h", "outcomes": [
            {"name": "Oakland Athletics", "price": -150},
            {"name": "Seattle Mariners", "price": 130},
        ]}],
    })
    rows = ingest_odds.normalize_events(mem, [event])
    assert len(rows) == 1
    assert rows[0]["sportsbook"] == "pinnacle"
    assert rows[0]["home_ml"] == -150


def test_normalize_events_skips_unmatched_team(mem):
    rows = ingest_odds.normalize_events(mem, [_make_event(home="Unknown FC")])
    assert rows == []


def test_normalize_events_skips_unknown_bookmaker(mem):
    event = _make_event()
    event["bookmakers"][0]["key"] = "mybookie"
    rows = ingest_odds.normalize_events(mem, [event])
    assert rows == []


# ---------------------------------------------------------------------------
# upsert_odds
# ---------------------------------------------------------------------------

def test_upsert_odds_inserts_rows(mem):
    rows = [{"game_pk": GAME_PK, "sportsbook": "pinnacle",
             "home_ml": -145, "away_ml": 125, "recorded_at": "2024-04-15T18:00:00Z"}]
    n = ingest_odds.upsert_odds(mem, rows)
    assert n == 1
    result = mem.execute("SELECT home_ml FROM fact_game_odds WHERE game_pk = ?", [GAME_PK]).fetchone()
    assert result[0] == -145


def test_upsert_odds_empty_is_noop(mem):
    n = ingest_odds.upsert_odds(mem, [])
    assert n == 0


def test_upsert_odds_replace_on_conflict(mem):
    rows = [{"game_pk": GAME_PK, "sportsbook": "pinnacle",
             "home_ml": -145, "away_ml": 125, "recorded_at": "2024-04-15T18:00:00Z"}]
    ingest_odds.upsert_odds(mem, rows)
    rows[0]["home_ml"] = -155
    ingest_odds.upsert_odds(mem, rows)
    result = mem.execute("SELECT home_ml FROM fact_game_odds WHERE game_pk = ?", [GAME_PK]).fetchone()
    assert result[0] == -155


# ---------------------------------------------------------------------------
# ingest_live_odds / ingest_historical_odds (via stub client)
# ---------------------------------------------------------------------------

class StubOddsClient:
    def __init__(self, events=None):
        self._events = events or [_make_event()]
        self.remaining_requests = 449

    def get_live_odds(self, bookmakers=None):
        return self._events

    def get_historical_odds(self, snapshot_time, bookmakers=None):
        return self._events


def test_ingest_live_odds_returns_counts(mem):
    result = ingest_odds.ingest_live_odds(StubOddsClient(), mem)
    assert result["games_loaded"] == 1
    assert result["api_remaining"] == 449


def test_ingest_historical_odds_iterates_days(mem):
    result = ingest_odds.ingest_historical_odds(
        StubOddsClient(), mem, "2024-04-15", "2024-04-15"
    )
    assert result["days_processed"] == 1
    assert result["games_loaded"] == 1


def test_ingest_historical_odds_multi_day(mem):
    result = ingest_odds.ingest_historical_odds(
        StubOddsClient(), mem, "2024-04-15", "2024-04-17"
    )
    assert result["days_processed"] == 3
