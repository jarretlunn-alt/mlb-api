"""Odds ingestion: fetch from The Odds API, normalize, upsert to fact_game_odds.

Two public entry points:
- ingest_live_odds(client, con)                     — today's upcoming games
- ingest_historical_odds(client, con, start, end)   — backfill by date range

Normalisation matches Odds API team names to game_pks via dim_team + fact_game.
Unmatched events are skipped silently; add entries to _TEAM_NAME_ALIASES when
The Odds API uses a non-standard abbreviation.
"""

from __future__ import annotations

from datetime import date as _date, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .odds_client import OddsApiClient

# Preference order used when choosing which bookmaker's line to store.
_PREFERRED_BOOKS = ["pinnacle", "draftkings", "fanduel", "betmgm"]

# Odds API team name → dim_team.name when they differ.
# Most names match the MLB Stats API exactly; add entries as mismatches appear.
_TEAM_NAME_ALIASES: dict[str, str] = {
    "Arizona D-backs": "Arizona Diamondbacks",
    "D-backs": "Arizona Diamondbacks",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _match_game(con, game_date: str, home_name: str, away_name: str) -> int | None:
    """Return game_pk for the game on game_date with these home/away teams, or None."""
    home = _TEAM_NAME_ALIASES.get(home_name, home_name)
    away = _TEAM_NAME_ALIASES.get(away_name, away_name)
    row = con.execute(
        """
        SELECT fg.game_pk
        FROM fact_game fg
        JOIN dim_team ht ON fg.home_team_id = ht.team_id
        JOIN dim_team awt ON fg.away_team_id = awt.team_id
        WHERE fg.official_date = ?
          AND (ht.name = ? OR ht.abbreviation = ?)
          AND (awt.name = ? OR awt.abbreviation = ?)
        LIMIT 1
        """,
        [game_date, home, home, away, away],
    ).fetchone()
    return row[0] if row else None


def normalize_events(con, events: list[dict]) -> list[dict]:
    """Convert a list of Odds API event dicts to fact_game_odds rows.

    Each output row: {game_pk, sportsbook, home_ml, away_ml, recorded_at}.
    Events that cannot be matched to a game_pk are skipped.
    For each game only the highest-priority available bookmaker is kept.
    """
    rows = []
    for event in events:
        game_date = event["commence_time"][:10]
        home_name = event["home_team"]
        away_name = event["away_team"]
        game_pk = _match_game(con, game_date, home_name, away_name)
        if game_pk is None:
            continue
        for book_key in _PREFERRED_BOOKS:
            book = next(
                (b for b in event.get("bookmakers", []) if b["key"] == book_key),
                None,
            )
            if book is None:
                continue
            market = next(
                (m for m in book.get("markets", []) if m["key"] == "h2h"), None
            )
            if market is None:
                continue
            by_name = {o["name"]: o["price"] for o in market.get("outcomes", [])}
            if home_name not in by_name or away_name not in by_name:
                continue
            rows.append(
                {
                    "game_pk": game_pk,
                    "sportsbook": book_key,
                    "home_ml": int(by_name[home_name]),
                    "away_ml": int(by_name[away_name]),
                    "recorded_at": book.get("last_update", event["commence_time"]),
                }
            )
            break  # use only the best available book per game
    return rows


def upsert_odds(con, rows: list[dict]) -> int:
    """INSERT OR REPLACE rows into fact_game_odds. Returns number of rows written."""
    if not rows:
        return 0
    con.executemany(
        "INSERT OR REPLACE INTO fact_game_odds "
        "(game_pk, sportsbook, home_ml, away_ml, recorded_at) VALUES (?, ?, ?, ?, ?)",
        [
            (r["game_pk"], r["sportsbook"], r["home_ml"], r["away_ml"], r["recorded_at"])
            for r in rows
        ],
    )
    return len(rows)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def ingest_live_odds(client: "OddsApiClient", con) -> dict:
    """Fetch current opening lines and upsert them. Uses ~1 API request."""
    events = client.get_live_odds()
    rows = normalize_events(con, events)
    n = upsert_odds(con, rows)
    return {
        "games_loaded": n,
        "api_remaining": client.remaining_requests,
    }


def ingest_historical_odds(
    client: "OddsApiClient", con, start_date: str, end_date: str
) -> dict:
    """Backfill odds day by day between start_date and end_date (inclusive).

    Uses one API request per day — expensive on the free tier. Schedule the
    snapshot at 18:00 UTC (~2 h before first pitch) to capture opening lines
    before sharp movement.

    Requires a paid Odds API plan for dates older than a few days.
    """
    start = _date.fromisoformat(start_date)
    end = _date.fromisoformat(end_date)
    total_games = 0
    days = 0
    current = start
    while current <= end:
        snapshot = f"{current.isoformat()}T18:00:00Z"
        events = client.get_historical_odds(snapshot)
        rows = normalize_events(con, events)
        total_games += upsert_odds(con, rows)
        days += 1
        current += timedelta(days=1)
    return {
        "days_processed": days,
        "games_loaded": total_games,
        "api_remaining": client.remaining_requests,
    }
