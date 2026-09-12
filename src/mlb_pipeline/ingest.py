"""Ingestion orchestration: fetch -> save raw -> normalize -> upsert.

Idempotent by construction: raw files are keyed by date/gamePk and DB loads
use INSERT OR REPLACE on primary keys, so re-running a date is safe.
"""

from __future__ import annotations

import datetime as dt

from . import db, normalize, raw_store
from .config import Settings


def date_range(start_date: str, end_date: str):
    """Yield YYYY-MM-DD strings from start_date to end_date inclusive."""
    start = dt.date.fromisoformat(start_date)
    end = dt.date.fromisoformat(end_date)
    if end < start:
        raise ValueError(f"end_date {end_date} is before start_date {start_date}")
    current = start
    while current <= end:
        yield current.isoformat()
        current += dt.timedelta(days=1)


def load_game(con, feed: dict, boxscore: dict) -> None:
    """Upsert one game's rows into all warehouse tables."""
    game_pk = feed["gamePk"]
    db.upsert(con, "dim_team", normalize.normalize_teams(feed))
    db.upsert(con, "dim_player", normalize.normalize_players(feed))
    db.upsert(con, "fact_game", [normalize.normalize_game(feed)])
    db.upsert(con, "fact_team_game", normalize.normalize_team_games(game_pk, boxscore))
    db.upsert(con, "fact_player_game_batting", normalize.normalize_batting(game_pk, boxscore))
    db.upsert(con, "fact_player_game_pitching", normalize.normalize_pitching(game_pk, boxscore))


def ingest_date(client, con, settings: Settings, date_str: str) -> dict:
    """Ingest all completed games for one date."""
    schedule = client.get_schedule(date_str, date_str)
    raw_store.save_raw(schedule, raw_store.raw_path(settings.raw_dir, "schedule", date_str))

    game_pks = normalize.extract_game_pks(schedule, only_final=True)
    for game_pk in game_pks:
        feed = client.get_feed_live(game_pk)
        raw_store.save_raw(feed, raw_store.raw_path(settings.raw_dir, "feed_live", game_pk))
        boxscore = client.get_boxscore(game_pk)
        raw_store.save_raw(boxscore, raw_store.raw_path(settings.raw_dir, "boxscore", game_pk))
        load_game(con, feed, boxscore)

    return {"date": date_str, "games_loaded": len(game_pks)}


def ingest_date_range(client, con, settings: Settings, start_date: str, end_date: str) -> list[dict]:
    return [ingest_date(client, con, settings, d) for d in date_range(start_date, end_date)]
