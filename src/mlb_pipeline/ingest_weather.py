"""Weather ingestion: fetch from Open-Meteo, upsert to fact_game_weather.

Entry points:
- ingest_weather_for_date(client, con, game_date)  — all games on a date
- ingest_weather_range(client, con, start_date, end_date)  — date range backfill
"""

from __future__ import annotations

import datetime as dt
from datetime import timezone as _tz
from typing import TYPE_CHECKING

from .parks import PARKS, wind_out_mph as _wind_out

if TYPE_CHECKING:
    from .weather_client import WeatherClient


def _fetch_row(client: "WeatherClient", con, game_pk: int, home_team_id: int, game_date: str) -> dict | None:
    """Fetch weather for one game; return a fact_game_weather row dict or None."""
    park = PARKS.get(home_team_id)
    if park is None:
        return None

    try:
        w = client.get_weather(
            lat=park["lat"],
            lon=park["lon"],
            game_date=game_date,
            local_hour=19,
            timezone=park["timezone"],
        )
    except Exception:
        return None

    out_bearing = park["out_bearing_deg"]
    if out_bearing is not None:
        w_out = _wind_out(w["wind_deg"], w["wind_mph"], out_bearing)
    else:
        w_out = 0.0  # dome / retractable — no wind effect

    return {
        "game_pk":      game_pk,
        "temp_f":       w["temp_f"],
        "wind_mph":     w["wind_mph"],
        "wind_deg":     w["wind_deg"],
        "wind_out_mph": w_out,
        "precip_prob":  w["precip_prob"],
        "fetched_at":   dt.datetime.now(_tz.utc).isoformat(),
    }


def ingest_weather_for_date(client: "WeatherClient", con, game_date: str) -> dict:
    """Fetch and upsert weather for all games on game_date."""
    games = con.execute(
        "SELECT game_pk, home_team_id FROM fact_game WHERE official_date = ?",
        [game_date],
    ).fetchall()

    rows = []
    for game_pk, home_team_id in games:
        row = _fetch_row(client, con, game_pk, home_team_id, game_date)
        if row is not None:
            rows.append(row)

    _upsert(con, rows)
    return {"date": game_date, "games_loaded": len(rows)}


def ingest_weather_range(client: "WeatherClient", con, start_date: str, end_date: str) -> dict:
    """Backfill weather for all games in a date range."""
    start = dt.date.fromisoformat(start_date)
    end   = dt.date.fromisoformat(end_date)
    total = 0
    current = start
    while current <= end:
        result = ingest_weather_for_date(client, con, current.isoformat())
        total += result["games_loaded"]
        current += dt.timedelta(days=1)
    return {"days_processed": (end - start).days + 1, "games_loaded": total}


def _upsert(con, rows: list[dict]) -> None:
    if not rows:
        return
    con.executemany(
        "INSERT OR REPLACE INTO fact_game_weather "
        "(game_pk, temp_f, wind_mph, wind_deg, wind_out_mph, precip_prob, fetched_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (r["game_pk"], r["temp_f"], r["wind_mph"], r["wind_deg"],
             r["wind_out_mph"], r["precip_prob"], r["fetched_at"])
            for r in rows
        ],
    )
