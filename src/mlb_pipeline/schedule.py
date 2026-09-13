"""Upcoming-schedule ingest and Pythagorean win-probability predictions.

fetch_schedule  — upserts scheduled (not-yet-played) games into fact_game
                  using INSERT OR IGNORE so completed rows are never touched.
predict_games   — generates GamePrediction rows for all Scheduled games on a
                  given date using a Log5 / Pythagorean model derived from
                  season-to-date fact_team_game records.
"""

from __future__ import annotations

import datetime as dt
import json
import math

from . import db, normalize
from .config import Settings

HOME_ADVANTAGE = 0.03     # flat +3% for home team before Log5
PYTH_EXP = 1.83
LEAGUE_AVG_RS = 4.5       # fallback when a team has no history
MODEL_NAME = "pythagorean"


# ---------------------------------------------------------------------------
# Schedule fetch
# ---------------------------------------------------------------------------

def fetch_schedule(client, con, start_date: str, end_date: str) -> dict:
    """Fetch the MLB schedule for a date range and upsert non-final games.

    Only games that are NOT already Final are inserted (INSERT OR IGNORE), so
    completed rows in fact_game are never touched regardless of re-runs.
    """
    schedule = client.get_schedule(start_date, end_date)
    rows = []
    for date_entry in schedule.get("dates", []):
        for game in date_entry.get("games", []):
            if not normalize.is_final(game):
                rows.append(normalize.normalize_scheduled_game(game))

    db.insert_ignore(con, "fact_game", rows)
    return {"start_date": start_date, "end_date": end_date, "games_scheduled": len(rows)}


# ---------------------------------------------------------------------------
# Pythagorean predictor
# ---------------------------------------------------------------------------

def _pythagorean(rs: float, ra: float) -> float:
    """P(win) from season-to-date runs scored / allowed. Neutral = 0.5."""
    if rs <= 0 and ra <= 0:
        return 0.5
    rs_e = max(rs, 0.01) ** PYTH_EXP
    ra_e = max(ra, 0.01) ** PYTH_EXP
    return rs_e / (rs_e + ra_e)


def _log5(p_home: float, p_away: float) -> float:
    """Bill James Log5: combine two independent win probabilities."""
    num = p_home * (1 - p_away)
    den = num + (1 - p_home) * p_away
    return num / den if den > 0 else 0.5


def _team_win_prob(con, team_id: int, season: int | None) -> float:
    """Pythagorean win% for a team from fact_team_game this season."""
    where = "WHERE team_id = ?"
    params: list = [team_id]
    if season:
        where += " AND game_pk IN (SELECT game_pk FROM fact_game WHERE season = ?)"
        params.append(season)
    try:
        row = con.execute(
            f"SELECT sum(runs_scored), sum(runs_allowed) FROM fact_team_game {where}",
            params,
        ).fetchone()
        if row and row[0] is not None and row[1] is not None and (row[0] + row[1]) > 0:
            return _pythagorean(float(row[0]), float(row[1]))
    except Exception:
        pass
    return _pythagorean(LEAGUE_AVG_RS, LEAGUE_AVG_RS)


def predict_games(con, date_str: str) -> list[dict]:
    """Return prediction dicts for all Scheduled games on date_str.

    Predictions are also upserted into fact_prediction.
    """
    rows = con.execute(
        "SELECT game_pk, home_team_id, away_team_id, season "
        "FROM fact_game WHERE official_date = ? AND status = 'Scheduled'",
        [date_str],
    ).fetchall()

    if not rows:
        return []

    now = dt.datetime.now(dt.timezone.utc).isoformat()
    predictions = []
    pred_rows = []

    for game_pk, home_id, away_id, season in rows:
        p_home_raw = _team_win_prob(con, home_id, season)
        p_away_raw = _team_win_prob(con, away_id, season)

        home_win_prob = _log5(p_home_raw + HOME_ADVANTAGE, p_away_raw)
        home_win_prob = max(0.01, min(0.99, home_win_prob))
        away_win_prob = 1.0 - home_win_prob

        features = {
            "home_pyth": round(p_home_raw, 4),
            "away_pyth": round(p_away_raw, 4),
            "home_advantage": HOME_ADVANTAGE,
        }
        pred = {
            "game_pk": game_pk,
            "home_win_prob": round(home_win_prob, 4),
            "away_win_prob": round(away_win_prob, 4),
            "model_name": MODEL_NAME,
        }
        predictions.append(pred)
        pred_rows.append({
            "prediction_id": f"{MODEL_NAME}_{game_pk}",
            "game_pk": game_pk,
            "model_name": MODEL_NAME,
            "predicted_at": now,
            "home_win_prob": home_win_prob,
            "away_win_prob": away_win_prob,
            "pred_total": None,
            "features_json": json.dumps(features),
        })

    db.upsert(con, "fact_prediction", pred_rows)
    return predictions
