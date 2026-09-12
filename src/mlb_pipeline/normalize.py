"""Pure functions that turn raw MLB Stats API JSON into warehouse rows.

No I/O here: inputs are parsed API payloads, outputs are lists of dicts
whose keys match the columns in sql/schema.sql.
"""

from __future__ import annotations

# codedGameState F = Final, O = Game Over
FINAL_CODED_STATES = {"F", "O"}

BATTING_STAT_COLUMNS = {
    "at_bats": "atBats",
    "runs": "runs",
    "hits": "hits",
    "doubles": "doubles",
    "triples": "triples",
    "home_runs": "homeRuns",
    "rbi": "rbi",
    "walks": "baseOnBalls",
    "strikeouts": "strikeOuts",
    "stolen_bases": "stolenBases",
}

PITCHING_STAT_COLUMNS = {
    "outs": "outs",
    "hits": "hits",
    "runs": "runs",
    "earned_runs": "earnedRuns",
    "walks": "baseOnBalls",
    "strikeouts": "strikeOuts",
    "home_runs": "homeRuns",
    "pitches": "numberOfPitches",
}


def is_final(schedule_game: dict) -> bool:
    status = schedule_game.get("status", {})
    return (
        status.get("codedGameState") in FINAL_CODED_STATES
        or status.get("abstractGameState") == "Final"
    )


def extract_game_pks(schedule: dict, only_final: bool = True) -> list[int]:
    """gamePk values from a schedule payload, optionally completed games only."""
    pks = []
    for date_entry in schedule.get("dates", []):
        for game in date_entry.get("games", []):
            if not only_final or is_final(game):
                pks.append(game["gamePk"])
    return pks


def normalize_teams(feed: dict) -> list[dict]:
    """dim_team rows from a feed/live payload."""
    rows = []
    for side in ("away", "home"):
        team = feed["gameData"]["teams"][side]
        rows.append(
            {
                "team_id": team["id"],
                "name": team.get("name"),
                "abbreviation": team.get("abbreviation"),
                "team_code": team.get("teamCode"),
                "league": (team.get("league") or {}).get("name"),
                "division": (team.get("division") or {}).get("name"),
                "venue": (team.get("venue") or {}).get("name"),
            }
        )
    return rows


def normalize_players(feed: dict) -> list[dict]:
    """dim_player rows from a feed/live payload's gameData.players."""
    rows = []
    for player in feed["gameData"].get("players", {}).values():
        rows.append(
            {
                "player_id": player["id"],
                "full_name": player.get("fullName"),
                "primary_position": (player.get("primaryPosition") or {}).get("abbreviation"),
                "bat_side": (player.get("batSide") or {}).get("code"),
                "pitch_hand": (player.get("pitchHand") or {}).get("code"),
            }
        )
    return rows


def normalize_game(feed: dict) -> dict:
    """fact_game row from a feed/live payload."""
    game_data = feed["gameData"]
    home = game_data["teams"]["home"]
    away = game_data["teams"]["away"]
    linescore_teams = feed.get("liveData", {}).get("linescore", {}).get("teams", {})
    home_score = (linescore_teams.get("home") or {}).get("runs")
    away_score = (linescore_teams.get("away") or {}).get("runs")

    winning_team_id = None
    if home_score is not None and away_score is not None:
        if home_score > away_score:
            winning_team_id = home["id"]
        elif away_score > home_score:
            winning_team_id = away["id"]

    season = game_data.get("game", {}).get("season")
    return {
        "game_pk": feed["gamePk"],
        "official_date": game_data.get("datetime", {}).get("officialDate"),
        "season": int(season) if season is not None else None,
        "game_type": game_data.get("game", {}).get("type"),
        "status": game_data.get("status", {}).get("detailedState"),
        "venue": (game_data.get("venue") or {}).get("name"),
        "home_team_id": home["id"],
        "away_team_id": away["id"],
        "home_score": home_score,
        "away_score": away_score,
        "winning_team_id": winning_team_id,
    }


def normalize_team_games(game_pk: int, boxscore: dict) -> list[dict]:
    """fact_team_game rows (one per side) from a boxscore payload."""
    teams = boxscore["teams"]
    runs = {
        side: teams[side].get("teamStats", {}).get("batting", {}).get("runs")
        for side in ("away", "home")
    }
    rows = []
    for side, opponent in (("away", "home"), ("home", "away")):
        team = teams[side]
        stats = team.get("teamStats", {})
        scored, allowed = runs[side], runs[opponent]
        rows.append(
            {
                "game_pk": game_pk,
                "team_id": team["team"]["id"],
                "opponent_team_id": teams[opponent]["team"]["id"],
                "is_home": side == "home",
                "runs_scored": scored,
                "runs_allowed": allowed,
                "hits": stats.get("batting", {}).get("hits"),
                "errors": stats.get("fielding", {}).get("errors"),
                "win": (scored > allowed) if scored is not None and allowed is not None else None,
            }
        )
    return rows


def normalize_batting(game_pk: int, boxscore: dict) -> list[dict]:
    """fact_player_game_batting rows. Players with no batting stats are skipped."""
    rows = []
    for side in ("away", "home"):
        team = boxscore["teams"][side]
        team_id = team["team"]["id"]
        for player in team.get("players", {}).values():
            batting = player.get("stats", {}).get("batting") or {}
            if not batting:
                continue
            row = {
                "game_pk": game_pk,
                "player_id": player["person"]["id"],
                "team_id": team_id,
            }
            row.update({col: batting.get(api, 0) for col, api in BATTING_STAT_COLUMNS.items()})
            rows.append(row)
    return rows


def normalize_pitching(game_pk: int, boxscore: dict) -> list[dict]:
    """fact_player_game_pitching rows. Players with no pitching stats are skipped."""
    rows = []
    for side in ("away", "home"):
        team = boxscore["teams"][side]
        team_id = team["team"]["id"]
        for player in team.get("players", {}).values():
            pitching = player.get("stats", {}).get("pitching") or {}
            if not pitching:
                continue
            row = {
                "game_pk": game_pk,
                "player_id": player["person"]["id"],
                "team_id": team_id,
                "innings_pitched": pitching.get("inningsPitched"),
            }
            row.update({col: pitching.get(api, 0) for col, api in PITCHING_STAT_COLUMNS.items()})
            rows.append(row)
    return rows
