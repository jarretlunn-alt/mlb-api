"""Shared feature engineering for the analytics layer.

Pre-game features are computed from the warehouse for a (game_pk, as_of_date)
pair. Every feature function only *reads* from the DuckDB connection it is
given; the only writers in this module are the explicit materializers
(`seed_park_factors`, `refresh_pitcher_log`, `refresh_team_rolling`), which
are idempotent INSERT OR REPLACE loads.

Leakage rule: `as_of_date` is exclusive. Features "as of" a date use only
games whose official_date is strictly earlier, so building features for a
game with as_of_date = its own official_date never sees that game's result.

Derivation notes (the warehouse has no lineup/starter information):
- A team's *starter* in a game is the pitcher who recorded the most outs
  (ties: most pitches). For an opener + bulk-reliever game this flags the
  bulk pitcher, which is the more useful "starter" for projection purposes.
- rest_days is days since the pitcher's previous appearance of any kind.
- OPS vs RHP/LHP attributes every plate appearance in a game to the
  handedness of the opposing team's starter (see above).
- FIP uses (13*HR + 3*BB - 2*K) / IP + FIP_CONSTANT; HBP is not tracked.
"""

from __future__ import annotations

import datetime as dt

from . import db

FIP_CONSTANT = 3.10

# Approximate 2024-season park factors (3-year style, 1.00 = neutral).
# Keyed by MLB team_id; park_id is the MLB Stats API venue id.
# Exact values are not important for the MVP; they are meant to be replaced
# by a data-driven estimate later.
PARK_FACTORS: dict[int, dict] = {
    108: {"park_id": 1, "name": "Angel Stadium", "run_factor": 1.00, "hr_factor": 1.03, "handedness": "neutral"},
    109: {"park_id": 15, "name": "Chase Field", "run_factor": 1.03, "hr_factor": 1.00, "handedness": "neutral"},
    110: {"park_id": 2, "name": "Oriole Park at Camden Yards", "run_factor": 1.00, "hr_factor": 0.93, "handedness": "lhb"},
    111: {"park_id": 3, "name": "Fenway Park", "run_factor": 1.06, "hr_factor": 0.95, "handedness": "rhb"},
    112: {"park_id": 17, "name": "Wrigley Field", "run_factor": 1.01, "hr_factor": 1.00, "handedness": "neutral"},
    113: {"park_id": 2602, "name": "Great American Ball Park", "run_factor": 1.04, "hr_factor": 1.15, "handedness": "lhb"},
    114: {"park_id": 5, "name": "Progressive Field", "run_factor": 0.98, "hr_factor": 0.95, "handedness": "neutral"},
    115: {"park_id": 19, "name": "Coors Field", "run_factor": 1.12, "hr_factor": 1.10, "handedness": "neutral"},
    116: {"park_id": 2394, "name": "Comerica Park", "run_factor": 0.98, "hr_factor": 0.90, "handedness": "neutral"},
    117: {"park_id": 2392, "name": "Daikin Park", "run_factor": 1.00, "hr_factor": 1.03, "handedness": "rhb"},
    118: {"park_id": 7, "name": "Kauffman Stadium", "run_factor": 1.01, "hr_factor": 0.90, "handedness": "neutral"},
    119: {"park_id": 22, "name": "Dodger Stadium", "run_factor": 0.99, "hr_factor": 1.08, "handedness": "neutral"},
    120: {"park_id": 3309, "name": "Nationals Park", "run_factor": 1.01, "hr_factor": 1.02, "handedness": "neutral"},
    121: {"park_id": 3289, "name": "Citi Field", "run_factor": 0.98, "hr_factor": 1.00, "handedness": "neutral"},
    133: {"park_id": 2529, "name": "Sutter Health Park", "run_factor": 1.03, "hr_factor": 1.02, "handedness": "neutral"},
    134: {"park_id": 31, "name": "PNC Park", "run_factor": 0.99, "hr_factor": 0.90, "handedness": "neutral"},
    135: {"park_id": 2680, "name": "Petco Park", "run_factor": 0.96, "hr_factor": 0.94, "handedness": "neutral"},
    136: {"park_id": 680, "name": "T-Mobile Park", "run_factor": 0.92, "hr_factor": 0.95, "handedness": "neutral"},
    137: {"park_id": 2395, "name": "Oracle Park", "run_factor": 0.95, "hr_factor": 0.85, "handedness": "rhb"},
    138: {"park_id": 2889, "name": "Busch Stadium", "run_factor": 0.98, "hr_factor": 0.92, "handedness": "neutral"},
    139: {"park_id": 2523, "name": "George M. Steinbrenner Field", "run_factor": 1.02, "hr_factor": 1.08, "handedness": "lhb"},
    140: {"park_id": 5325, "name": "Globe Life Field", "run_factor": 0.99, "hr_factor": 1.02, "handedness": "neutral"},
    141: {"park_id": 14, "name": "Rogers Centre", "run_factor": 1.01, "hr_factor": 1.03, "handedness": "neutral"},
    142: {"park_id": 3312, "name": "Target Field", "run_factor": 1.00, "hr_factor": 1.00, "handedness": "neutral"},
    143: {"park_id": 2681, "name": "Citizens Bank Park", "run_factor": 1.01, "hr_factor": 1.08, "handedness": "lhb"},
    144: {"park_id": 4705, "name": "Truist Park", "run_factor": 1.02, "hr_factor": 1.05, "handedness": "neutral"},
    145: {"park_id": 4, "name": "Rate Field", "run_factor": 1.01, "hr_factor": 1.08, "handedness": "rhb"},
    146: {"park_id": 4169, "name": "loanDepot park", "run_factor": 0.96, "hr_factor": 0.92, "handedness": "neutral"},
    147: {"park_id": 3313, "name": "Yankee Stadium", "run_factor": 1.02, "hr_factor": 1.12, "handedness": "lhb"},
    158: {"park_id": 32, "name": "American Family Field", "run_factor": 1.00, "hr_factor": 1.05, "handedness": "neutral"},
}

# Columns in a game feature row that are outcomes, not inputs. Model code
# must drop these from X; they are included so one row serves train + test.
LABEL_COLUMNS = ("home_score", "away_score", "total_runs", "home_win")

PITCHER_FEATURE_KEYS = (
    "era_last_n",
    "fip_last_n",
    "k_per_9",
    "bb_per_9",
    "hr_per_9",
    "avg_rest_days",
    "n_starts",
)

TEAM_FEATURE_KEYS = (
    "runs_per_game",
    "runs_allowed_per_game",
    "ops_vs_rhp",
    "ops_vs_lhp",
    "bullpen_fip",
    "games",
)

# Pitcher appearances derived from the base tables. Shared by the feature
# functions and by refresh_pitcher_log so they can never disagree.
_PITCHER_LOG_SQL = """
    WITH appearances AS (
        SELECT
            p.game_pk,
            p.player_id,
            p.team_id,
            g.official_date,
            p.outs,
            p.hits,
            p.runs,
            p.earned_runs,
            p.walks,
            p.strikeouts,
            p.home_runs,
            p.pitches,
            row_number() OVER (
                PARTITION BY p.game_pk, p.team_id
                ORDER BY p.outs DESC NULLS LAST, p.pitches DESC NULLS LAST, p.player_id
            ) AS team_rank
        FROM fact_player_game_pitching p
        JOIN fact_game g ON g.game_pk = p.game_pk
    )
    SELECT
        game_pk,
        player_id,
        team_id,
        team_rank = 1 AS is_starter,
        official_date,
        date_diff(
            'day',
            lag(official_date) OVER (PARTITION BY player_id ORDER BY official_date, game_pk),
            official_date
        )::INTEGER AS rest_days,
        outs,
        hits,
        runs,
        earned_runs,
        walks,
        strikeouts,
        home_runs,
        pitches
    FROM appearances
"""


def _as_date(value) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value))


def _safe_div(numerator, denominator):
    if not denominator:
        return None
    return numerator / denominator


def _fip(home_runs: int, walks: int, strikeouts: int, outs: int):
    innings = outs / 3.0
    if innings <= 0:
        return None
    return (13 * home_runs + 3 * walks - 2 * strikeouts) / innings + FIP_CONSTANT


def _ops(at_bats: int, hits: int, doubles: int, triples: int, home_runs: int, walks: int):
    """OBP + SLG without HBP/SF (not tracked in the warehouse)."""
    plate_appearances = at_bats + walks
    if not at_bats or not plate_appearances:
        return None
    total_bases = hits + doubles + 2 * triples + 3 * home_runs
    return (hits + walks) / plate_appearances + total_bases / at_bats


# ---------------------------------------------------------------------------
# Materializers (writers)
# ---------------------------------------------------------------------------


def seed_park_factors(con) -> int:
    """Upsert the static PARK_FACTORS into dim_park. Idempotent."""
    rows = [
        {
            "park_id": park["park_id"],
            "name": park["name"],
            "team_id": team_id,
            "run_factor": park["run_factor"],
            "hr_factor": park["hr_factor"],
            "handedness": park["handedness"],
        }
        for team_id, park in PARK_FACTORS.items()
    ]
    return db.upsert(con, "dim_park", rows)


def refresh_pitcher_log(con) -> int:
    """Rebuild fact_pitcher_log from the base tables. Idempotent."""
    con.execute(f"INSERT OR REPLACE INTO fact_pitcher_log {_PITCHER_LOG_SQL}")
    return con.execute("SELECT count(*) FROM fact_pitcher_log").fetchone()[0]


def refresh_team_rolling(con, as_of_date, windows=(7, 15, 30), team_ids=None) -> int:
    """Materialize fact_team_rolling for every team (or team_ids) as of a date.

    Teams with no games inside a window get no row for that window.
    """
    as_of = _as_date(as_of_date)
    if team_ids is None:
        team_ids = [r[0] for r in con.execute("SELECT team_id FROM dim_team ORDER BY team_id").fetchall()]
    rows = []
    for team_id in team_ids:
        for window in windows:
            feats = team_rolling_features(con, team_id, as_of, window)
            if feats is None:
                continue
            rows.append(
                {
                    "team_id": team_id,
                    "as_of_date": as_of,
                    "window_days": window,
                    "runs_scored": feats["runs_per_game"],
                    "runs_allowed": feats["runs_allowed_per_game"],
                    "ops_vs_rhp": feats["ops_vs_rhp"],
                    "ops_vs_lhp": feats["ops_vs_lhp"],
                    "fip": feats["bullpen_fip"],
                }
            )
    return db.upsert(con, "fact_team_rolling", rows)


# ---------------------------------------------------------------------------
# Feature readers
# ---------------------------------------------------------------------------


def pitcher_features(con, player_id: int, as_of_date, n_starts: int = 5) -> dict | None:
    """Recent performance for a starting pitcher over their last n starts
    before as_of_date (exclusive).

    Returns era_last_n, fip_last_n, k_per_9, bb_per_9, hr_per_9,
    avg_rest_days, n_starts. Returns None if fewer than 2 starts are found.
    """
    as_of = _as_date(as_of_date)
    starts = con.execute(
        f"""
        SELECT outs, earned_runs, walks, strikeouts, home_runs, rest_days
        FROM ({_PITCHER_LOG_SQL})
        WHERE player_id = ? AND is_starter AND official_date < ?
        ORDER BY official_date DESC, game_pk DESC
        LIMIT ?
        """,
        [player_id, as_of, n_starts],
    ).fetchall()
    if len(starts) < 2:
        return None

    outs = sum(r[0] or 0 for r in starts)
    earned_runs = sum(r[1] or 0 for r in starts)
    walks = sum(r[2] or 0 for r in starts)
    strikeouts = sum(r[3] or 0 for r in starts)
    home_runs = sum(r[4] or 0 for r in starts)
    rest = [r[5] for r in starts if r[5] is not None]
    innings = outs / 3.0

    return {
        "era_last_n": _safe_div(9 * earned_runs, innings),
        "fip_last_n": _fip(home_runs, walks, strikeouts, outs),
        "k_per_9": _safe_div(9 * strikeouts, innings),
        "bb_per_9": _safe_div(9 * walks, innings),
        "hr_per_9": _safe_div(9 * home_runs, innings),
        "avg_rest_days": (sum(rest) / len(rest)) if rest else None,
        "n_starts": len(starts),
    }


def team_rolling_features(con, team_id: int, as_of_date, window: int = 15) -> dict | None:
    """Team offensive/defensive stats over games in [as_of - window, as_of).

    Returns runs_per_game, runs_allowed_per_game, ops_vs_rhp, ops_vs_lhp,
    bullpen_fip, games. Returns None if the team played no games in the window.
    Split/bullpen values are None when the window has no qualifying data.
    """
    as_of = _as_date(as_of_date)
    start = as_of - dt.timedelta(days=window)
    params = [team_id, start, as_of]

    games, runs_per_game, runs_allowed_per_game = con.execute(
        """
        SELECT count(*), avg(tg.runs_scored), avg(tg.runs_allowed)
        FROM fact_team_game tg
        JOIN fact_game g ON g.game_pk = tg.game_pk
        WHERE tg.team_id = ? AND g.official_date >= ? AND g.official_date < ?
        """,
        params,
    ).fetchone()
    if not games:
        return None

    # Batting split by the opposing starter's throwing hand.
    split_rows = con.execute(
        f"""
        WITH starters AS (
            SELECT game_pk, team_id, player_id FROM ({_PITCHER_LOG_SQL}) WHERE is_starter
        )
        SELECT
            dp.pitch_hand,
            sum(b.at_bats), sum(b.hits), sum(b.doubles), sum(b.triples), sum(b.home_runs), sum(b.walks)
        FROM fact_player_game_batting b
        JOIN fact_game g ON g.game_pk = b.game_pk
        JOIN fact_team_game tg ON tg.game_pk = b.game_pk AND tg.team_id = b.team_id
        LEFT JOIN starters s ON s.game_pk = b.game_pk AND s.team_id = tg.opponent_team_id
        LEFT JOIN dim_player dp ON dp.player_id = s.player_id
        WHERE b.team_id = ? AND g.official_date >= ? AND g.official_date < ?
        GROUP BY dp.pitch_hand
        """,
        params,
    ).fetchall()
    ops_by_hand = {row[0]: _ops(*(int(v or 0) for v in row[1:])) for row in split_rows}

    bullpen = con.execute(
        f"""
        SELECT coalesce(sum(home_runs), 0), coalesce(sum(walks), 0),
               coalesce(sum(strikeouts), 0), coalesce(sum(outs), 0)
        FROM ({_PITCHER_LOG_SQL})
        WHERE team_id = ? AND official_date >= ? AND official_date < ? AND NOT is_starter
        """,
        params,
    ).fetchone()

    return {
        "runs_per_game": float(runs_per_game) if runs_per_game is not None else None,
        "runs_allowed_per_game": float(runs_allowed_per_game) if runs_allowed_per_game is not None else None,
        "ops_vs_rhp": ops_by_hand.get("R"),
        "ops_vs_lhp": ops_by_hand.get("L"),
        "bullpen_fip": _fip(*(int(v) for v in bullpen)),
        "games": int(games),
    }


def game_context_features(con, game_pk: int) -> dict | None:
    """Park factors and home/away flags for a game.

    Park is resolved from the home team's entry in dim_park; unknown parks
    are treated as neutral (1.0). Returns None if the game is not in fact_game.
    """
    row = con.execute(
        """
        SELECT g.home_team_id, g.away_team_id, g.official_date,
               pk.park_id, pk.run_factor, pk.hr_factor, pk.handedness
        FROM fact_game g
        LEFT JOIN dim_park pk ON pk.team_id = g.home_team_id
        WHERE g.game_pk = ?
        """,
        [game_pk],
    ).fetchone()
    if row is None:
        return None
    home_team_id, away_team_id, official_date, park_id, run_factor, hr_factor, handedness = row
    return {
        "game_pk": game_pk,
        "official_date": official_date,
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "park_id": park_id,
        "park_run_factor": float(run_factor) if run_factor is not None else 1.0,
        "park_hr_factor": float(hr_factor) if hr_factor is not None else 1.0,
        "park_handedness": handedness or "neutral",
        "home_is_home": True,
        "away_is_home": False,
    }


def _game_starters(con, game_pk: int) -> dict[int, int]:
    """{team_id: player_id} of each team's derived starter for a played game."""
    rows = con.execute(
        f"SELECT team_id, player_id FROM ({_PITCHER_LOG_SQL}) WHERE game_pk = ? AND is_starter",
        [game_pk],
    ).fetchall()
    return {team_id: player_id for team_id, player_id in rows}


def build_game_feature_row(
    con,
    game_pk: int,
    as_of_date,
    home_pitcher_id: int | None = None,
    away_pitcher_id: int | None = None,
    n_starts: int = 5,
    window: int = 15,
) -> dict | None:
    """Assemble all features for one game into a flat dict for model input.

    Starting pitchers default to the derived starters from the game's own
    boxscore (works for played games); pass home/away pitcher ids explicitly
    for games that have not been played yet (probable pitchers).
    Outcome columns (LABEL_COLUMNS) are included when the game has scores.

    Returns None if the game is unknown, a starter cannot be determined, a
    starter has fewer than 2 prior starts, or a team has no games in the window.
    """
    context = game_context_features(con, game_pk)
    if context is None:
        return None
    as_of = _as_date(as_of_date)
    home_team_id, away_team_id = context["home_team_id"], context["away_team_id"]

    if home_pitcher_id is None or away_pitcher_id is None:
        starters = _game_starters(con, game_pk)
        home_pitcher_id = home_pitcher_id or starters.get(home_team_id)
        away_pitcher_id = away_pitcher_id or starters.get(away_team_id)
    if home_pitcher_id is None or away_pitcher_id is None:
        return None

    home_sp = pitcher_features(con, home_pitcher_id, as_of, n_starts)
    away_sp = pitcher_features(con, away_pitcher_id, as_of, n_starts)
    home_team = team_rolling_features(con, home_team_id, as_of, window)
    away_team = team_rolling_features(con, away_team_id, as_of, window)
    if any(part is None for part in (home_sp, away_sp, home_team, away_team)):
        return None

    row = {
        "game_pk": game_pk,
        "official_date": context["official_date"],
        "as_of_date": as_of,
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "home_pitcher_id": home_pitcher_id,
        "away_pitcher_id": away_pitcher_id,
        "park_run_factor": context["park_run_factor"],
        "park_hr_factor": context["park_hr_factor"],
        "park_handedness": context["park_handedness"],
        "home_is_home": context["home_is_home"],
        "away_is_home": context["away_is_home"],
    }
    for side, sp, team in (("home", home_sp, home_team), ("away", away_sp, away_team)):
        row.update({f"{side}_sp_{k}": v for k, v in sp.items()})
        row.update({f"{side}_{k}": v for k, v in team.items()})

    # Opening line (optional — None when fact_game_odds has no row for this game).
    # To activate as a model feature: add "line_home_win_prob" to ensemble.FEATURE_NAMES
    # and raw.get("line_home_win_prob") to _feature_row() once odds are backfilled.
    odds = game_odds_features(con, game_pk)
    if odds is not None:
        row.update(odds)

    # Weather (optional — None when fact_game_weather has no row for this game).
    # To activate as model features: add "temp_f", "wind_out_mph" to ensemble.FEATURE_NAMES
    # and raw.get(...) to _feature_row() once weather is backfilled.
    weather = game_weather_features(con, game_pk)
    if weather is not None:
        row.update(weather)

    home_score, away_score = con.execute(
        "SELECT home_score, away_score FROM fact_game WHERE game_pk = ?", [game_pk]
    ).fetchone()
    if home_score is not None and away_score is not None:
        row["home_score"] = home_score
        row["away_score"] = away_score
        row["total_runs"] = home_score + away_score
        row["home_win"] = home_score > away_score
    return row


def _vig_free_prob(home_ml: int, away_ml: int) -> float:
    """Remove the bookmaker vig and return the fair home win probability.

    Converts American moneylines to implied probabilities then normalises so
    the two sides sum to 1.0.  Works for favourites (negative) and underdogs
    (positive).
    """
    def implied(ml: int) -> float:
        return abs(ml) / (abs(ml) + 100) if ml < 0 else 100 / (ml + 100)
    p_h, p_a = implied(home_ml), implied(away_ml)
    return p_h / (p_h + p_a)


def game_odds_features(con, game_pk: int) -> dict | None:
    """Vig-free opening line probability from fact_game_odds.

    Returns {'line_home_win_prob': float} using the sharpest available
    bookmaker (Pinnacle preferred), or None if no odds are stored for this
    game.
    """
    row = con.execute(
        """
        SELECT home_ml, away_ml
        FROM fact_game_odds
        WHERE game_pk = ?
        ORDER BY CASE sportsbook
            WHEN 'pinnacle'   THEN 0
            WHEN 'draftkings' THEN 1
            WHEN 'fanduel'    THEN 2
            ELSE 3
        END
        LIMIT 1
        """,
        [game_pk],
    ).fetchone()
    if row is None:
        return None
    return {"line_home_win_prob": _vig_free_prob(row[0], row[1])}


def game_weather_features(con, game_pk: int) -> dict | None:
    """Weather conditions from fact_game_weather, or None if not yet fetched.

    Returns {'temp_f', 'wind_mph', 'wind_out_mph', 'precip_prob'}.
    wind_out_mph is already direction-adjusted (positive = tailwind toward CF).
    """
    row = con.execute(
        "SELECT temp_f, wind_mph, wind_out_mph, precip_prob "
        "FROM fact_game_weather WHERE game_pk = ?",
        [game_pk],
    ).fetchone()
    if row is None:
        return None
    return {
        "temp_f":       row[0],
        "wind_mph":     row[1],
        "wind_out_mph": row[2],
        "precip_prob":  row[3],
    }


def feature_row_keys(n_starts_prefixes=("home", "away")) -> list[str]:
    """Ordered list of every non-label key build_game_feature_row can emit."""
    keys = [
        "game_pk",
        "official_date",
        "as_of_date",
        "home_team_id",
        "away_team_id",
        "home_pitcher_id",
        "away_pitcher_id",
        "park_run_factor",
        "park_hr_factor",
        "park_handedness",
        "home_is_home",
        "away_is_home",
    ]
    for side in n_starts_prefixes:
        keys.extend(f"{side}_sp_{k}" for k in PITCHER_FEATURE_KEYS)
        keys.extend(f"{side}_{k}" for k in TEAM_FEATURE_KEYS)
    return keys
