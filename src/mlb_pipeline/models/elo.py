"""Elo ratings and Pythagorean win expectation for MLB teams.

Ratings live in memory as ``{team_id: rating}`` and are rebuilt by replaying
``fact_game`` (see :func:`build_ratings_from_history`). Nothing is persisted here
except through :func:`predict_game` + ``mlb_pipeline.predict.save_prediction``.
"""

from __future__ import annotations

from mlb_pipeline.predict import GamePrediction

MODEL_NAME = "elo"

DEFAULT_RATING = 1500.0
K_FACTOR = 20.0
SEASON_REGRESS = 0.4  # pull 40% toward mean at season start
HOME_ADVANTAGE = 25.0  # rating points added to the home side


def expected_score(rating_a: float, rating_b: float) -> float:
    """Elo win probability for team A given ratings."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_ratings(
    ratings: dict[int, float],
    home_id: int,
    away_id: int,
    home_won: bool,
    home_advantage: float = HOME_ADVANTAGE,
    k_factor: float = K_FACTOR,
) -> dict[int, float]:
    """Return new ratings dict after one game result (input dict is not mutated).

    Unknown teams start at DEFAULT_RATING. The update is zero-sum: the winner
    gains exactly what the loser loses.
    """
    new = dict(ratings)
    home = new.get(home_id, DEFAULT_RATING)
    away = new.get(away_id, DEFAULT_RATING)
    exp_home = expected_score(home + home_advantage, away)
    delta = k_factor * ((1.0 if home_won else 0.0) - exp_home)
    new[home_id] = home + delta
    new[away_id] = away - delta
    return new


def regress_to_mean(
    ratings: dict[int, float],
    fraction: float = SEASON_REGRESS,
    mean: float = DEFAULT_RATING,
) -> dict[int, float]:
    """Pull every rating ``fraction`` of the way toward ``mean`` (season rollover)."""
    return {team: r + fraction * (mean - r) for team, r in ratings.items()}


def build_ratings_from_history(
    con,
    through_date: str,
    home_advantage: float = HOME_ADVANTAGE,
    k_factor: float = K_FACTOR,
    season_regress: float = SEASON_REGRESS,
) -> dict[int, float]:
    """Replay all fact_game rows up to and including through_date and return ratings.

    Only games with status='Final' and a decided score are used. Games are replayed
    in (season, official_date, game_pk) order; when the season changes, all ratings
    are regressed ``season_regress`` toward DEFAULT_RATING before the new season's
    games are applied.
    """
    rows = con.execute(
        """
        SELECT season, home_team_id, away_team_id, home_score, away_score
        FROM fact_game
        WHERE status = 'Final'
          AND official_date <= CAST(? AS DATE)
          AND home_score IS NOT NULL
          AND away_score IS NOT NULL
          AND home_score <> away_score
        ORDER BY season, official_date, game_pk
        """,
        [through_date],
    ).fetchall()

    ratings: dict[int, float] = {}
    current_season = None
    for season, home_id, away_id, home_score, away_score in rows:
        if current_season is not None and season != current_season and ratings:
            ratings = regress_to_mean(ratings, season_regress)
        current_season = season
        ratings = update_ratings(
            ratings,
            home_id,
            away_id,
            home_won=home_score > away_score,
            home_advantage=home_advantage,
            k_factor=k_factor,
        )
    return ratings


def predict(
    ratings: dict[int, float],
    home_id: int,
    away_id: int,
    home_advantage: float = HOME_ADVANTAGE,
) -> dict:
    """Return {'home_win_prob': float, 'away_win_prob': float, 'model': 'elo'}."""
    home = ratings.get(home_id, DEFAULT_RATING)
    away = ratings.get(away_id, DEFAULT_RATING)
    home_prob = expected_score(home + home_advantage, away)
    return {
        "home_win_prob": home_prob,
        "away_win_prob": 1.0 - home_prob,
        "model": MODEL_NAME,
    }


def predict_game(
    ratings: dict[int, float],
    game_pk: int,
    home_id: int,
    away_id: int,
    home_advantage: float = HOME_ADVANTAGE,
) -> GamePrediction:
    """Wrap :func:`predict` in the unified GamePrediction interface (no run total)."""
    probs = predict(ratings, home_id, away_id, home_advantage)
    return GamePrediction(
        game_pk=game_pk,
        model_name=MODEL_NAME,
        home_win_prob=probs["home_win_prob"],
        away_win_prob=probs["away_win_prob"],
        pred_total=None,
        features={
            "home_team_id": home_id,
            "away_team_id": away_id,
            "home_rating": ratings.get(home_id, DEFAULT_RATING),
            "away_rating": ratings.get(away_id, DEFAULT_RATING),
            "home_advantage": home_advantage,
        },
    )


def pythagorean_win_pct(
    runs_scored: float, runs_allowed: float, exponent: float = 1.83
) -> float:
    """Pythagorean expectation: RS^e / (RS^e + RA^e). Returns 0.5 when no runs at all."""
    if runs_scored < 0 or runs_allowed < 0:
        raise ValueError("runs must be non-negative")
    if runs_scored == 0 and runs_allowed == 0:
        return 0.5
    rs = runs_scored**exponent
    ra = runs_allowed**exponent
    return rs / (rs + ra)
