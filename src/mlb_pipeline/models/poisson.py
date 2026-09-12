"""Poisson run-scoring model.

Each team's runs in a game are modelled as an independent Poisson process.
Given an expected-runs rate (lambda) for the home and away side, moneyline
and total (over/under) probabilities follow analytically:

- P(home wins) / P(away wins) / P(tie) come from summing the joint PMF over
  the (home_runs, away_runs) score grid.
- The sum of two independent Poissons is Poisson(lam_home + lam_away), so
  over/under probabilities are a single survival-function lookup.

Lambdas are estimated log5-style from rolling team features::

    lambda = (team_runs_per_game * opp_runs_allowed_per_game / league_avg)
             * park_factor * weather_adjustment

Baseball games cannot end tied, so `predict()` reallocates the tie mass
proportionally to the two win probabilities (extra innings favour the side
that is already more likely to win), which keeps home_win_prob +
away_win_prob == 1.0 as the unified prediction interface expects. The raw
tie probability is still reported in `features`.

This module only reads from the DuckDB connection it is given.
"""

from __future__ import annotations

import math

from scipy.stats import poisson as scipy_poisson

from mlb_pipeline import features as _features
from mlb_pipeline.predict import GamePrediction

MODEL_NAME = "poisson"

# MLB league-average runs per team per game; used both as the log5
# normaliser and as the fallback lambda for teams without enough history.
LEAGUE_AVG_RUNS = 4.5

# Rolling windows tried in order when fetching team features.
DEFAULT_WINDOW = 15
FALLBACK_WINDOWS = (15, 30)

# A team needs at least this many games inside the window before its rolling
# rates are trusted; otherwise it is projected at the league average.
MIN_GAMES = 5

# Lower bound on any estimated lambda so PMFs never degenerate.
MIN_LAMBDA = 0.1

__all__ = [
    "GamePrediction",
    "LEAGUE_AVG_RUNS",
    "MODEL_NAME",
    "estimate_lambda",
    "over_prob",
    "poisson_pmf",
    "predict",
    "win_prob_from_lambdas",
]


# ---------------------------------------------------------------------------
# Pure probability helpers
# ---------------------------------------------------------------------------


def poisson_pmf(k: int, lam: float) -> float:
    """P(X = k) for X ~ Poisson(lam)."""
    if lam < 0:
        raise ValueError("lam must be non-negative")
    if k < 0:
        return 0.0
    return float(scipy_poisson.pmf(k, lam))


def win_prob_from_lambdas(lam_home: float, lam_away: float, max_runs: int = 30) -> dict:
    """Compute P(home wins), P(away wins), P(tie) by summing over score matrices.

    Each side's PMF is evaluated on 0..max_runs and the joint (independent)
    distribution is summed below, above, and on the diagonal. The truncated
    tail mass beyond max_runs is negligible for realistic lambdas; the three
    probabilities are renormalised so they sum to exactly 1.0.

    Returns {'home_win_prob', 'away_win_prob', 'tie_prob'}.
    """
    if max_runs < 0:
        raise ValueError("max_runs must be non-negative")
    ks = range(max_runs + 1)
    home_pmf = [poisson_pmf(k, lam_home) for k in ks]
    away_pmf = [poisson_pmf(k, lam_away) for k in ks]

    home_win = away_win = tie = 0.0
    for h, ph in enumerate(home_pmf):
        for a, pa in enumerate(away_pmf):
            joint = ph * pa
            if h > a:
                home_win += joint
            elif a > h:
                away_win += joint
            else:
                tie += joint

    total = home_win + away_win + tie
    if total <= 0:  # pragma: no cover - only reachable with absurd inputs
        raise ValueError("no probability mass within max_runs; increase max_runs")
    return {
        "home_win_prob": home_win / total,
        "away_win_prob": away_win / total,
        "tie_prob": tie / total,
    }


def over_prob(lam_home: float, lam_away: float, total_line: float) -> float:
    """P(home_runs + away_runs > total_line).

    The total is Poisson(lam_home + lam_away). For a half-run line (8.5) this
    is P(T >= 9); for a whole-number line (8) a push at exactly 8 counts as
    neither over nor under, so it is still P(T >= 9).
    """
    lam_total = lam_home + lam_away
    if lam_total < 0:
        raise ValueError("lambdas must be non-negative")
    if total_line < 0:
        return 1.0
    # sf(k) = P(T > k) for integer k.
    return float(scipy_poisson.sf(math.floor(total_line), lam_total))


def estimate_lambda(
    team_runs_per_game: float,
    opp_runs_allowed_per_game: float,
    league_avg: float,
    park_factor: float = 1.0,
    weather_adjustment: float = 1.0,
) -> float:
    """Log5-style lambda: (off * def / avg) * park * weather.

    A team that scores at the league rate against a defence that allows the
    league rate projects to exactly league_avg (times park and weather).
    """
    if league_avg <= 0:
        raise ValueError("league_avg must be positive")
    lam = (team_runs_per_game * opp_runs_allowed_per_game / league_avg) * park_factor * weather_adjustment
    return max(float(lam), MIN_LAMBDA)


# ---------------------------------------------------------------------------
# Warehouse-backed prediction
# ---------------------------------------------------------------------------


def _team_rates(con, team_id: int, as_of_date: str, league_avg: float) -> dict:
    """Runs scored / allowed per game for a team, with league-average fallback.

    Tries each window in FALLBACK_WINDOWS and uses the first with at least
    MIN_GAMES games and non-null rates. Returns a dict with runs_per_game,
    runs_allowed_per_game, games, window, and a `fallback` flag.
    """
    for window in FALLBACK_WINDOWS:
        feats = _features.team_rolling_features(con, team_id, as_of_date, window)
        if not feats:
            continue
        rpg = feats.get("runs_per_game")
        rapg = feats.get("runs_allowed_per_game")
        games = int(feats.get("games") or 0)
        if rpg is None or rapg is None or games < MIN_GAMES:
            continue
        return {
            "runs_per_game": float(rpg),
            "runs_allowed_per_game": float(rapg),
            "games": games,
            "window": window,
            "fallback": False,
        }
    return {
        "runs_per_game": league_avg,
        "runs_allowed_per_game": league_avg,
        "games": 0,
        "window": None,
        "fallback": True,
    }


def _park_run_factor(con, game_pk: int) -> float:
    """Home park run factor for a game; neutral (1.0) if unknown."""
    context = _features.game_context_features(con, game_pk)
    if not context:
        return 1.0
    factor = context.get("park_run_factor")
    return float(factor) if factor is not None else 1.0


def predict(
    con,
    game_pk: int,
    home_id: int,
    away_id: int,
    as_of_date: str,
    total_line: float = 8.5,
    league_avg: float = LEAGUE_AVG_RUNS,
    weather_adjustment: float = 1.0,
) -> GamePrediction:
    """Fetch rolling features from con, estimate lambdas, return GamePrediction.

    Uses features.team_rolling_features and features.game_context_features.
    Falls back to league-average lambda (4.5) for teams with insufficient
    history (fewer than MIN_GAMES games in the rolling window) and to a
    neutral park when the game or park is unknown.

    home_win_prob + away_win_prob == 1.0: the Poisson tie mass is split
    proportionally between the two sides. pred_total is lam_home + lam_away.
    """
    home = _team_rates(con, home_id, as_of_date, league_avg)
    away = _team_rates(con, away_id, as_of_date, league_avg)
    park_factor = _park_run_factor(con, game_pk)

    lam_home = estimate_lambda(
        home["runs_per_game"], away["runs_allowed_per_game"], league_avg, park_factor, weather_adjustment
    )
    lam_away = estimate_lambda(
        away["runs_per_game"], home["runs_allowed_per_game"], league_avg, park_factor, weather_adjustment
    )

    outcome = win_prob_from_lambdas(lam_home, lam_away)
    decided = outcome["home_win_prob"] + outcome["away_win_prob"]
    home_win_prob = outcome["home_win_prob"] / decided
    away_win_prob = outcome["away_win_prob"] / decided

    p_over = over_prob(lam_home, lam_away, total_line)
    # A whole-number line has push mass; a half-run line does not.
    p_push = poisson_pmf(int(total_line), lam_home + lam_away) if float(total_line).is_integer() else 0.0
    p_under = max(0.0, 1.0 - p_over - p_push)

    features = {
        "as_of_date": str(as_of_date),
        "league_avg": league_avg,
        "park_run_factor": park_factor,
        "weather_adjustment": weather_adjustment,
        "lam_home": lam_home,
        "lam_away": lam_away,
        "home_runs_per_game": home["runs_per_game"],
        "home_runs_allowed_per_game": home["runs_allowed_per_game"],
        "home_games": home["games"],
        "home_window": home["window"],
        "home_fallback": home["fallback"],
        "away_runs_per_game": away["runs_per_game"],
        "away_runs_allowed_per_game": away["runs_allowed_per_game"],
        "away_games": away["games"],
        "away_window": away["window"],
        "away_fallback": away["fallback"],
        "tie_prob_raw": outcome["tie_prob"],
        "total_line": total_line,
        "over_prob": p_over,
        "under_prob": p_under,
        "push_prob": p_push,
    }

    return GamePrediction(
        game_pk=game_pk,
        model_name=MODEL_NAME,
        home_win_prob=home_win_prob,
        away_win_prob=away_win_prob,
        pred_total=lam_home + lam_away,
        features=features,
    )
