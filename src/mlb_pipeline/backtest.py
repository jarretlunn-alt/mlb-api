"""Walk-forward backtesting harness for game-outcome models.

The harness is model-agnostic: models are passed in as callables and this
module never imports ``elo``, ``poisson``, or any other model. A fold trains
on seasons ``1..N-1`` (the caller's model is expected to have been fitted on
those seasons) and evaluates on season ``N``, accumulating one prediction row
per completed game which is then scored.

Prediction rows are plain dicts with the keys::

    game_pk, date, home_id, away_id, home_win_prob, actual_home_win

and optionally ``market_home_prob`` -- the market's *fair* (vig-free)
probability that the home team wins. When absent, ROI simulation falls back to
``DEFAULT_MARKET_HOME_PROB``, a synthetic market that prices every game at the
long-run MLB home-win rate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

# Game statuses (fact_game.status, MLB ``detailedState``) that carry a final
# result. Anything else is skipped by run_backtest.
FINAL_STATUSES: frozenset[str] = frozenset({"Final", "Completed Early", "Game Over"})

# Long-run MLB home-team win rate; used as a synthetic fair market price when a
# prediction row carries no ``market_home_prob``.
DEFAULT_MARKET_HOME_PROB = 0.54

_EPS = 1e-15


@dataclass
class BacktestResult:
    predictions: list[dict] = field(default_factory=list)
    brier_score: float = float("nan")  # lower is better; 0.25 = coin-flip baseline
    log_loss: float = float("nan")
    calibration: dict = field(default_factory=dict)  # {bucket_label: {...}}
    roi_flat_bet: float = 0.0  # $1 on every game where edge > min_edge
    roi_kelly: float = 0.0  # quarter-Kelly ROI
    test_season: int | None = None
    train_seasons: list[int] = field(default_factory=list)
    n_games: int = 0
    n_skipped: int = 0
    bets_placed: int = 0


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def _prob(pred: dict) -> float:
    return float(pred["home_win_prob"])


def _outcome(pred: dict) -> float:
    return 1.0 if pred["actual_home_win"] else 0.0


def brier_score(predictions: list[dict]) -> float:
    """Mean squared error of home_win_prob vs actual_home_win.

    Returns NaN for an empty prediction set.
    """
    if not predictions:
        return float("nan")
    total = sum((_prob(p) - _outcome(p)) ** 2 for p in predictions)
    return total / len(predictions)


def log_loss(predictions: list[dict]) -> float:
    """Negative mean log likelihood.

    Probabilities are clipped to ``[1e-15, 1 - 1e-15]`` so a confident wrong
    call is penalised heavily but never produces an infinite loss.
    Returns NaN for an empty prediction set.
    """
    if not predictions:
        return float("nan")
    total = 0.0
    for p in predictions:
        prob = min(max(_prob(p), _EPS), 1.0 - _EPS)
        y = _outcome(p)
        total -= y * math.log(prob) + (1.0 - y) * math.log(1.0 - prob)
    return total / len(predictions)


def _bucket_label(index: int, n_buckets: int) -> str:
    lo = index / n_buckets
    hi = (index + 1) / n_buckets
    return f"{lo:.2f}-{hi:.2f}"


def calibration_buckets(predictions: list[dict], n_buckets: int = 10) -> dict:
    """Bin predictions by predicted probability; compute actual win rate per bucket.

    Buckets are equal-width over ``[0, 1]`` and keyed by a range string such as
    ``"0.50-0.60"``. Every bucket is present so the structure is stable; an
    empty bucket has ``n == 0`` and ``None`` for both rates. A probability of
    exactly 1.0 lands in the top bucket.
    """
    if n_buckets < 1:
        raise ValueError("n_buckets must be >= 1")

    sums: list[list[float]] = [[0.0, 0.0, 0] for _ in range(n_buckets)]
    for p in predictions:
        prob = _prob(p)
        index = min(int(prob * n_buckets), n_buckets - 1)
        index = max(index, 0)
        sums[index][0] += prob
        sums[index][1] += _outcome(p)
        sums[index][2] += 1

    out: dict[str, dict[str, Any]] = {}
    for i, (prob_sum, win_sum, n) in enumerate(sums):
        out[_bucket_label(i, n_buckets)] = {
            "predicted_prob": prob_sum / n if n else None,
            "actual_win_rate": win_sum / n if n else None,
            "n": n,
        }
    return out


# --------------------------------------------------------------------------- #
# Betting simulation
# --------------------------------------------------------------------------- #


def _market_home_prob(pred: dict) -> float:
    value = pred.get("market_home_prob")
    if value is None:
        return DEFAULT_MARKET_HOME_PROB
    return float(value)


def _kelly_stake(model_prob: float, decimal_odds: float, fraction: float) -> float:
    """Fractional Kelly stake as a share of bankroll, clamped to [0, 1]."""
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    full = (b * model_prob - (1.0 - model_prob)) / b
    return min(max(full * fraction, 0.0), 1.0)


def simulate_roi(
    predictions: list[dict],
    market_vig: float = 0.045,
    min_edge: float = 0.03,
    kelly_fraction: float = 0.25,
) -> dict:
    """Simulate flat-bet and fractional-Kelly ROI.

    market_vig: the sportsbook take (reduces implied probability).
    min_edge: only bet when model_prob - implied_prob > min_edge.
    Returns {'flat_bet_roi', 'kelly_roi', 'bets_placed', 'total_games'}.

    The market is synthetic: each row's fair home probability comes from
    ``market_home_prob`` (or ``DEFAULT_MARKET_HOME_PROB``), the book applies
    the vig proportionally to both sides, and decimal odds are the inverse of
    the vigged implied probability. Each game is considered on both sides and
    at most one side is bet. ROI is total profit divided by total amount
    staked; with no bets placed both ROIs are 0.0.
    """
    if market_vig < 0:
        raise ValueError("market_vig must be >= 0")
    if not 0 < kelly_fraction <= 1:
        raise ValueError("kelly_fraction must be in (0, 1]")

    flat_staked = flat_profit = 0.0
    kelly_staked = kelly_profit = 0.0
    bets_placed = 0

    for pred in predictions:
        model_home = _prob(pred)
        fair_home = _market_home_prob(pred)
        home_won = bool(pred["actual_home_win"])

        # Candidate sides: (model_prob, implied_prob, side_won)
        sides = (
            (model_home, fair_home * (1.0 + market_vig), home_won),
            (1.0 - model_home, (1.0 - fair_home) * (1.0 + market_vig), not home_won),
        )
        best = None
        for model_p, implied_p, won in sides:
            edge = model_p - implied_p
            if edge > min_edge and (best is None or edge > best[0]):
                best = (edge, model_p, implied_p, won)
        if best is None:
            continue

        _, model_p, implied_p, won = best
        implied_p = min(max(implied_p, _EPS), 1.0)
        decimal_odds = 1.0 / implied_p
        payoff = (decimal_odds - 1.0) if won else -1.0  # per $1 staked

        bets_placed += 1
        flat_staked += 1.0
        flat_profit += payoff

        stake = _kelly_stake(model_p, decimal_odds, kelly_fraction)
        kelly_staked += stake
        kelly_profit += stake * payoff

    return {
        "flat_bet_roi": flat_profit / flat_staked if flat_staked else 0.0,
        "kelly_roi": kelly_profit / kelly_staked if kelly_staked else 0.0,
        "bets_placed": bets_placed,
        "total_games": len(predictions),
    }


# --------------------------------------------------------------------------- #
# Walk-forward fold
# --------------------------------------------------------------------------- #

_TEST_GAMES_SQL = """
SELECT
    g.game_pk,
    g.official_date,
    g.home_team_id,
    g.away_team_id,
    g.status,
    g.winning_team_id,
    g.home_score,
    g.away_score,
    htg.win AS home_win
FROM fact_game AS g
LEFT JOIN fact_team_game AS htg
    ON htg.game_pk = g.game_pk AND htg.team_id = g.home_team_id
WHERE g.season = ?
ORDER BY g.official_date, g.game_pk
"""


def _actual_home_win(row: dict) -> bool | None:
    """Resolve the home result from fact_team_game, then fact_game fallbacks."""
    if row["home_win"] is not None:
        return bool(row["home_win"])
    if row["winning_team_id"] is not None:
        return row["winning_team_id"] == row["home_team_id"]
    hs, as_ = row["home_score"], row["away_score"]
    if hs is not None and as_ is not None and hs != as_:
        return hs > as_
    return None


def _extract_home_win_prob(prediction: Any) -> float:
    """Pull home_win_prob from a GamePrediction-like object, dict, or number."""
    if isinstance(prediction, (int, float)):
        prob = float(prediction)
    elif isinstance(prediction, dict):
        prob = float(prediction["home_win_prob"])
    else:
        prob = float(getattr(prediction, "home_win_prob"))
    if not 0.0 <= prob <= 1.0:
        raise ValueError(f"home_win_prob out of range: {prob!r}")
    return prob


def run_backtest(
    con,
    predict_fn: Callable,
    train_seasons: list[int],
    test_season: int,
    market_vig: float = 0.045,
    min_edge: float = 0.03,
    kelly_fraction: float = 0.25,
) -> BacktestResult:
    """Run one walk-forward fold.

    - Calls predict_fn(con, game_pk, home_id, away_id, as_of_date) for each test game.
    - predict_fn must be a callable that returns a GamePrediction (anything with a
      ``home_win_prob`` attribute; a dict with that key or a bare float also works).
    - Scores results against actual outcomes from fact_game + fact_team_game.

    Games whose status is not final, or whose outcome cannot be resolved, are
    skipped and counted in ``BacktestResult.n_skipped``. ``as_of_date`` is the
    game's official date as an ISO string, so a well-behaved ``predict_fn``
    only sees information available before first pitch.
    """
    train_seasons = sorted(int(s) for s in train_seasons)
    if test_season in train_seasons:
        raise ValueError(f"test_season {test_season} must not appear in train_seasons")
    if any(s > test_season for s in train_seasons):
        raise ValueError("walk-forward requires every train season to precede test_season")

    cursor = con.execute(_TEST_GAMES_SQL, [test_season])
    columns = [c[0] for c in cursor.description]
    rows = [dict(zip(columns, r)) for r in cursor.fetchall()]

    predictions: list[dict] = []
    skipped = 0
    for row in rows:
        if row["status"] not in FINAL_STATUSES:
            skipped += 1
            continue
        actual = _actual_home_win(row)
        if actual is None:
            skipped += 1
            continue

        as_of_date = row["official_date"].isoformat() if row["official_date"] else None
        prediction = predict_fn(
            con, row["game_pk"], row["home_team_id"], row["away_team_id"], as_of_date
        )
        predictions.append(
            {
                "game_pk": row["game_pk"],
                "date": as_of_date,
                "home_id": row["home_team_id"],
                "away_id": row["away_team_id"],
                "home_win_prob": _extract_home_win_prob(prediction),
                "actual_home_win": actual,
            }
        )

    roi = simulate_roi(
        predictions,
        market_vig=market_vig,
        min_edge=min_edge,
        kelly_fraction=kelly_fraction,
    )
    return BacktestResult(
        predictions=predictions,
        brier_score=brier_score(predictions),
        log_loss=log_loss(predictions),
        calibration=calibration_buckets(predictions),
        roi_flat_bet=roi["flat_bet_roi"],
        roi_kelly=roi["kelly_roi"],
        test_season=test_season,
        train_seasons=train_seasons,
        n_games=len(predictions),
        n_skipped=skipped,
        bets_placed=roi["bets_placed"],
    )


__all__ = [
    "BacktestResult",
    "brier_score",
    "log_loss",
    "calibration_buckets",
    "simulate_roi",
    "run_backtest",
]
